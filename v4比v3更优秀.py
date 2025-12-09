import requests
import pandas as pd
import numpy as np

# ===========================
# 参数设置
# ===========================
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
START = "2024-01-01"
END   = "2024-03-01"

TARGET_UP = 0.02      # 目标涨幅 2%
HORIZON   = 24        # 未来 24 根 = 2 小时

ROLL_SLOPE_SHORT = 20
ROLL_SLOPE_LONG  = 50
ROLL_VOL         = 50
ROLL_ATR_MEAN    = 50
ROLL_SCORE_STATS = 500
ROLL_VCR_STATS   = 200  # 波动压缩统计窗口


# ===========================
# 获取 Binance K 线
# ===========================
def get_klines(symbol, interval, start, end):
    url = "https://fapi.binance.com/fapi/v1/klines"

    def to_ms(t):
        return int(pd.to_datetime(t, utc=True).timestamp()*1000)

    start_ms = to_ms(start)
    end_ms = to_ms(end)

    out = []
    s = start_ms

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": s,
            "endTime": end_ms,
            "limit": 1000
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            break

        out.extend(data)
        s = data[-1][0] + 1
        if s >= end_ms:
            break

    cols = ["open_time","open","high","low","close","volume",
            "close_time","qa","n_trades","tbb","tbq","ignore"]

    df = pd.DataFrame(out, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")

    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)

    return df


# ===========================
# 技术指标（无未来函数）
# ===========================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    open_ = df["open"]

    # ATR
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=1).mean()
    df["atr_pct"] = df["atr"] / close

    # EMA 趋势参考
    df["ema_fast"]  = close.ewm(span=20, adjust=False).mean()
    df["ema_slow"]  = close.ewm(span=60, adjust=False).mean()
    df["ema_trend"] = close.ewm(span=96, adjust=False).mean()

    # 斜率
    def calc_slope(x):
        y = np.array(x, dtype=float)
        if len(y) < 2:
            return np.nan
        return np.polyfit(range(len(y)), y, 1)[0]

    df["slope_20"] = close.rolling(ROLL_SLOPE_SHORT).apply(calc_slope, raw=True)
    df["slope_50"] = close.rolling(ROLL_SLOPE_LONG).apply(calc_slope, raw=True)

    # 斜率加速度（二阶）
    df["accel_3"] = df["slope_20"] - df["slope_20"].shift(3)

    # 成交量 z-score
    df["vol_ma"] = df["volume"].rolling(ROLL_VOL, min_periods=1).mean()
    df["vol_z"]  = (df["volume"] - df["vol_ma"]) / (df["vol_ma"] + 1e-9)

    # ATR 百分比 z-score
    df["atr_pct_ma"] = df["atr_pct"].rolling(ROLL_ATR_MEAN, min_periods=1).mean()
    df["atr_pct_z"]  = (df["atr_pct"] - df["atr_pct_ma"]) / (df["atr_pct_ma"] + 1e-9)

    # 实体/影线比例
    rng  = (high - low).replace(0, np.nan)
    body = (close - open_).abs()
    df["body_pct"] = (body / rng).clip(0, 1)

    # 短期收益率
    df["ret_3"] = close / close.shift(3) - 1.0

    # 上一根高点
    df["prev_high"] = high.shift(1)

    # 订单流代理：收盘位置（买方压力）
    df["buy_pressure"] = ((close - low) / (high - low + 1e-9)).clip(0, 1)

    # 不平衡大阳线标记
    df["imbalance_bar"] = (
        (df["body_pct"] > 0.6) &
        (df["buy_pressure"] > 0.75) &
        (close > prev_close)
    )

    # 波动压缩比：VCR
    df["vcr"] = df["atr_pct"] / (df["atr_pct_ma"] + 1e-9)

    # VCR 的动态阈值（历史分位）
    vcr_series = df["vcr"].replace([np.inf, -np.inf], np.nan)
    df["vcr_q25"] = vcr_series.rolling(ROLL_VCR_STATS, min_periods=50).quantile(0.25)

    return df


# ===========================
# V7-Combo 信号生成
# ===========================
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    # -------- 1. 大趋势过滤 --------
    base_trend_ok = (
        (df["slope_50"] > 0) &
        (df["close"] > df["ema_trend"]) &
        (df["ema_fast"] > df["ema_slow"])
    )

    # -------- 2. 特征标准化 --------
    df["norm_slope"] = (df["slope_20"] / df["close"]).clip(-0.01, 0.01) * 1000
    vol_z = df["vol_z"].clip(-3, 5)
    atr_z = df["atr_pct_z"].clip(-3, 5)
    body  = df["body_pct"].fillna(0)
    ret3  = df["ret_3"].clip(-0.03, 0.06)
    accel = df["accel_3"].fillna(0)

    # -------- 3. 深度动能质量得分 --------
    # 比之前更强调加速度 & buy_pressure
    quality_score = (
        1.0 * df["norm_slope"] +
        1.4 * body*100 +
        0.9 * vol_z*10 +
        0.8 * atr_z*10 +
        1.0 * ret3*100 +
        1.2 * accel*1000 +
        1.0 * df["buy_pressure"]*50
    )
    df["quality_score"] = quality_score

    score_mean = df["quality_score"].rolling(ROLL_SCORE_STATS, min_periods=150).mean()
    score_std  = df["quality_score"].rolling(ROLL_SCORE_STATS, min_periods=150).std()
    high_quality = df["quality_score"] > (score_mean + 1.0 * score_std)

    # -------- 4. 加速度/结构条件 --------
    slope_accel = (
        (df["slope_20"] > df["slope_50"]) &
        (df["slope_20"] > df["slope_20"].shift(1)) &
        (df["accel_3"] > 0)
    )

    # -------- 5. 强爆发通道（不平衡大阳 + 放量 + 突破） --------
    strong_imbalance = (
        df["imbalance_bar"] &
        (df["vol_z"] > 1.0) &
        (df["close"] > df["prev_high"])
    )

    # -------- 6. 波动压缩 + 爆发通道 --------
    vcr_ok = df["vcr"] < df["vcr_q25"]  # 当前波动低于历史 25% 分位（压缩）
    compression_break = (
        vcr_ok &
        (df["close"] > df["prev_high"]) &
        (df["vol_z"] > 0.5) &
        (df["body_pct"] > 0.4)
    )

    # -------- 7. 通道组合 --------
    # 通道 1：趋势 + 结构 + 高质量
    ch_trend_quality = base_trend_ok & slope_accel & high_quality

    # 通道 2：趋势 + 压缩爆发 + 质量一般也可以
    ch_compression = base_trend_ok & compression_break

    # 通道 3：趋势 + 订单流不平衡大阳线
    ch_imbalance = base_trend_ok & strong_imbalance

    final_signal = ch_trend_quality | ch_compression | ch_imbalance

    df["signal"] = final_signal.astype(int)
    return df


# ===========================
# 未来 2% 涨幅标签（评估用）
# ===========================
def label_future(df: pd.DataFrame) -> pd.DataFrame:
    closes = df["close"].values
    highs  = df["high"].values
    n = len(df)

    hit = np.zeros(n, dtype=bool)
    t2p = np.full(n, np.nan)

    for i in range(n - 1):
        entry = closes[i]
        target = entry * (1 + TARGET_UP)
        start = i + 1
        end   = min(i + 1 + HORIZON, n)
        if start >= n:
            break

        future_highs = highs[start:end]
        if future_highs.size == 0:
            continue

        idx = np.where(future_highs >= target)[0]
        if idx.size > 0:
            hit[i] = True
            t2p[i] = idx[0] + 1

    df["hit_2pct"] = hit
    df["time_to_2pct"] = t2p
    return df


# ===========================
# 构造开仓记录
# ===========================
def build_trades(df: pd.DataFrame) -> pd.DataFrame:
    entries = df[df["signal"] == 1]
    trades = []

    for t, row in entries.iterrows():
        trades.append({
            "entry_time": t,
            "entry_price": row["close"],
            "hit_2pct": bool(row["hit_2pct"]),
            "time_to_2pct_bars": row["time_to_2pct"],
            "time_to_2pct_minutes": None if pd.isna(row["time_to_2pct"]) else row["time_to_2pct"] * 5,
            "quality_score": row["quality_score"],
            "buy_pressure": row["buy_pressure"],
            "vcr": row["vcr"],
        })

    return pd.DataFrame(trades)


# ===========================
# 评估
# ===========================
def evaluate(df: pd.DataFrame, trades: pd.DataFrame):
    total_opportunities = int(df["hit_2pct"].sum())
    total_signals = len(trades)
    total_hits = int(trades["hit_2pct"].sum())

    precision = total_hits / total_signals if total_signals > 0 else np.nan
    recall    = total_hits / total_opportunities if total_opportunities > 0 else np.nan

    total_bars = len(df)
    baseline = total_opportunities / total_bars if total_bars > 0 else np.nan

    print("\n=== 历史 2% 涨幅机会 ===")
    print("总机会数:", total_opportunities)

    print("\n=== 策略捕捉结果（V7-Combo 深度动能+订单流代理版） ===")
    print("开仓次数:", total_signals)
    print("命中次数:", total_hits)
    print("Precision:", "NaN" if np.isnan(precision) else f"{precision:.2%}")
    print("Recall:", "NaN" if np.isnan(recall) else f"{recall:.2%}")

    print("\n=== 随机基准 ===")
    print("机会密度:", "NaN" if np.isnan(baseline) else f"{baseline:.2%}")
    edge = precision / baseline if (baseline and not np.isnan(precision)) else np.nan
    print("统计优势（edge）:", "NaN" if np.isnan(edge) else f"{edge:.2f}x")

    if not trades.empty:
        print("\n=== 成功样本（前 10 条） ===")
        print(trades[trades["hit_2pct"]].head(10))

        print("\n=== 失败样本（前 10 条） ===")
        print(trades[~trades["hit_2pct"]].head(10))


# ===========================
# 主程序
# ===========================
def main():
    print("下载数据中...")
    df = get_klines(SYMBOL, INTERVAL, START, END)
    print("K线数量:", len(df))

    df = add_indicators(df)
    df = generate_signals(df)
    df = label_future(df)

    trades = build_trades(df)
    evaluate(df, trades)

if __name__ == "__main__":
    main()
