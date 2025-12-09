import numpy as np
import pandas as pd
import requests
import time
import os
from dataclasses import dataclass

# =========================================
# Binance 永续合约 K线下载
# =========================================
def get_binance_klines(symbol, interval, start, end, limit=1000):
    url = "https://fapi.binance.com/fapi/v1/klines"

    def to_ms(t):
        return int(pd.to_datetime(t, utc=True).timestamp() * 1000)

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
            "limit": limit,
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            raise ValueError(resp.text)

        data = resp.json()
        if not data:
            break

        out.extend(data)
        last_ts = data[-1][0]
        s = last_ts + 1

        if s >= end_ms:
            break

        time.sleep(0.05)

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "_1", "_2", "_3", "_4", "_5", "_6"
    ]
    df = pd.DataFrame(out, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")
    return df[["open", "high", "low", "close", "volume"]].astype(float)


# =========================================
# BTC 5m 数据（本地缓存）
# =========================================
def load_or_fetch_btc_5m(start, end, filename="btc5m_data.csv"):
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)

    if os.path.exists(filename):
        df = pd.read_csv(filename, parse_dates=["open_time"])
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.set_index("open_time")
        df = df[["open", "high", "low", "close", "volume"]].astype(float)

        if df.index.min() <= start_ts and df.index.max() >= end_ts:
            print("使用本地缓存 BTC 5m 数据")
            return df.loc[start_ts:end_ts]

        print("缓存不足，重新下载...")

    df = get_binance_klines("BTCUSDT", "5m", start, end)
    df2 = df.copy().reset_index()
    df2.to_csv(filename, index=False)
    print(f"已保存 BTC 5m 数据到本地: {filename}")
    return df


# =========================================
# 15m KAMA 计算（自适应均线）
# =========================================
def compute_kama(close, er_window=10, fast=2, slow=30):
    close = close.astype(float)
    change = close.diff(er_window).abs()
    volatility = close.diff().abs().rolling(er_window).sum()
    er = change / volatility.replace(0, np.nan)

    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = pd.Series(index=close.index, dtype=float)
    first_valid = er_window
    kama.iloc[first_valid] = close.iloc[first_valid]

    for i in range(first_valid + 1, len(close)):
        if np.isnan(sc.iloc[i]) or np.isnan(kama.iloc[i-1]):
            kama.iloc[i] = kama.iloc[i-1]
        else:
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])

    return kama


# =========================================
# 15m ADX 计算（趋势强度）
# =========================================
def compute_adx_15m(high, low, close, period=14):
    high_shift = high.shift(1)
    low_shift = low.shift(1)
    close_shift = close.shift(1)

    up_move = high - high_shift
    down_move = low_shift - low

    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    tr1 = high - low
    tr2 = (high - close_shift).abs()
    tr3 = (low - close_shift).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period).mean()

    return adx, plus_di, minus_di


# =========================================
# 15m 多因子趋势评分 TS15 + 动量 + ADX
# =========================================
def compute_TS15_trend(df15, cfg):
    close = df15["close"]
    high = df15["high"]
    low = df15["low"]

    # 1) 动量（幅度）
    mom_win = cfg["mom_window"]
    mom = close / close.shift(mom_win) - 1

    # 2) EMA 斜率（快 - 慢）
    ema_fast = close.ewm(span=cfg["ema_fast"], adjust=False).mean()
    ema_slow = close.ewm(span=cfg["ema_slow"], adjust=False).mean()
    ema_slope = ema_fast - ema_slow

    # 3) KAMA 斜率
    kama = compute_kama(close, er_window=cfg["kama_er"],
                        fast=cfg["kama_fast"], slow=cfg["kama_slow"])
    kama_slope = kama.diff()

    # 4) ADX（趋势强度）
    adx, plus_di, minus_di = compute_adx_15m(high, low, close,
                                             period=cfg["adx_period"])

    # 标准化（滚动 z-score，避免某个因子爆炸）
    def zscore(x, win):
        mean = x.rolling(win).mean()
        std = x.rolling(win).std()
        return (x - mean) / (std + 1e-9)

    mom_z = zscore(mom, cfg["zwin"])
    ema_z = zscore(ema_slope, cfg["zwin"])
    kama_z = zscore(kama_slope, cfg["zwin"])

    TS = (
        cfg["w_mom"] * mom_z +
        cfg["w_ema"] * ema_z +
        cfg["w_kama"] * kama_z
    )

    out = pd.DataFrame(index=df15.index)
    out["TS"] = TS
    out["mom"] = mom
    out["ema_slope"] = ema_slope
    out["kama_slope"] = kama_slope
    out["ADX"] = adx
    return out


# =========================================
# 5m ATR（用于突破强度过滤 & 止损）
# =========================================
def compute_atr_5m(df5, window=20):
    high = df5["high"]
    low = df5["low"]
    close = df5["close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr


# =========================================
# Trade 结构体
# =========================================
@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry_price: float
    exit_price: float
    qty: float
    profit: float
    return_pct: float
    reason: str
    bars_held: int


# =========================================
# 回测逻辑：15m 小趋势 + 5m 突破执行（带 ADX + 动量幅度过滤）
# =========================================
def backtest_15m_trend(df5, df15, cfg):
    # 15m 结构（过去 N 根高低点）
    rh15 = df15["high"].rolling(cfg["struct_window"]).max().shift(1)
    rl15 = df15["low"].rolling(cfg["struct_window"]).min().shift(1)

    # 15m 多因子趋势 + 动量幅度 + ADX
    ts15 = compute_TS15_trend(df15, cfg)
    TS_prev = ts15["TS"].shift(1)
    mom_prev = ts15["mom"].shift(1)
    adx_prev = ts15["ADX"].shift(1)

    # 映射到 5m 时间轴（全部用前一根15m的值 → 无未来）
    df5["RH15"] = rh15.reindex(df5.index, method="ffill")
    df5["RL15"] = rl15.reindex(df5.index, method="ffill")
    df5["TS15"] = TS_prev.reindex(df5.index, method="ffill")
    df5["MOM15"] = mom_prev.reindex(df5.index, method="ffill")
    df5["ADX15"] = adx_prev.reindex(df5.index, method="ffill")

    # 5m ATR
    df5["ATR"] = compute_atr_5m(df5, window=cfg["atr_window"])

    df5 = df5.dropna(subset=["RH15", "RL15", "TS15", "MOM15", "ADX15", "ATR"])

    equity = cfg["capital"]
    max_equity = equity
    trades = []
    drawdowns = []

    holding = False
    entry_price = 0.0
    entry_time = None
    entry_idx = None
    qty = 0.0
    side = None
    stop_price = None

    idx = df5.index.to_list()

    # TS 持续性计数（在 5m 上看 TS 是否连续满足）
    ts_long_run = 0
    ts_short_run = 0

    for i in range(1, len(idx)):
        t_prev = idx[i - 1]
        t_curr = idx[i]

        row_p = df5.loc[t_prev]
        row_c = df5.loc[t_curr]

        ts15_prev = row_p["TS15"]
        mom15_prev = row_p["MOM15"]
        adx15_prev = row_p["ADX15"]
        rh = row_p["RH15"]
        rl = row_p["RL15"]
        close_p = row_p["close"]
        atr_p = row_p["ATR"]

        # === 15m 小趋势三重过滤（方向 + 幅度 + 强度） ===
        long_trend_bar = (
            (ts15_prev > cfg["ts_long_th"]) &
            (mom15_prev >= cfg["mom_amp_th"]) &
            (adx15_prev >= cfg["adx_trend_th"])
        )
        short_trend_bar = (
            (ts15_prev < cfg["ts_short_th"]) &
            (mom15_prev <= -cfg["mom_amp_th"]) &
            (adx15_prev >= cfg["adx_trend_th"])
        )

        # 更新持续性计数（在 5m 上统计连续满足的小趋势 bar）
        if long_trend_bar:
            ts_long_run += 1
        else:
            ts_long_run = 0

        if short_trend_bar:
            ts_short_run += 1
        else:
            ts_short_run = 0

        # ===== 平仓逻辑 =====
        if holding:
            exit_flag = False
            reason = ""

            # 1）ATR 硬止损
            if side == "long" and close_p <= stop_price:
                exit_flag = True
                reason = "hard_stop"
            elif side == "short" and close_p >= stop_price:
                exit_flag = True
                reason = "hard_stop"

            # 2）15m 趋势结束（TS + ADX 弱化）
            if not exit_flag:
                if side == "long":
                    if (ts15_prev <= cfg["ts_exit_th"]) or (adx15_prev < cfg["adx_exit_th"]):
                        exit_flag = True
                        reason = "ts_or_adx_exit"
                elif side == "short":
                    if (ts15_prev >= -cfg["ts_exit_th"]) or (adx15_prev < cfg["adx_exit_th"]):
                        exit_flag = True
                        reason = "ts_or_adx_exit"

            if exit_flag:
                exit_price = row_c["open"]
                if side == "long":
                    gross = (exit_price - entry_price) * qty
                else:
                    gross = (entry_price - exit_price) * qty

                fee = cfg["fee"] * (entry_price * qty + exit_price * qty)
                net = gross - fee

                equity += net
                max_equity = max(max_equity, equity)
                drawdowns.append(1 - equity / max_equity)

                bars_held = i - entry_idx

                trades.append(Trade(
                    entry_time, t_curr, side,
                    entry_price, exit_price, qty,
                    net, net / cfg["capital"], reason, bars_held
                ))

                holding = False
                side = None
                stop_price = None
                continue

        # ===== 开仓逻辑 =====
        if not holding:
            # 15m 小趋势持续性过滤
            long_trend_ok = (ts_long_run >= cfg["ts_run_min"])
            short_trend_ok = (ts_short_run >= cfg["ts_run_min"])

            # 5m 结构突破（针对15m区间）+ ATR 突破强度
            long_break = (
                long_trend_ok and
                (close_p > rh) and
                ((close_p - rh) > cfg["atr_k"] * atr_p)
            )
            short_break = (
                cfg["enable_short"] and
                short_trend_ok and
                (close_p < rl) and
                ((rl - close_p) > cfg["atr_k"] * atr_p)
            )

            if long_break:
                side = "long"
                entry_time = t_curr
                entry_idx = i
                entry_price = row_c["open"]
                qty = cfg["margin"] * cfg["leverage"] / entry_price

                stop_price = entry_price - cfg["stop_atr_mult"] * atr_p

                holding = True

            elif short_break:
                side = "short"
                entry_time = t_curr
                entry_idx = i
                entry_price = row_c["open"]
                qty = cfg["margin"] * cfg["leverage"] / entry_price

                stop_price = entry_price + cfg["stop_atr_mult"] * atr_p

                holding = True

    return trades, equity, drawdowns


# =========================================
# 统计函数
# =========================================
def analyze_trades(trades, capital):
    longs = [t for t in trades if t.side == "long"]
    shorts = [t for t in trades if t.side == "short"]

    def stats(ts):
        if not ts:
            return dict(
                n=0, wins=0, win_rate=0.0,
                avg_pnl=0.0, med_pnl=0.0,
                total_ret=0.0, pf=0.0,
                avg_bars=0.0
            )
        profits = np.array([t.profit for t in ts])
        rets = np.array([t.return_pct for t in ts])
        pos = profits[profits > 0]
        neg = profits[profits < 0]
        wins = len(pos)
        n = len(ts)
        win_rate = wins / n * 100
        avg_pnl = profits.mean()
        med_pnl = np.median(profits)
        total_ret = rets.sum() * 100
        pf = pos.sum() / abs(neg.sum()) if len(neg) > 0 else np.inf
        avg_bars = np.mean([t.bars_held for t in ts])
        return dict(
            n=n, wins=wins, win_rate=win_rate,
            avg_pnl=avg_pnl, med_pnl=med_pnl,
            total_ret=total_ret, pf=pf,
            avg_bars=avg_bars
        )

    long_s = stats(longs)
    short_s = stats(shorts)
    total_s = stats(trades)

    return long_s, short_s, total_s


# =========================================
# 主程序：抓 15m 小趋势（动量幅度 + ADX + 多因子）
# =========================================
if __name__ == "__main__":
    START = "2022-01-01"
    END = "2024-06-30"

    df5 = load_or_fetch_btc_5m(START, END)

    # 15m 重采样（用 '15min' 避免 FutureWarning）
    df15 = df5.resample("15min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()

    cfg = {
        "capital": 1000,
        "margin": 300,
        "leverage": 3,
        "fee": 0.0004,

        # 15m 结构窗口（多少根15m构成RH/RL）
        "struct_window": 8,     # 8 根15m ≈ 2小时区间

        # 15m 多因子参数
        "mom_window": 24,       # 动量：24根15m ≈ 6小时
        "ema_fast": 8,
        "ema_slow": 21,
        "kama_er": 10,
        "kama_fast": 2,
        "kama_slow": 30,
        "zwin": 48,             # z-score 窗口

        # 因子权重
        "w_mom": 0.5,
        "w_ema": 0.25,
        "w_kama": 0.25,

        # ADX 参数（15m 趋势强度）
        "adx_period": 14,
        "adx_trend_th": 18,     # ADX ≥ 18 才认为有趋势环境
        "adx_exit_th": 14,      # ADX < 14 认为趋势衰退

        # 动量幅度过滤
        "mom_amp_th": 0.015,    # 过去6小时涨幅 ≥ 1.5%

        # TS 阈值（方向）
        "ts_long_th": 0.0,      # TS > 0 认为多头趋势方向
        "ts_short_th": 0.0,
        "ts_exit_th": 0.0,      # TS 回落到0附近则认为趋势方向消失
        "ts_run_min": 2,        # 小趋势至少连续 2 根 5m

        # 5m ATR & 突破参数
        "atr_window": 20,
        "atr_k": 0.8,           # 从 0.6 提升到 0.8，减少假突破

        # ATR 止损倍数
        "stop_atr_mult": 2.0,

        # 是否启用空头（先关，专注多头小趋势）
        "enable_short": False,
    }

    print("运行回测（15m 小趋势 + 多因子 + ADX + 动量幅度 + 5m 执行，纯多头）...")
    trades, equity, dds = backtest_15m_trend(df5, df15, cfg)

    long_s, short_s, total_s = analyze_trades(trades, cfg["capital"])

    print("\n===== 回测结果（整体） =====")
    print(f"初始资金: {cfg['capital']:.2f}")
    print(f"结束资金: {equity:.2f}")
    print(f"总收益率: {(equity / cfg['capital'] - 1)*100:.2f}%")
    print(f"交易次数: {total_s['n']}")
    if dds:
        print(f"最大回撤: {max(dds)*100:.2f}%")
    print(f"整体胜率: {total_s['win_rate']:.2f}%")
    print(f"整体 Profit Factor: {total_s['pf']:.2f}")
    print(f"平均每笔PNL: {total_s['avg_pnl']:.4f}, 中位数PNL: {total_s['med_pnl']:.4f}")
    print(f"平均持仓bar数: {total_s['avg_bars']:.2f}")

    print("\n===== 多头统计 =====")
    print(f"多头交易数: {long_s['n']}, 胜率: {long_s['win_rate']:.2f}%, PF: {long_s['pf']:.2f}")
    print(f"多头总收益率: {long_s['total_ret']:.2f}%")
    print(f"多头平均PNL: {long_s['avg_pnl']:.4f}, 中位PNL: {long_s['med_pnl']:.4f}")
    print(f"多头平均持仓bar数: {long_s['avg_bars']:.2f}")

    print("\n===== 空头统计 =====")
    print(f"空头交易数: {short_s['n']}, 胜率: {short_s['win_rate']:.2f}%, PF: {short_s['pf']:.2f}")
    print(f"空头总收益率: {short_s['total_ret']:.2f}%")
    print(f"空头平均PNL: {short_s['avg_pnl']:.4f}, 中位PNL: {short_s['med_pnl']:.4f}")
    print(f"空头平均持仓bar数: {short_s['avg_bars']:.2f}")

    print("\n前 5 笔交易：")
    for t in trades[:5]:
        print(t)
