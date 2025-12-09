import pandas as pd
import numpy as np
import requests
import time
from dataclasses import dataclass, asdict
import math


# =====================================================
# 配置参数
# =====================================================
INIT_EQUITY = 1000.0
TRADE_NOTIONAL = 300.0
LEVERAGE = 5.0
FEE_RATE = 0.0004

# 回测时间段（UTC）
BACKTEST_START = "2024-01-01 00:00:00"
BACKTEST_END   = "2024-06-30 23:55:00"

# 趋势因子阈值
TIS_LONG_TH  = 0.0005
TIS_SHORT_TH = -0.0005

# 止盈止损
TP_PCT = 0.006      # 0.6%
SL_PCT = 0.0015     # 0.15%

SYMBOL = "BTCUSDT"


# =====================================================
# 1. 全自动 Binance 5m K线获取（含 taker & 自动OI）
# =====================================================
def fetch_binance_5m_full(symbol: str, start_ts, end_ts):
    """
    从 Binance 拉取 5m K线：
    open, high, low, close, volume, takerBuyBaseVolume
    并为每一根生成 OI（实时获取）
    """
    url = "https://fapi.binance.com/fapi/v1/klines"

    ms_start = int(start_ts.timestamp() * 1000)
    ms_end = int(end_ts.timestamp() * 1000)

    all_rows = []
    limit = 1500

    print("=== 正在从 Binance 获取 5m K线（含成交量）===")

    while True:
        params = {
            "symbol": symbol,
            "interval": "5m",
            "startTime": ms_start,
            "endTime": ms_end,
            "limit": limit
        }

        try:
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"K线请求失败: {e}, 重试中...")
            time.sleep(1)
            continue

        if not data:
            break

        all_rows.extend(data)

        last_open = data[-1][0]
        ms_start = last_open + 5 * 60 * 1000

        if ms_start >= ms_end:
            break

        time.sleep(0.1)

    # 解析
    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","num_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ]

    df = pd.DataFrame(all_rows, columns=cols)
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("time")

    float_cols = ["open","high","low","close","volume","taker_buy_base"]
    df[float_cols] = df[float_cols].astype(float)

    # =====================================================
    # 获取 OI（通过 openInterest 接口轮询构造 5m 序列）
    # =====================================================
    print("=== 正在获取持仓量 OI ===")
    oi_list = []
    oi_url = "https://fapi.binance.com/fapi/v1/openInterest"

    for _ in range(len(df)):
        try:
            resp = requests.get(oi_url, params={"symbol": symbol}, timeout=5)
            resp.raise_for_status()
            oi_list.append(float(resp.json()["openInterest"]))
        except:
            oi_list.append(np.nan)
        time.sleep(0.05)

    df["oi"] = pd.Series(oi_list, index=df.index).ffill()

    return df


# =====================================================
# 2. 多周期（15m & 1h）
# =====================================================
def build_multi_tf(df_5m):
    df_15m = df_5m.resample("15T", label="right").agg({
        "open":"first","high":"max","low":"min","close":"last",
        "volume":"sum","oi":"last"
    }).dropna()

    df_1h = df_5m.resample("1H", label="right").agg({
        "open":"first","high":"max","low":"min","close":"last",
        "volume":"sum","oi":"last"
    }).dropna()

    return df_15m, df_1h


# =====================================================
# 3. 机构级趋势因子
# =====================================================

# --- 因子1：OI 加速度 ---
def add_oi_accel(df):
    df = df.copy()
    df["d_oi"] = df["oi"].diff()
    df["dd_oi"] = df["d_oi"].diff()

    base = df["oi"].shift(2).abs() + 1e-9
    df["OIA"] = df["dd_oi"] / base

    df["vol_ma6"] = df["volume"].rolling(6).mean()
    df["OIA_adj"] = df["OIA"] * (df["volume"] / (df["vol_ma6"] + 1e-9))
    df["OIA_adj"] = df["OIA_adj"].replace([np.inf,-np.inf],0).fillna(0)
    return df


# --- 因子2：成交簇点火 DCI ---
def add_dci(df):
    df = df.copy()
    taker_buy = df["taker_buy_base"]
    taker_sell = df["volume"] - taker_buy

    agg_ratio = (taker_buy - taker_sell).abs() / (df["volume"] + 1e-9)
    micro_jump = (df["close"] - df["open"]) / (df["open"] + 1e-9)

    df["DCI"] = agg_ratio * micro_jump
    df["DCI"] = df["DCI"].fillna(0)
    return df


# --- 因子3：结构突破 SBS ---
def add_sbs(df, win=3):
    df = df.copy()
    swing_high = df["high"].shift(1).rolling(win).max()
    swing_low  = df["low"].shift(1).rolling(win).min()

    rng = (df["high"] - df["low"]).abs() + 1e-9
    body_pct = (df["close"] - df["open"]).abs() / rng

    sbs_long  = ((df["close"] - swing_high) / (swing_high + 1e-9)) * body_pct
    sbs_short = ((swing_low - df["close"]) / (swing_low + 1e-9)) * body_pct

    df["SBS"] = np.where(
        df["close"] > swing_high, sbs_long,
        np.where(df["close"] < swing_low, sbs_short, 0)
    )

    df["SBS"] = df["SBS"].fillna(0)
    return df


# --- 单根振幅 ---
def add_bar_vol(df):
    df = df.copy()
    df["bar_volatility"] = (df["high"] - df["low"]) / (df["open"] + 1e-9)
    return df


# --- 1h 趋势过滤器 ---
def add_1h_features(df_5m, df_1h):
    df_1h = df_1h.copy()
    df_1h["ma20"] = df_1h["close"].rolling(20).mean()
    df_1h["slope"] = df_1h["ma20"].diff()

    feat = df_1h[["ma20","slope"]].reindex(df_5m.index, method="ffill")
    return df_5m.join(feat)


# --- 综合趋势启动评分 TIS ---
def add_TIS(df):
    df = df.copy()
    df["TIS"] = (
        0.4 * df["OIA_adj"] +
        0.35 * df["DCI"] +
        0.25 * df["SBS"]
    )
    return df


# =====================================================
# 4. 回测引擎（无未来函数）
# =====================================================
@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_after_fee: float
    reason: str


def backtest(df):
    equity = INIT_EQUITY
    trades = []

    pos = None
    entry_price = 0
    qty = 0
    entry_time = None

    cooldown_until = None
    consec_loss = 0

    idx = df.index

    for i in range(len(df)-1):
        now = idx[i]
        nxt = idx[i+1]
        row = df.iloc[i]
        next_open = df.iloc[i+1]["open"]

        # 冷静期
        if cooldown_until and now < cooldown_until:
            allow_entry = False
        else:
            allow_entry = True

        # === 有持仓：检查平仓 ===
        if pos is not None:
            high = row["high"]
            low  = row["low"]

            tp = entry_price * (1+TP_PCT if pos=="long" else 1-TP_PCT)
            sl = entry_price * (1-SL_PCT if pos=="long" else 1+SL_PCT)

            exit_flag = False
            exit_px = None
            reason = ""

            # intrabar 先 SL
            if pos=="long":
                if low<=sl:
                    exit_px=sl; exit_flag=True; reason="SL"
                elif high>=tp:
                    exit_px=tp; exit_flag=True; reason="TP"
            else:
                if high>=sl:
                    exit_px=sl; exit_flag=True; reason="SL"
                elif low<=tp:
                    exit_px=tp; exit_flag=True; reason="TP"

            # 趋势衰竭（TIS 转向）
            if not exit_flag:
                if (pos=="long" and row["TIS"]<0) or (pos=="short" and row["TIS"]>0):
                    exit_px = row["close"]
                    exit_flag=True
                    reason="TIS_reverse"

            if exit_flag:
                if pos=="long":
                    pnl = (exit_px-entry_price)*qty
                else:
                    pnl = (entry_price-exit_px)*abs(qty)

                fee = abs(qty)*entry_price * FEE_RATE * 2
                pnl_after_fee = pnl-fee

                equity += pnl_after_fee

                trades.append(Trade(entry_time, now, pos,
                                    entry_price, exit_px, qty,
                                    pnl, pnl_after_fee, reason))

                consec_loss = consec_loss+1 if pnl_after_fee<0 else 0
                if consec_loss>=3:
                    cooldown_until = now + pd.Timedelta(minutes=60)
                    consec_loss=0

                pos=None
                continue

        # === 无持仓：找开仓机会 ===
        if pos is None and allow_entry:
            trend_long  = (row["close"]>row["ma20"]) and (row["slope"]>0)
            trend_short = (row["close"]<row["ma20"]) and (row["slope"]<0)
            small_vol   = (row["bar_volatility"]<=0.002)

            long_sig  = trend_long  and small_vol and (row["TIS"]>TIS_LONG_TH)
            short_sig = trend_short and small_vol and (row["TIS"]<TIS_SHORT_TH)

            margin = TRADE_NOTIONAL/LEVERAGE
            if equity < margin*1.2:
                long_sig=False; short_sig=False

            if long_sig:
                pos="long"
                entry_price = next_open
                qty = TRADE_NOTIONAL/entry_price
                entry_time = nxt

            elif short_sig:
                pos="short"
                entry_price = next_open
                qty = -TRADE_NOTIONAL/entry_price
                entry_time = nxt

    return equity, trades


# =====================================================
# 5. 主程序
# =====================================================
def main():
    start_ts = pd.to_datetime(BACKTEST_START, utc=True)
    end_ts   = pd.to_datetime(BACKTEST_END, utc=True)

    # === 下载数据 ===
    df_5m = fetch_binance_5m_full(SYMBOL, start_ts, end_ts)

    # === 多周期 ===
    df_15m, df_1h = build_multi_tf(df_5m)

    # === 计算因子 ===
    df = df_5m.copy()
    df = add_oi_accel(df)
    df = add_dci(df)
    df = add_sbs(df)
    df = add_bar_vol(df)

    df = add_1h_features(df, df_1h)
    df = add_TIS(df)
    df = df.dropna()

    # === 回测 ===
    equity, trades = backtest(df)

    print("\n===== 回测结果 =====")
    print("初始资金:", INIT_EQUITY)
    print("结束资金:", equity)
    print("总收益:", equity - INIT_EQUITY)
    print("交易笔数:", len(trades))

    if trades:
        df_t = pd.DataFrame([asdict(t) for t in trades])
        win_rate = (df_t[df_t["pnl_after_fee"]>0].shape[0] / len(df_t)) * 100

        curve = INIT_EQUITY + df_t["pnl_after_fee"].cumsum()
        peak = curve.cummax()
        dd = (curve-peak)/peak
        max_dd = dd.min()*100

        print("胜率: %.2f%%" % win_rate)
        print("最大回撤: %.2f%%" % max_dd)


if __name__ == "__main__":
    main()
