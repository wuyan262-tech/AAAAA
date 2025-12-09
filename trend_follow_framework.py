"""
自学习趋势跟随策略系统
====================

本模块实现了在 BTCUSDT 永续合约上运行的趋势跟随框架，涵盖数据下载、特征生成、信号产生、回测、参数搜索以及伪实盘循环。
代码以函数化和模块化方式组织，方便扩展到其他币种或周期。
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1/klines"


# ============================== 数据模块 ==============================
class DataLoader:
    """负责与 Binance 期货接口交互，下载并增量更新 K 线数据。"""

    def __init__(self, base_url: str = BINANCE_FUTURES_URL):
        self.base_url = base_url

    @staticmethod
    def _ms(ts: dt.datetime) -> int:
        """将 datetime 转为毫秒时间戳。"""
        return int(ts.timestamp() * 1000)

    def get_klines(
        self, symbol: str, interval: str, start: dt.datetime, end: dt.datetime
    ) -> pd.DataFrame:
        """
        从 Binance 期货接口获取指定时间段的 K 线数据。

        参数:
            symbol: 交易对，例如 "BTCUSDT"。
            interval: 周期字符串，例如 "15m" 或 "1h"。
            start/end: 时间范围，闭区间，Binance 单次最多返回 1500 根。

        返回:
            DataFrame，包含 open_time, open, high, low, close, volume 列。
        """

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": self._ms(start),
            "endTime": self._ms(end),
            "limit": 1500,
        }
        all_rows: List[List] = []
        while True:
            resp = requests.get(self.base_url, params=params, timeout=10)
            resp.raise_for_status()
            rows = resp.json()
            if not rows:
                break
            all_rows.extend(rows)
            # 下一次从最后一根的 open_time + 1ms 开始，避免重复
            last_open_time = rows[-1][0]
            params["startTime"] = last_open_time + 1
            # Binance 对 endTime 是包含的，提前退出
            if last_open_time >= params["endTime"]:
                break
        df = pd.DataFrame(
            all_rows,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trade_num",
                "taker_base",
                "taker_quote",
                "ignore",
            ],
        )
        if df.empty:
            return df
        # 转换数据类型
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df = df[["open_time", "open", "high", "low", "close", "volume"]]
        return df

    def incremental_update(
        self,
        symbol: str,
        interval: str,
        history_path: str,
        start: dt.datetime,
    ) -> pd.DataFrame:
        """
        增量更新本地历史数据文件（csv）。

        - 如果文件不存在，则从 start 下载到当前 UTC 时间。
        - 如果已存在，则从文件最后一根 K 线的时间向后补齐到当前 UTC 时间。
        """

        now = dt.datetime.utcnow()
        if os.path.exists(history_path):
            history = pd.read_csv(history_path, parse_dates=["open_time"])
            last_time = history["open_time"].max()
            fetch_start = last_time + pd.Timedelta(milliseconds=1)
        else:
            history = pd.DataFrame()
            fetch_start = start
        if fetch_start >= now:
            return history
        new_data = self.get_klines(symbol, interval, fetch_start, now)
        if history.empty:
            updated = new_data
        else:
            updated = pd.concat([history, new_data], ignore_index=True)
            updated = updated.drop_duplicates(subset=["open_time"]).sort_values("open_time")
        updated.to_csv(history_path, index=False)
        return updated


# ============================== 特征与信号模块 ==============================
def compute_features(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """基于价格数据计算趋势跟随所需指标。"""

    out = df.copy()
    fast = int(params.get("fast_ema", 20))
    slow = int(params.get("slow_ema", 60))
    breakout = int(params.get("breakout_window", 50))
    atr_win = int(params.get("atr_window", 14))

    out["fast_ema"] = out["close"].ewm(span=fast, adjust=False).mean()
    out["slow_ema"] = out["close"].ewm(span=slow, adjust=False).mean()
    # 真实波幅 ATR
    high_low = out["high"] - out["low"]
    high_close = (out["high"] - out["close"].shift(1)).abs()
    low_close = (out["low"] - out["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["atr"] = tr.rolling(window=atr_win, min_periods=1).mean()
    out["ret"] = out["close"].pct_change()
    out["momentum"] = out["close"].pct_change(periods=breakout)
    out["volatility"] = out["ret"].rolling(atr_win).std()
    # 突破参考使用前一根 K 线之前的窗口，避免当根内部未来数据
    out["rolling_high"] = out["high"].shift(1).rolling(breakout).max()
    out["rolling_low"] = out["low"].shift(1).rolling(breakout).min()
    return out


def generate_signals(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    基于趋势条件生成交易信号。

    信号在每根 K 线收盘时确定，实际交易假设在下一根开盘成交，避免未来函数。
    """

    data = compute_features(df, params)
    vol_floor = float(params.get("vol_floor", 0.0005))
    vol_cap = float(params.get("vol_cap", 0.02))

    bullish_trend = data["fast_ema"] > data["slow_ema"]
    bearish_trend = data["fast_ema"] < data["slow_ema"]
    # 动能强度
    bullish_momo = data["momentum"] > 0
    bearish_momo = data["momentum"] < 0
    vol_ok = (data["volatility"] > vol_floor) & (data["volatility"] < vol_cap)

    long_break = data["close"] > data["rolling_high"]
    short_break = data["close"] < data["rolling_low"]

    long_signal = bullish_trend & bullish_momo & vol_ok & long_break
    short_signal = bearish_trend & bearish_momo & vol_ok & short_break

    signal = pd.Series(0, index=data.index)
    signal = signal.where(~long_signal, 1)
    signal = signal.where(~short_signal, -1)
    data["signal"] = signal
    data["exec_signal"] = data["signal"].astype(int)
    return data


# ============================== 回测模块 ==============================
@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    fee_rate: float = 0.0004  # 双边手续费
    slippage: float = 0.0002  # 滑点比例
    position_size: float = 1.0  # 每次名义仓位（单位：币），示例：1 BTC


def backtest(df: pd.DataFrame, params: Dict, config: BacktestConfig) -> Dict:
    """
    使用生成的 exec_signal 进行简化回测。

    成交规则：
    - 信号在 bar[t] 生成，订单在 bar[t+1] 的开盘价成交。
    - 允许多空反手，成交价格考虑滑点和手续费。
    - 止损：开仓时按 ATR * atr_stop_k 固定距离，触发时按当根极值价成交。
    """

    data = generate_signals(df, params).copy()
    data = data.dropna().reset_index(drop=True)
    if data.empty:
        return {}

    capital = config.initial_capital
    position = 0  # 持仓数量，正为多，负为空
    entry_price = 0.0
    stop_price: Optional[float] = None
    equity_curve = []
    trades: List[Dict] = []
    atr_k = float(params.get("atr_stop_k", 2.0))

    for i in range(len(data) - 1):
        row = data.loc[i]
        next_row = data.loc[i + 1]
        next_open = next_row["open"]
        signal = row["exec_signal"]

        # 持仓止损检查，使用当前 bar 的极值价格（不含未来 bar）
        if position > 0 and stop_price is not None and row["low"] <= stop_price:
            pnl = (stop_price - entry_price) * position
            fee = abs(stop_price * position) * config.fee_rate
            pnl -= fee
            capital += pnl
            trades.append({"exit_time": row["open_time"], "pnl": pnl, "direction": "long"})
            position = 0
            stop_price = None
        elif position < 0 and stop_price is not None and row["high"] >= stop_price:
            pnl = (entry_price - stop_price) * abs(position)
            fee = abs(stop_price * position) * config.fee_rate
            pnl -= fee
            capital += pnl
            trades.append({"exit_time": row["open_time"], "pnl": pnl, "direction": "short"})
            position = 0
            stop_price = None

        # 平旧开新（反手）
        if position != 0 and signal != np.sign(position):
            pnl = (next_open - entry_price) * position
            fee = abs(next_open * position) * config.fee_rate
            pnl -= fee
            capital += pnl
            trades.append(
                {
                    "exit_time": next_row["open_time"],
                    "pnl": pnl,
                    "direction": "long" if position > 0 else "short",
                }
            )
            position = 0
            stop_price = None

        if signal != 0 and position == 0:
            # 开新仓
            trade_price = next_open * (1 + config.slippage * np.sign(signal))
            position = signal * config.position_size
            entry_price = trade_price
            stop_price = entry_price - np.sign(position) * atr_k * row["atr"]
            # stop_price 逻辑：多仓 entry - atr*k，空仓 entry + atr*k
            fee = abs(trade_price * position) * config.fee_rate
            capital -= fee
        # 更新持仓浮盈
        if position != 0:
            current_pnl = (row["close"] - entry_price) * position
            equity_curve.append(capital + current_pnl)
        else:
            equity_curve.append(capital)

    equity = pd.Series(equity_curve, index=data.index[:-1])
    returns = equity.pct_change().fillna(0)
    total_return = equity.iloc[-1] / config.initial_capital - 1
    max_dd = ((equity.cummax() - equity) / equity.cummax()).max()
    sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252 * 24 * 4)

    result = {
        "equity_curve": equity,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "trades": trades,
    }
    return result


# ============================== 参数搜索模块 ==============================
def parameter_grid() -> Iterable[Dict]:
    """生成一组简单的网格搜索参数。"""

    for fast in [10, 20, 30]:
        for slow in [50, 60, 90]:
            if fast >= slow:
                continue
            for breakout in [30, 50, 70]:
                for atr_stop_k in [1.5, 2.0, 2.5]:
                    yield {
                        "fast_ema": fast,
                        "slow_ema": slow,
                        "breakout_window": breakout,
                        "atr_stop_k": atr_stop_k,
                    }


def evaluate_parameters(
    df: pd.DataFrame, config: BacktestConfig, lookback_days: int = 60
) -> Tuple[Dict, Dict]:
    """
    在最近 lookback_days 数据上搜索最佳参数。

    目标函数使用 年化收益 / (1 + 最大回撤)，兼顾收益与风险。
    """

    end_time = df["open_time"].max()
    start_time = end_time - pd.Timedelta(days=lookback_days)
    train = df[df["open_time"] >= start_time].reset_index(drop=True)
    best_score = -np.inf
    best_params: Dict = {}
    best_result: Dict = {}
    for params in parameter_grid():
        res = backtest(train, params, config)
        if not res:
            continue
        annual_return = (1 + res["total_return"]) ** (365 / lookback_days) - 1
        score = annual_return / (1 + res["max_drawdown"])
        if score > best_score:
            best_score = score
            best_params = params
            best_result = res
    return best_params, best_result


def save_best_params(
    path: str,
    date: dt.date,
    symbol: str,
    interval: str,
    params: Dict,
):
    """保存每日最佳参数到本地 json。"""

    payload = {
        "date": date.isoformat(),
        "symbol": symbol,
        "interval": interval,
        "params": params,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_best_params(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================== 伪实盘模块 ==============================
@dataclass
class Position:
    direction: int = 0  # 1=多，-1=空
    size: float = 0.0
    entry_price: float = 0.0


class LiveSimulator:
    """使用昨日最优参数，在最新行情上生成信号并跟踪持仓。"""

    def __init__(self, symbol: str, interval: str, data_loader: DataLoader, history_path: str):
        self.symbol = symbol
        self.interval = interval
        self.data_loader = data_loader
        self.history_path = history_path
        self.position = Position()
        self.realized_pnl = 0.0

    def run_once(self, params: Dict, lookback: int = 200):
        now = dt.datetime.utcnow()
        start = now - pd.Timedelta(days=7)
        df = self.data_loader.get_klines(self.symbol, self.interval, start, now)
        if df.empty:
            print("未获取到最新 K 线")
            return
        df = df.tail(lookback).reset_index(drop=True)
        data = generate_signals(df, params)
        latest = data.iloc[-1]
        signal = latest["exec_signal"]
        price = latest["close"]

        # 简单持仓更新逻辑：当前 bar 收盘时根据上一 bar 信号执行
        if self.position.direction != 0 and signal != self.position.direction:
            pnl = (price - self.position.entry_price) * self.position.direction * self.position.size
            self.realized_pnl += pnl
            self.position = Position()
        if signal != 0 and self.position.direction == 0:
            self.position = Position(direction=signal, size=1.0, entry_price=price)

        float_pnl = 0.0
        if self.position.direction != 0:
            float_pnl = (price - self.position.entry_price) * self.position.direction * self.position.size

        print(
            f"时间: {latest['open_time']} | 价格: {price:.2f} | 信号: {signal} | "
            f"持仓: {self.position.direction} | 浮盈: {float_pnl:.2f} | 已实现: {self.realized_pnl:.2f}"
        )


# ============================== 调度流程 ==============================
def daily_learning_loop():
    """示例性主流程：每日更新数据、搜索参数、保存结果并伪实盘。"""

    symbol = "BTCUSDT"
    interval = "15m"
    history_path = "btc_15m_history.csv"
    best_param_path = "best_params.json"
    lookback_days = 90
    data_loader = DataLoader()
    config = BacktestConfig()

    # 1) 更新或下载历史数据
    start = dt.datetime.utcnow() - pd.Timedelta(days=200)
    df = data_loader.incremental_update(symbol, interval, history_path, start)
    if df.empty:
        print("无法下载历史数据")
        return

    # 2) 每日优化（示例：直接运行一次）
    best_params, result = evaluate_parameters(df, config, lookback_days)
    today = dt.date.today()
    save_best_params(best_param_path, today, symbol, interval, best_params)

    print(
        f"日期: {today} | 搜索完成，最优参数: {best_params} | 总收益: {result.get('total_return', 0):.2%} | "
        f"最大回撤: {result.get('max_drawdown', 0):.2%}"
    )

    # 3) 伪实盘：使用前一天参数，这里为了演示使用刚保存的参数
    params_for_live = best_params
    simulator = LiveSimulator(symbol, interval, data_loader, history_path)
    # 真实部署时可用 while True 轮询，这里仅运行一次
    simulator.run_once(params_for_live)


# ============================== 使用说明 ==============================
USAGE = """
使用说明
--------
1. 安装依赖
   pip install pandas numpy requests

2. 下载或更新历史数据并单次回测
   - 调用 DataLoader.incremental_update 下载 BTCUSDT 15m 数据到本地 csv。
   - 使用 evaluate_parameters/backtest 进行回测。

3. 启动每日自学习 + 伪实盘
   - 直接运行本文件：python trend_follow_framework.py
   - 代码会更新数据、在最近 90 天搜索参数、保存 best_params.json，并用该参数运行一次伪实盘打印持仓信息。

4. Binance API 说明
   - 本代码仅使用公开 K 线接口，无需 API Key。若要接入真实交易，可在 DataLoader 中替换为带鉴权的下单接口，并在 LiveSimulator 中接入。
"""


if __name__ == "__main__":
    daily_learning_loop()
