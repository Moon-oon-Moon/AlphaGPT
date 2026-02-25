"""
沪深300风格指数轮动策略 — 均值方差优化模块

支持两种 µ / Σ 估计方式:
  - 方案一 (DAILY):  直接使用日频超额收益数据估计，再年化
  - 方案二 (MONTHLY): 先将日频数据复利合成为月频，再对月频数据估计

优化目标: 最大化夏普比率 µᵀw / sqrt(wᵀΣw)，等价于最大化 µᵀw / wᵀΣw
权重约束: 每个标的 0% ≤ wᵢ ≤ 50%，且权重之和 = 1
"""

from __future__ import annotations

import warnings
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# 枚举：估计频率
# ---------------------------------------------------------------------------

class EstimationFreq(str, Enum):
    DAILY = "daily"
    MONTHLY = "monthly"


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class MeanVarianceOptimizer:
    """
    均值方差优化器，用于沪深300风格指数月频轮动。

    参数
    ----------
    excess_returns : pd.DataFrame
        日频超额收益率，index 为交易日 (DatetimeIndex)，
        columns 为各 ETF/指数，values 为相对于沪深300全收益的日频超额收益。
    lookback_months : int
        回看月数，默认 6。
    w_lower : float
        单只标的权重下限，默认 0.0。
    w_upper : float
        单只标的权重上限，默认 0.5。
    freq : EstimationFreq | str
        估计频率，'daily' 或 'monthly'，默认 'monthly'。
    trading_days_per_year : int
        年交易日数，用于日频年化，默认 252。
    """

    def __init__(
        self,
        excess_returns: pd.DataFrame,
        lookback_months: int = 6,
        w_lower: float = 0.0,
        w_upper: float = 0.5,
        freq: EstimationFreq | str = EstimationFreq.MONTHLY,
        trading_days_per_year: int = 252,
    ) -> None:
        if not isinstance(excess_returns.index, pd.DatetimeIndex):
            excess_returns = excess_returns.copy()
            excess_returns.index = pd.to_datetime(excess_returns.index)

        self.excess_returns = excess_returns.sort_index()
        self.lookback_months = lookback_months
        self.w_lower = w_lower
        self.w_upper = w_upper
        self.freq = EstimationFreq(freq)
        self.trading_days_per_year = trading_days_per_year
        self.n_assets = excess_returns.shape[1]

        # 预先将日频数据转换为月频（方案二备用）
        self._monthly: pd.DataFrame = self._to_monthly(self.excess_returns)

    # ------------------------------------------------------------------
    # 私有工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
        """将日频超额收益复利合成为月频超额收益。"""
        return (1 + daily).resample("ME").apply(lambda x: x.prod() - 1)

    def _estimate_daily(self, date: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        """
        方案一：日频估计。

        截取 [date - lookback_months 个月, date) 窗口的日频数据，
        计算日均超额收益并年化，计算日频协方差矩阵并年化。

        返回
        -------
        mu : ndarray, shape (n,)   年化日均超额收益向量
        cov : ndarray, shape (n, n)  年化日频协方差矩阵
        """
        cutoff = date - pd.DateOffset(months=self.lookback_months)
        hist = self.excess_returns.loc[
            (self.excess_returns.index >= cutoff) & (self.excess_returns.index < date)
        ]
        if len(hist) < 2:
            raise ValueError(
                f"日频窗口数据不足（仅 {len(hist)} 条），无法估计协方差。"
            )
        mu = hist.mean().values * self.trading_days_per_year
        cov = hist.cov().values * self.trading_days_per_year
        return mu, cov

    def _estimate_monthly(self, date: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        """
        方案二：月频估计。

        截取最近 lookback_months 个完整月度数据，计算月均超额收益并年化，
        计算月频协方差矩阵并年化。

        返回
        -------
        mu : ndarray, shape (n,)   年化月均超额收益向量
        cov : ndarray, shape (n, n)  年化月频协方差矩阵
        """
        hist = self._monthly[self._monthly.index < date].iloc[-self.lookback_months :]
        if len(hist) < 2:
            raise ValueError(
                f"月频窗口数据不足（仅 {len(hist)} 个月），无法估计协方差。"
            )
        mu = hist.mean().values * 12
        cov = hist.cov().values * 12
        return mu, cov

    def _estimate(self, date: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        """根据 self.freq 选择估计方式。"""
        if self.freq == EstimationFreq.DAILY:
            return self._estimate_daily(date)
        return self._estimate_monthly(date)

    # ------------------------------------------------------------------
    # 优化
    # ------------------------------------------------------------------

    @staticmethod
    def _sharpe_neg(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
        """目标函数的负值（scipy.minimize 最小化）。"""
        port_ret = mu @ w
        port_var = w @ cov @ w
        if port_var <= 0:
            return 1e10
        return -port_ret / np.sqrt(port_var)

    def optimize(self, date: pd.Timestamp) -> Optional[np.ndarray]:
        """
        在给定调仓日计算最优持仓权重。

        参数
        ----------
        date : pd.Timestamp
            当前调仓日（回看此日期之前的数据）。

        返回
        -------
        weights : ndarray, shape (n,) 或 None（优化失败时）
        """
        mu, cov = self._estimate(date)

        n = self.n_assets
        bounds = [(self.w_lower, self.w_upper)] * n
        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
        w0 = np.ones(n) / n  # 等权初始猜测

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                self._sharpe_neg,
                w0,
                args=(mu, cov),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 1000},
            )

        if not result.success:
            warnings.warn(f"[{date.date()}] 优化未收敛: {result.message}")
            return None

        weights = result.x
        weights = np.clip(weights, self.w_lower, self.w_upper)
        weights /= weights.sum()  # 重新归一化以消除数值误差
        return weights

    # ------------------------------------------------------------------
    # 回测循环
    # ------------------------------------------------------------------

    def backtest(self, rebalance_dates: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
        """
        对全样本进行月频回测，返回每期持仓权重。

        参数
        ----------
        rebalance_dates : DatetimeIndex, optional
            调仓日序列。若为 None，则自动取月频数据中每月最后一个交易日。

        返回
        -------
        weights_df : pd.DataFrame
            行为调仓日，列为各 ETF/指数，值为持仓权重。
        """
        if rebalance_dates is None:
            # 取月频序列中有完整历史数据的日期
            monthly_idx = self._monthly.index
            start_offset = self.lookback_months
            rebalance_dates = monthly_idx[start_offset:]

        records = {}
        for dt in rebalance_dates:
            try:
                w = self.optimize(dt)
                if w is not None:
                    records[dt] = dict(zip(self.excess_returns.columns, w))
            except ValueError:
                continue  # 数据不足，跳过

        weights_df = pd.DataFrame.from_dict(records, orient="index")
        weights_df.index.name = "date"
        return weights_df


# ---------------------------------------------------------------------------
# 工具函数：对比两种方案
# ---------------------------------------------------------------------------

def compare_methods(
    excess_returns: pd.DataFrame,
    date: pd.Timestamp,
    lookback_months: int = 6,
    w_lower: float = 0.0,
    w_upper: float = 0.5,
) -> pd.DataFrame:
    """
    在同一调仓日分别用日频与月频方法估计 µ、Σ 并计算最优权重，
    返回对比结果。

    方案一 vs 方案二 的主要差异
    ------------------------------
    1. 样本量：日频约使用 ~126 个数据点，月频仅 6 个，估计误差更大。
    2. 序列相关：日频收益存在较强自相关（动量/均值回复），月频数据相对独立。
    3. 年化方式：日频乘以 252，月频乘以 12；两者在理论上等价，但实际样本
       估计值存在差异（月频复利效应 vs 日频简单累加）。
    4. 实务惯例：月频调仓策略通常优先使用月频数据估计，以保持频率一致性，
       避免日内噪声对协方差矩阵的污染，但当月份数据极少（如刚上市的 ETF）
       时，可退而求其次使用日频数据。

    参数
    ----------
    excess_returns : pd.DataFrame
        日频超额收益率数据。
    date : pd.Timestamp
        调仓日。
    lookback_months : int
        回看月数。
    w_lower / w_upper : float
        权重下/上限。

    返回
    -------
    result : pd.DataFrame
        包含日频估计与月频估计的 µ、最优权重对比。
    """
    opt_daily = MeanVarianceOptimizer(
        excess_returns, lookback_months, w_lower, w_upper, freq=EstimationFreq.DAILY
    )
    opt_monthly = MeanVarianceOptimizer(
        excess_returns, lookback_months, w_lower, w_upper, freq=EstimationFreq.MONTHLY
    )

    mu_d, cov_d = opt_daily._estimate_daily(date)
    mu_m, cov_m = opt_monthly._estimate_monthly(date)

    w_d = opt_daily.optimize(date)
    w_m = opt_monthly.optimize(date)

    assets = excess_returns.columns.tolist()
    result = pd.DataFrame(
        {
            "mu_daily (ann.)": mu_d,
            "mu_monthly (ann.)": mu_m,
            "weight_daily": w_d if w_d is not None else [np.nan] * len(assets),
            "weight_monthly": w_m if w_m is not None else [np.nan] * len(assets),
        },
        index=assets,
    )
    return result
