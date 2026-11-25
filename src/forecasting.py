"""Time-series forecasting for Mode B with statsmodels and a regression fallback."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd


def _monthly_series(df: pd.DataFrame, date_col: str, value_col: str) -> pd.Series:
    work = df[[date_col, value_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[date_col, value_col])
    if work.empty:
        raise ValueError("No valid dates/values available for forecasting.")
    series = (
        work.set_index(date_col)[value_col]
        .sort_index()
        .resample("MS")
        .sum()
        .astype(float)
    )
    series = series[series.index.notna()]
    return series


def _linear_forecast(series: pd.Series, periods: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    y = series.to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    if len(y) < 2:
        raise ValueError("At least two periods are required.")
    slope, intercept = np.polyfit(x, y, 1)
    future_x = np.arange(len(y), len(y) + periods, dtype=float)
    predictions = slope * future_x + intercept
    residuals = y - (slope * x + intercept)
    residual_std = float(np.std(residuals, ddof=1)) if len(y) > 2 else float(np.std(residuals))
    spread = 1.96 * max(residual_std, abs(float(np.mean(y))) * 0.03)
    return predictions, predictions - spread, predictions + spread, "Linear trend (fallback)"


def forecast_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    *,
    periods: int = 6,
    non_negative: bool = True,
) -> pd.DataFrame:
    """Forecast monthly values with a confidence interval.

    Tries Holt-Winters/Exponential Smoothing from statsmodels, then ARIMA, then a
    robust linear trend. The returned method is exposed so the UI can be honest.
    """
    series = _monthly_series(df, date_col, value_col)
    if len(series) < 4:
        raise ValueError("At least four monthly observations are needed for a forecast.")

    forecast_index = pd.date_range(
        start=series.index[-1] + pd.offsets.MonthBegin(1), periods=periods, freq="MS"
    )
    forecast_values: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    method = ""

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(series) >= 24 and np.std(series) > 1e-6:
                model = ExponentialSmoothing(
                    series,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=12,
                    initialization_method="estimated",
                )
                fit = model.fit(optimized=True)
                method = "Holt-Winters exponential smoothing (statsmodels)"
            else:
                model = ExponentialSmoothing(
                    series,
                    trend="add",
                    seasonal=None,
                    initialization_method="estimated",
                )
                fit = model.fit(optimized=True)
                method = "Exponential smoothing (statsmodels)"
            prediction = fit.get_forecast(periods)
            forecast_values = np.asarray(prediction.predicted_mean, dtype=float)
            conf = np.asarray(prediction.conf_int(alpha=0.20), dtype=float)
            lower = conf[:, 0]
            upper = conf[:, 1]
    except Exception:
        forecast_values = None

    if forecast_values is None:
        try:
            from statsmodels.tsa.arima.model import ARIMA

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ARIMA(series, order=(1, 1, 1)).fit()
                prediction = fit.get_forecast(periods)
                forecast_values = np.asarray(prediction.predicted_mean, dtype=float)
                conf = np.asarray(prediction.conf_int(alpha=0.20), dtype=float)
                lower = conf[:, 0]
                upper = conf[:, 1]
                method = "ARIMA(1,1,1) (statsmodels)"
        except Exception:
            forecast_values = None

    if forecast_values is None or lower is None or upper is None:
        forecast_values, lower, upper, method = _linear_forecast(series, periods)

    if non_negative:
        forecast_values = np.maximum(forecast_values, 0)
        lower = np.maximum(lower, 0)
        upper = np.maximum(upper, 0)
    lower = np.minimum(lower, forecast_values)
    upper = np.maximum(upper, forecast_values)

    return pd.DataFrame(
        {
            "date": forecast_index,
            "forecast": forecast_values,
            "lower": lower,
            "upper": upper,
            "method": method,
        }
    )


def forecast_business(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    revenue_col: str = "revenue",
    cost_col: str = "cost",
    periods: int = 6,
) -> dict[str, Any]:
    """Forecast revenue and, when costs exist, profit."""
    result: dict[str, Any] = {"periods": periods, "revenue": None, "profit": None, "method": None}
    revenue_forecast = forecast_series(df, date_col, revenue_col, periods=periods, non_negative=True)
    result["revenue"] = revenue_forecast
    result["method"] = str(revenue_forecast["method"].iloc[0])
    if cost_col in df.columns:
        try:
            cost_forecast = forecast_series(df, date_col, cost_col, periods=periods, non_negative=True)
            profit = revenue_forecast[["date", "forecast"]].rename(columns={"forecast": "revenue"})
            profit["cost"] = cost_forecast["forecast"].to_numpy()
            profit["forecast"] = profit["revenue"] - profit["cost"]
            profit["lower"] = revenue_forecast["lower"].to_numpy() - cost_forecast["upper"].to_numpy()
            profit["upper"] = revenue_forecast["upper"].to_numpy() - cost_forecast["lower"].to_numpy()
            profit["method"] = f"Revenue and cost: {result['method']}"
            result["profit"] = profit
        except Exception:
            result["profit"] = None
    return result
