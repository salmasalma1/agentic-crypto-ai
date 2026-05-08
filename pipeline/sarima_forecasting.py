"""Step 4-B: SARIMA forecasting."""
import numpy as np
import pandas as pd
from utils.config import FORECAST_STEPS


def prepare_time_series(df: pd.DataFrame, date_col: str, target_col: str):
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    temp = df[[date_col, target_col]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp[target_col] = pd.to_numeric(temp[target_col], errors="coerce")
    temp = temp.dropna(subset=[date_col, target_col]).sort_values(date_col)
    if temp.empty:
        raise ValueError("No valid time-series rows after date/target cleaning.")
    series = temp.set_index(date_col).resample("D")[target_col].mean().interpolate("linear").ffill().bfill()
    if len(series) < 8:
        raise ValueError("SARIMA requires at least 8 dated observations for this project demo.")
    return series


def _naive_forecast(series, forecast_steps):
    last = float(series.iloc[-1])
    if len(series) >= 4:
        slope = float((series.iloc[-1] - series.iloc[-4]) / 3)
    else:
        slope = 0.0
    idx = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq="D")
    values = [last + slope * (i + 1) for i in range(forecast_steps)]
    return pd.Series(values, index=idx)


def run_sarima_forecast(df: pd.DataFrame, date_col: str, target_col: str, forecast_steps: int = FORECAST_STEPS):
    series = prepare_time_series(df, date_col, target_col)
    engine = "SARIMAX"
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False)
        forecast = fitted.get_forecast(steps=forecast_steps)
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
    except Exception:
        engine = "naive_forecast_fallback"
        forecast_mean = _naive_forecast(series, forecast_steps)
        std = float(series.tail(min(14, len(series))).std() or 0.0)
        forecast_ci = pd.DataFrame({"lower": forecast_mean - 1.96 * std, "upper": forecast_mean + 1.96 * std}, index=forecast_mean.index)

    last_actual = float(series.iloc[-1])
    final_forecast = float(forecast_mean.iloc[-1])
    expected_change_pct = 0.0 if last_actual == 0 else ((final_forecast - last_actual) / abs(last_actual)) * 100
    volatility_pct = 0.0 if last_actual == 0 else (float(np.std(forecast_mean)) / abs(last_actual)) * 100
    forecast_risk_score = min(abs(expected_change_pct) / 10, 1.0) * 0.6 + min(volatility_pct / 10, 1.0) * 0.4
    forecast_risk_score = max(0.0, min(1.0, forecast_risk_score))

    forecast_df = pd.DataFrame({"forecast_date": forecast_mean.index, "forecast_value": forecast_mean.values})
    if forecast_ci is not None:
        forecast_df["lower_bound"] = forecast_ci.iloc[:, 0].values
        forecast_df["upper_bound"] = forecast_ci.iloc[:, 1].values
    return {
        "series": series,
        "forecast_df": forecast_df,
        "last_actual_value": round(last_actual, 4),
        "final_forecast_value": round(final_forecast, 4),
        "expected_change_pct": round(float(expected_change_pct), 4),
        "forecast_volatility_pct": round(float(volatility_pct), 4),
        "forecast_risk_score": round(float(forecast_risk_score), 4),
        "engine": engine,
    }
