import pandas as pd

from src.business_analysis import analyze_business, build_manual_snapshot, prepare_business_data
from src.data_generation import generate_business_history
from src.forecasting import forecast_series


def test_business_analysis_end_to_end():
    df = generate_business_history(months=24, sector="ecommerce", preset_name="tunisia", seed=5)
    report = analyze_business(
        df,
        sector="ecommerce",
        preset_name="tunisia",
        language="en",
        periods=6,
    )
    assert report["mode"] == "business"
    kpis = report["kpis"]
    assert kpis["total_revenue"] > 0
    assert kpis["net_margin_pct"] is not None
    assert len(report["benchmarks"]) == 6
    assert report["forecast"]["revenue"]
    assert len(report["forecast"]["revenue"]) == 6
    assert report["actions"]
    assert 35 <= report["confidence"] <= 93
    assert report["data_quality"]["months"] == 24
    assert report["disclaimer"]


def test_prepare_business_data_keeps_date_and_optional_columns():
    frame = pd.DataFrame(
        {
            "month": ["2024-01", "2024-02", "2024-03"],
            "sales": [10000, 11000, 12100],
            "expenses": [7000, 7600, 8200],
        }
    )
    monthly, flags = prepare_business_data(frame)
    assert flags["mapping"]["date"] == "month"
    assert flags["mapping"]["revenue"] == "sales"
    assert flags["mapping"]["cost"] == "expenses"
    assert flags["mapping"]["price"] is None
    assert flags["mapping"]["churn_rate"] is None
    assert list(monthly["date"].dt.strftime("%Y-%m")) == ["2024-01", "2024-02", "2024-03"]
    assert "date" in monthly.columns


def test_forecast_series_has_bands():
    df = generate_business_history(months=18, sector="saas", preset_name="tunisia", seed=6)
    forecast = forecast_series(df, "month", "revenue", periods=4)
    assert len(forecast) == 4
    assert {"forecast", "lower", "upper", "method"}.issubset(forecast.columns)
    assert (forecast["lower"] <= forecast["forecast"]).all()
    assert (forecast["forecast"] <= forecast["upper"]).all()


def test_manual_snapshot():
    frame, flags = build_manual_snapshot(25000, 18000, 300, 1500, 2.5)
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 12
    assert flags["manual_snapshot"] is True
    report = analyze_business(
        frame,
        sector="ecommerce",
        preset_name="tunisia",
        language="fr",
        manual_snapshot=True,
    )
    assert report["data_quality"]["manual_snapshot"] is True
    assert report["kpis"]["last_month_revenue"] > 0
