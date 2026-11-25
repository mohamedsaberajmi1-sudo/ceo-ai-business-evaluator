"""Mode B: KPI computation, diagnostics, benchmark comparison and actions."""

from __future__ import annotations

import unicodedata
from typing import Any

import numpy as np
import pandas as pd

from .config import currency_for, get_bot_name, get_preset, get_sector_profile, sector_label
from .forecasting import forecast_business
from .i18n import t
from .utils import clamp, json_safe, safe_float

_DATE_ALIASES = ("date", "month", "mois", "period", "periode", "période", "ds")
_REVENUE_ALIASES = ("revenue", "revenues", "sales", "turnover", "chiffre_affaires", "chiffre d'affaires", "ca", "ventes", "إيرادات", "مبيعات")
_COST_ALIASES = ("cost", "costs", "total_cost", "total_costs", "expenses", "charges", "dépenses", "depenses", "تكاليف", "مصاريف")
_CUSTOMER_ALIASES = ("customers", "clients", "active_customers", "clients_actifs", "عملاء")
_NEW_CUSTOMER_ALIASES = ("new_customers", "new_clients", "nouveaux_clients", "عملاء_جدد")
_MARKETING_ALIASES = ("marketing_spend", "marketing", "ad_spend", "ads", "dépenses_marketing", "تسويق")
_PRICE_ALIASES = ("price", "avg_price", "average_price", "prix", "سعر")
_CHURN_ALIASES = ("churn_rate", "churn", "attrition", "taux_attrition", "تسرب")
_COGS_ALIASES = ("cogs", "cost_of_goods", "cost_of_goods_sold", "coût_des_ventes", "تكلفة_المبيعات")


def _normalize_name(name: Any) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    return text.strip().lower().replace("-", "_").replace(" ", "_")


def _find_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    normalized = {_normalize_name(column): column for column in columns}
    # Exact match first. Empty keys are ignored: non-Latin aliases lose all
    # their characters during ASCII normalization and "" is a substring of
    # every column, which would otherwise match the first column by accident.
    for alias in aliases:
        key = _normalize_name(alias)
        if key and key in normalized:
            return normalized[key]
    # Then a conservative fuzzy pass. Short aliases such as "ca" are excluded
    # to avoid matching unrelated columns.
    for norm, original in normalized.items():
        if any(len(_normalize_name(alias)) >= 3 and _normalize_name(alias) in norm for alias in aliases):
            return original
    return None


def _canonicalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    out = df.copy()
    mapping: dict[str, str | None] = {}
    for canonical, aliases in (
        ("date", _DATE_ALIASES),
        ("revenue", _REVENUE_ALIASES),
        ("cost", _COST_ALIASES),
        ("customers", _CUSTOMER_ALIASES),
        ("new_customers", _NEW_CUSTOMER_ALIASES),
        ("marketing_spend", _MARKETING_ALIASES),
        ("price", _PRICE_ALIASES),
        ("churn_rate", _CHURN_ALIASES),
        ("cogs", _COGS_ALIASES),
    ):
        source = _find_column(list(out.columns), aliases)
        mapping[canonical] = source
        if source is not None and source != canonical:
            out = out.rename(columns={source: canonical})
    return out, mapping


def prepare_business_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize a user's CSV/Excel into a clean monthly DataFrame + quality flags."""
    if df is None or df.empty:
        raise ValueError("The uploaded dataset is empty.")
    work, mapping = _canonicalize_columns(df)
    if mapping["revenue"] is None:
        raise ValueError(
            "A revenue column is required. Accepted names include revenue, sales, turnover, CA."
        )
    flags: dict[str, Any] = {
        "cost_estimated": False,
        "customers_derived": False,
        "customers_missing": False,
        "marketing_missing": False,
        "date_generated": False,
    }
    work["revenue"] = pd.to_numeric(work["revenue"], errors="coerce")
    work = work.dropna(subset=["revenue"])
    if work.empty:
        raise ValueError("No valid revenue values found.")

    if mapping["date"] is None:
        dates = pd.date_range(end=pd.Timestamp.today().normalize().replace(day=1), periods=len(work), freq="MS")
        work["date"] = dates
        flags["date_generated"] = True
    else:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        invalid = work["date"].isna().sum()
        if invalid:
            work = work.dropna(subset=["date"])
        if work.empty:
            raise ValueError("No valid dates found in the date/month column.")
    work["date"] = work["date"].dt.to_period("M").dt.to_timestamp()

    for column in ("cost", "customers", "new_customers", "marketing_spend", "price", "churn_rate", "cogs"):
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")

    if mapping["cost"] is None:
        work["cost"] = work["revenue"] * 0.72
        flags["cost_estimated"] = True
    else:
        work["cost"] = work["cost"].fillna(work["revenue"] * 0.72)

    if mapping["marketing_spend"] is None:
        work["marketing_spend"] = 0.0
        flags["marketing_missing"] = True
    else:
        work["marketing_spend"] = work["marketing_spend"].fillna(0.0)

    if mapping["customers"] is None:
        if mapping["price"] is not None:
            work["customers"] = (work["revenue"] / work["price"].replace(0, np.nan)).round(1)
            flags["customers_derived"] = True
        else:
            work["customers"] = np.nan
            flags["customers_missing"] = True
    else:
        fallback_price = safe_float(work["price"].mean() if "price" in work else 1, 1)
        fallback_price = max(fallback_price, 1.0)
        work["customers"] = work["customers"].fillna(work["revenue"] / fallback_price)

    if "new_customers" not in work.columns:
        work["new_customers"] = np.nan
    if "churn_rate" not in work.columns:
        work["churn_rate"] = np.nan
    if "cogs" not in work.columns:
        work["cogs"] = np.nan
    if "price" not in work.columns:
        work["price"] = np.nan

    aggregation: dict[str, str] = {
        "revenue": "sum",
        "cost": "sum",
        "marketing_spend": "sum",
    }
    for column in ("new_customers", "cogs"):
        aggregation[column] = "sum"
    for column in ("customers", "price", "churn_rate"):
        aggregation[column] = "mean"

    monthly = (
        work.set_index("date")
        .groupby(pd.Grouper(freq="MS"))
        .agg(aggregation)
        .reset_index()
        .sort_values("date")
    )
    monthly = monthly[monthly["revenue"].notna()].reset_index(drop=True)
    if len(monthly) < 2:
        raise ValueError("At least two monthly observations are required.")
    flags["months"] = int(len(monthly))
    flags["mapping"] = {key: value for key, value in mapping.items()}
    return monthly, flags


def _monthly_growth(series: pd.Series) -> float:
    y = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(y) < 2:
        return 0.0
    positive = y > 0
    if positive.all():
        slope = float(np.polyfit(np.arange(len(y)), np.log(y), 1)[0])
        return (np.exp(slope) - 1) * 100
    first, last = y[0], y[-1]
    if first > 0:
        return ((last / first) ** (1 / max(len(y) - 1, 1)) - 1) * 100
    return 0.0


def _estimate_churn(df: pd.DataFrame, benchmark: float) -> tuple[float, bool]:
    if "churn_rate" in df.columns and df["churn_rate"].notna().any():
        values = df["churn_rate"].dropna()
        if values.max() <= 1.5:
            return float(values.mean() * 100), False
        return float(values.mean()), False
    if "new_customers" in df.columns and df["new_customers"].notna().any():
        customers = pd.to_numeric(df["customers"], errors="coerce")
        additions = pd.to_numeric(df["new_customers"], errors="coerce").fillna(0)
        churn_rates: list[float] = []
        for index in range(1, len(df)):
            previous = customers.iloc[index - 1]
            current = customers.iloc[index]
            new = additions.iloc[index]
            if pd.notna(previous) and previous > 0 and pd.notna(current):
                churn_rates.append(float(clamp((previous + new - current) / previous, 0, 1)))
        if churn_rates:
            return float(np.mean(churn_rates) * 100), False
    return float(benchmark * 100), True


def _status(actual: float | None, benchmark: float | None, higher_is_better: bool) -> str:
    if actual is None or benchmark is None or benchmark == 0:
        return "unknown"
    ratio = actual / benchmark if higher_is_better else benchmark / actual if actual != 0 else 0
    if higher_is_better:
        if actual >= benchmark * 1.02:
            return "good"
        if actual >= benchmark * 0.85:
            return "watch"
        return "bad"
    if actual <= benchmark * 1.02:
        return "good"
    if actual <= benchmark * 1.20:
        return "watch"
    return "bad"


def _kpis(df: pd.DataFrame, sector: str, flags: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = get_sector_profile(sector)
    benchmarks = profile.get("benchmark", {})
    total_revenue = float(df["revenue"].sum())
    total_cost = float(df["cost"].sum())
    total_profit = total_revenue - total_cost
    total_marketing = float(df["marketing_spend"].sum())
    revenue_series = df["revenue"].astype(float)
    cost_series = df["cost"].astype(float)
    last = df.iloc[-1]
    previous = df.iloc[-2]

    net_margin = total_profit / total_revenue * 100 if total_revenue > 0 else 0.0
    growth = _monthly_growth(revenue_series)
    churn, churn_estimated = _estimate_churn(df, float(benchmarks.get("churn", 0.045)))
    total_new = float(pd.to_numeric(df.get("new_customers", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    cac = total_marketing / total_new if total_new > 0 else None
    avg_customers = float(pd.to_numeric(df["customers"], errors="coerce").dropna().mean()) if df["customers"].notna().any() else None
    arpu = total_revenue / (avg_customers * max(len(df), 1)) if avg_customers and avg_customers > 0 else None
    ltv = arpu / (churn / 100) if arpu and churn > 0 else None
    ltv_cac = ltv / cac if ltv and cac and cac > 0 else None
    marketing_ratio = total_marketing / total_revenue * 100 if total_revenue > 0 else 0.0
    gross_margin = None
    if df["cogs"].notna().any() and float(df["cogs"].fillna(0).sum()) > 0:
        gross_margin = (total_revenue - float(df["cogs"].fillna(0).sum())) / total_revenue * 100
    volatility = float(revenue_series.std() / revenue_series.mean() * 100) if revenue_series.mean() else 0.0
    cost_growth = _monthly_growth(cost_series)
    revenue_growth = growth
    average_order_value = total_revenue / (avg_customers * max(len(df), 1)) if avg_customers else None

    kpis = {
        "total_revenue": total_revenue,
        "total_cost": total_cost,
        "total_profit": total_profit,
        "last_month_revenue": float(last["revenue"]),
        "last_month_profit": float(last["revenue"] - last["cost"]),
        "previous_month_revenue": float(previous["revenue"]),
        "monthly_growth_pct": float(growth),
        "net_margin_pct": float(net_margin),
        "gross_margin_pct": float(gross_margin) if gross_margin is not None else None,
        "churn_pct": float(churn),
        "churn_estimated": bool(churn_estimated),
        "cac": float(cac) if cac is not None else None,
        "ltv": float(ltv) if ltv is not None else None,
        "ltv_cac": float(ltv_cac) if ltv_cac is not None else None,
        "marketing_spend": total_marketing,
        "marketing_ratio_pct": float(marketing_ratio),
        "avg_customers": float(avg_customers) if avg_customers else None,
        "avg_order_value": float(average_order_value) if average_order_value else None,
        "revenue_volatility_pct": volatility,
        "cost_growth_pct": float(cost_growth),
        "months": int(len(df)),
    }
    benchmark_values = {
        "net_margin_pct": float(benchmarks.get("net_margin", 0.10)) * 100,
        "monthly_growth_pct": float(benchmarks.get("monthly_growth", 0.02)) * 100,
        "churn_pct": float(benchmarks.get("churn", 0.045)) * 100,
        "cac": float(benchmarks.get("cac", 80)),
        "ltv_cac": float(benchmarks.get("ltv_cac", 3.0)),
        "marketing_ratio_pct": float(benchmarks.get("marketing_ratio", 0.09)) * 100,
    }
    return kpis, benchmark_values


def _benchmark_rows(kpis: dict[str, Any], bench: dict[str, float], language: str) -> list[dict[str, Any]]:
    specs = [
        ("net_margin_pct", "bench_net_margin", "%", True),
        ("monthly_growth_pct", "bench_growth", "%", True),
        ("churn_pct", "bench_churn", "%", False),
        ("cac", "bench_cac", "currency", False),
        ("ltv_cac", "bench_ltv_cac", "ratio", True),
        ("marketing_ratio_pct", "bench_marketing_ratio", "%", False),
    ]
    rows: list[dict[str, Any]] = []
    for key, label_key, unit, higher in specs:
        actual = kpis.get(key)
        benchmark = bench.get(key)
        rows.append(
            {
                "key": key,
                "label": t(label_key, language),
                "actual": actual,
                "benchmark": benchmark,
                "unit": unit,
                "better": "higher" if higher else "lower",
                "status": _status(actual, benchmark, higher),
            }
        )
    return rows


def _problem(
    key: str,
    severity: str,
    kpis: dict[str, Any],
    bench: dict[str, float],
    language: str,
) -> dict[str, str]:
    replacements = {
        "profit": kpis.get("last_month_profit", 0),
        "margin": kpis.get("net_margin_pct", 0),
        "bench_margin": bench.get("net_margin_pct", 0),
        "growth": kpis.get("monthly_growth_pct", 0),
        "churn": kpis.get("churn_pct", 0),
        "bench_churn": bench.get("churn_pct", 0),
        "ltv_cac": kpis.get("ltv_cac") or 0,
        "marketing_ratio": kpis.get("marketing_ratio_pct", 0),
        "bench_marketing": bench.get("marketing_ratio_pct", 0),
        "volatility": kpis.get("revenue_volatility_pct", 0),
        "cost_growth": kpis.get("cost_growth_pct", 0),
        "revenue_growth": kpis.get("monthly_growth_pct", 0),
    }
    detail = t(f"problem_{key}_detail", language, **replacements)
    return {
        "key": key,
        "severity": severity,
        "title": t(f"problem_{key}_title", language),
        "detail": detail,
    }


def _business_actions(
    problems: list[dict[str, str]],
    kpis: dict[str, Any],
    bench: dict[str, float],
    language: str,
) -> list[dict[str, Any]]:
    revenue = max(kpis["total_revenue"] / max(kpis["months"], 1), 1.0)
    cost = max(kpis["total_cost"] / max(kpis["months"], 1), 1.0)
    marketing = max(kpis["marketing_spend"] / max(kpis["months"], 1), 1.0)
    action_map = {
        "negative_profit": ("business_action_profit", revenue * 0.05 + cost * 0.02),
        "low_margin": ("business_action_margin", revenue * 0.03),
        "declining": ("business_action_growth", revenue * 0.04),
        "high_churn": ("business_action_retention", revenue * 0.025),
        "ltv_cac": ("business_action_cac", marketing * 0.18),
        "high_marketing": ("business_action_marketing", marketing * 0.15),
        "cost_growth": ("business_action_cost", cost * 0.04),
        "volatility": ("business_action_forecast", revenue * 0.015),
    }
    actions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for problem in problems:
        if problem["key"] in seen or problem["key"] not in action_map:
            continue
        seen.add(problem["key"])
        label, impact = action_map[problem["key"]]
        actions.append(
            {
                "key": problem["key"],
                "action": t(label, language),
                "rationale": t(f"business_rationale_{problem['key']}", language),
                "impact_monthly": round(float(impact), 1),
                "impact_annual": round(float(impact * 12), 1),
                "priority": "high" if problem["severity"] == "high" else "medium",
                "confidence": "medium" if problem["key"] in {"volatility", "cost_growth"} else "high",
            }
        )
    if not actions:
        actions.append(
            {
                "key": "optimize",
                "action": t("business_action_optimize", language),
                "rationale": t("business_rationale_optimize", language),
                "impact_monthly": round(revenue * 0.02, 1),
                "impact_annual": round(revenue * 0.02 * 12, 1),
                "priority": "medium",
                "confidence": "medium",
            }
        )
    actions.sort(key=lambda item: item["impact_monthly"], reverse=True)
    return actions


def _detect_problems(
    kpis: dict[str, Any], bench: dict[str, float], language: str
) -> list[dict[str, str]]:
    problems: list[dict[str, str]] = []
    if kpis["last_month_profit"] < 0 or kpis["total_profit"] < 0:
        problems.append(_problem("negative_profit", "high", kpis, bench, language))
    if kpis["net_margin_pct"] < bench["net_margin_pct"] - 3:
        problems.append(_problem("low_margin", "high" if kpis["net_margin_pct"] < 0 else "medium", kpis, bench, language))
    if kpis["monthly_growth_pct"] < 0:
        problems.append(_problem("declining", "high", kpis, bench, language))
    if kpis["churn_pct"] > bench["churn_pct"] + 2:
        problems.append(_problem("high_churn", "high" if kpis["churn_pct"] > bench["churn_pct"] + 5 else "medium", kpis, bench, language))
    if kpis["ltv_cac"] is not None and kpis["ltv_cac"] < 3.0:
        problems.append(_problem("ltv_cac", "high", kpis, bench, language))
    if kpis["marketing_ratio_pct"] > bench["marketing_ratio_pct"] + 5:
        problems.append(_problem("high_marketing", "medium", kpis, bench, language))
    if kpis["revenue_volatility_pct"] > 32:
        problems.append(_problem("volatility", "medium", kpis, bench, language))
    if kpis["cost_growth_pct"] > kpis["monthly_growth_pct"] + 3:
        problems.append(_problem("cost_growth", "medium", kpis, bench, language))
    return problems[:6]


def _confidence(flags: dict[str, Any]) -> float:
    score = 62.0
    months = int(flags.get("months", 0))
    score += 8 if months >= 12 else 3 if months >= 6 else -8
    score += 6 if months >= 24 else 0
    score -= 12 if flags.get("cost_estimated") else 0
    score -= 9 if flags.get("customers_missing") else 0
    score -= 7 if flags.get("marketing_missing") else 0
    score -= 10 if flags.get("date_generated") else 0
    score -= 6 if flags.get("manual_snapshot") else 0
    return float(clamp(score, 35, 93))


def _forecast_records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None:
        return []
    out = frame.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return json_safe(out.to_dict(orient="records"))


def analyze_business(
    df: pd.DataFrame,
    *,
    sector: str = "other",
    preset_name: str = "tunisia",
    language: str = "en",
    periods: int = 6,
    manual_snapshot: bool = False,
) -> dict[str, Any]:
    """Run the complete Mode B analysis and return a serializable report."""
    monthly, flags = prepare_business_data(df)
    flags["manual_snapshot"] = manual_snapshot
    currency = currency_for(preset_name)
    kpis, benchmarks = _kpis(monthly, sector, flags)
    benchmark_rows = _benchmark_rows(kpis, benchmarks, language)
    problems = _detect_problems(kpis, benchmarks, language)
    actions = _business_actions(problems, kpis, benchmarks, language)
    forecast = {"revenue": [], "profit": [], "method": ""}
    try:
        forecast_result = forecast_business(monthly, periods=periods)
        forecast = {
            "revenue": _forecast_records(forecast_result.get("revenue")),
            "profit": _forecast_records(forecast_result.get("profit")),
            "method": forecast_result.get("method", ""),
        }
    except Exception as exc:
        forecast["error"] = str(exc)

    quality_notes: list[str] = []
    if flags.get("cost_estimated"):
        quality_notes.append(t("quality_cost_estimated", language))
    if flags.get("date_generated"):
        quality_notes.append(t("quality_date_generated", language))
    if flags.get("customers_missing"):
        quality_notes.append(t("quality_customers_missing", language))
    if flags.get("marketing_missing"):
        quality_notes.append(t("quality_marketing_missing", language))
    if manual_snapshot:
        quality_notes.append(t("quality_manual_snapshot", language))

    summary = t(
        "business_summary",
        language,
        revenue=round(kpis["last_month_revenue"]),
        profit=round(kpis["last_month_profit"]),
        growth=round(kpis["monthly_growth_pct"], 1),
        currency=currency,
    )
    period_start = pd.to_datetime(monthly["date"].iloc[0]).strftime("%Y-%m")
    period_end = pd.to_datetime(monthly["date"].iloc[-1]).strftime("%Y-%m")
    preview = monthly.tail(8).copy()
    preview["date"] = pd.to_datetime(preview["date"]).dt.strftime("%Y-%m-%d")
    return {
        "report_id": None,
        "mode": "business",
        "language": language,
        "bot_name": get_bot_name(),
        "preset": preset_name,
        "currency": currency,
        "title": f"{sector_label(sector, language)} — {period_start} → {period_end}",
        "sector": sector,
        "sector_label": sector_label(sector, language),
        "kpis": kpis,
        "benchmarks": benchmark_rows,
        "problems": problems,
        "actions": actions,
        "forecast": forecast,
        "data_quality": {
            "months": flags.get("months"),
            "period_start": period_start,
            "period_end": period_end,
            "flags": flags,
            "notes": quality_notes,
            "manual_snapshot": manual_snapshot,
        },
        "data_preview": json_safe(preview.to_dict(orient="records")),
        "confidence": _confidence(flags),
        "summary": summary,
        "disclaimer": t("disclaimer", language),
    }


def build_manual_snapshot(
    monthly_revenue: float,
    monthly_cost: float,
    customers: float,
    marketing_spend: float,
    monthly_growth_pct: float,
    *,
    months: int = 12,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a 12-month indicative history from a user's current snapshot."""
    growth = clamp(monthly_growth_pct / 100.0, -0.25, 0.25)
    end = pd.Timestamp.today().normalize().replace(day=1)
    dates = pd.date_range(end=end, periods=months, freq="MS")
    revenue = float(max(monthly_revenue, 1))
    cost = float(max(monthly_cost, 0))
    clients = float(max(customers, 1))
    marketing = float(max(marketing_spend, 0))
    records: list[dict[str, Any]] = []
    for index in range(months - 1, -1, -1):
        factor = (1 + growth) ** index
        records.append(
            {
                "month": dates[months - 1 - index].strftime("%Y-%m-%d"),
                "revenue": revenue * factor,
                "cost": cost * max(0.25, (1 + growth * 0.65) ** index),
                "customers": clients * factor,
                "new_customers": max(1, round(clients * 0.09)),
                "marketing_spend": marketing * max(0.5, (1 + growth * 0.5) ** index),
                "price": revenue / max(clients, 1),
                "churn_rate": 0.0,
            }
        )
    frame = pd.DataFrame(records)
    flags = {
        "manual_snapshot": True,
        "cost_estimated": False,
        "date_generated": False,
        "customers_missing": False,
        "marketing_missing": False,
    }
    return frame, flags
