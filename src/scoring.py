"""Transparent business scoring engine backed by the trained ML models.

The score combines:
1. a 12-criterion rubric (weights sum to 1), and
2. the classifier's estimated success probability.

Financial scenarios use explicit unit economics (price, customers, margin, fixed
costs, marketing) and regression predictions for ROI / break-even. Every output
contains a confidence level and a disclaimer.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import (
    currency_for,
    get_bot_name,
    get_config,
    get_paths,
    get_preset,
    get_sector_profile,
    preset_names,
    sector_label,
)
from .i18n import t
from .preprocessing import (
    FEATURE_COLUMNS,
    build_preprocessor,
    feature_display_name,
    get_estimator,
    raw_feature_from_transformed,
)
from .utils import clamp, safe_float

TARGET_MARKETS = ("b2c", "b2b", "b2b2c", "marketplace")
REVENUE_MODELS = ("subscription", "one_time", "commission", "freemium", "advertising", "hybrid")

CRITERIA: list[dict[str, Any]] = [
    {"key": "market_size_demand", "weight": 0.12, "label": "crit_market_size"},
    {"key": "market_growth_trend", "weight": 0.08, "label": "crit_market_growth"},
    {"key": "target_audience_fit", "weight": 0.10, "label": "crit_audience_fit"},
    {"key": "competition", "weight": 0.10, "label": "crit_competition"},
    {"key": "differentiation", "weight": 0.12, "label": "crit_differentiation"},
    {"key": "startup_cost", "weight": 0.08, "label": "crit_startup_cost"},
    {"key": "operational_feasibility", "weight": 0.08, "label": "crit_operations"},
    {"key": "revenue_model", "weight": 0.10, "label": "crit_revenue_model"},
    {"key": "scalability", "weight": 0.08, "label": "crit_scalability"},
    {"key": "timing", "weight": 0.05, "label": "crit_timing"},
    {"key": "regulatory_risk", "weight": 0.04, "label": "crit_regulation"},
    {"key": "team_readiness", "weight": 0.05, "label": "crit_team"},
]

_REVENUE_MODEL_SCORE = {
    "subscription": 8.6,
    "hybrid": 8.0,
    "commission": 7.0,
    "freemium": 6.2,
    "one_time": 5.5,
    "advertising": 4.5,
}
_SCALABILITY_ADJUST = {
    "subscription": 1.2,
    "hybrid": 0.8,
    "commission": 0.9,
    "freemium": 0.5,
    "one_time": -0.2,
    "advertising": 0.7,
}

_MODEL_CACHE: dict[str, Any] | None = None


def _models_dir() -> Path:
    return get_paths().models


@lru_cache(maxsize=1)
def load_models() -> dict[str, Any]:
    """Load persisted artifacts, returning an empty dict if training is missing."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    model_dir = _models_dir()
    metadata: dict[str, Any] = {}
    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        import json

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    loaded: dict[str, Any] = {"metadata": metadata, "available": False}
    classifier_path = model_dir / "classifier.joblib"
    roi_path = model_dir / "regression_roi.joblib"
    be_path = model_dir / "regression_breakeven.joblib"
    if classifier_path.exists():
        loaded["classifier"] = joblib.load(classifier_path)
    if roi_path.exists():
        loaded["roi_model"] = joblib.load(roi_path)
    if be_path.exists():
        loaded["breakeven_model"] = joblib.load(be_path)
    kmeans_path = model_dir / "kmeans.joblib"
    if kmeans_path.exists():
        loaded["kmeans"] = joblib.load(kmeans_path)
    shap_background = model_dir / "shap_background.joblib"
    if shap_background.exists():
        try:
            loaded["shap_background"] = joblib.load(shap_background)
        except Exception:
            pass
    loaded["available"] = all(key in loaded for key in ("classifier", "roi_model", "breakeven_model"))
    _MODEL_CACHE = loaded
    return loaded


def clear_model_cache() -> None:
    """Useful after retraining inside the same Python process."""
    global _MODEL_CACHE
    _MODEL_CACHE = None
    load_models.cache_clear()


def _normalize_target_market(value: Any) -> str:
    text = str(value or "b2c").strip().lower()
    if "b2b2c" in text or "b2b2c" in text.replace(" ", ""):
        return "b2b2c"
    if "marketplace" in text or "market place" in text or "platform" in text:
        return "marketplace"
    if "b2b" in text or "business" in text or "entreprise" in text or "شركات" in text:
        return "b2b"
    return "b2c"


def _normalize_revenue_model(value: Any) -> str:
    text = str(value or "one_time").strip().lower()
    aliases = {
        "abonnement": "subscription",
        "subscription": "subscription",
        "اشتراك": "subscription",
        "one time": "one_time",
        "one-time": "one_time",
        "once": "one_time",
        "commission": "commission",
        "عمولة": "commission",
        "freemium": "freemium",
        "مجاني": "freemium",
        "advertising": "advertising",
        "ads": "advertising",
        "إعلانات": "advertising",
        "hybrid": "hybrid",
        "مختلط": "hybrid",
    }
    for alias, model in aliases.items():
        if alias in text:
            return model
    return "one_time"


def _city_tier(location: str | None, preset_name: str, explicit: Any = None) -> int:
    if explicit is not None:
        try:
            return int(clamp(float(explicit), 1, 3))
        except (TypeError, ValueError):
            pass
    text = (location or "").lower()
    preset = get_preset(preset_name)
    cities = preset.get("cities", {}).get("en", [])
    if cities and text:
        if text.strip() == str(cities[0]).lower():
            return 1
        if text.strip() in [str(city).lower() for city in cities[:3]]:
            return 2
    return 2


def _funding_stage(budget: float) -> str:
    if budget < 15_000:
        return "bootstrapped"
    if budget < 45_000:
        return "friends_family"
    if budget < 150_000:
        return "seed"
    return "series_a"


def _normalized_inputs(inputs: dict[str, Any], preset_name: str) -> dict[str, Any]:
    preset = get_preset(preset_name)
    profile = get_sector_profile(str(inputs.get("sector", "other")))
    sector = str(inputs.get("sector", "other"))
    budget = max(500.0, safe_float(inputs.get("budget"), float(preset.get("typical_budget", 30000))))
    team_size = int(clamp(safe_float(inputs.get("team_size"), float(preset.get("typical_team_size", 3))), 1, 100))
    price = max(0.5, safe_float(inputs.get("price"), float(profile.get("typical_price", 50))))
    expected_customers = max(
        0.0,
        safe_float(inputs.get("expected_customers"), float(profile.get("typical_customers", 120))),
    )
    marketing_spend = max(
        0.0,
        safe_float(
            inputs.get("marketing_spend"),
            max(100.0, price * expected_customers * 0.08),
        ),
    )
    timeline_months = int(clamp(safe_float(inputs.get("timeline_months"), 9), 1, 60))
    competitor_intensity = clamp(safe_float(inputs.get("competitor_intensity"), float(profile.get("competition", 7))), 1, 10)
    differentiation = clamp(safe_float(inputs.get("differentiation"), 6), 1, 10)
    target_market = _normalize_target_market(inputs.get("target_market"))
    revenue_model = _normalize_revenue_model(inputs.get("revenue_model"))
    city_tier = _city_tier(inputs.get("location"), preset_name, inputs.get("city_tier"))
    founder_experience = clamp(safe_float(inputs.get("founder_experience_years"), 4), 0, 40)
    target_customers_text = str(inputs.get("target_customers", "") or "")

    return {
        "description": str(inputs.get("description", "") or ""),
        "sector": sector,
        "sector_label": sector_label(sector, str(inputs.get("language", "en"))),
        "target_market": target_market,
        "target_customers": target_customers_text,
        "location": str(inputs.get("location", "") or ""),
        "city_tier": city_tier,
        "budget": budget,
        "team_size": team_size,
        "price": price,
        "expected_customers": expected_customers,
        "marketing_spend": marketing_spend,
        "timeline_months": timeline_months,
        "competitor_intensity": competitor_intensity,
        "differentiation": differentiation,
        "revenue_model": revenue_model,
        "founder_experience_years": founder_experience,
        "funding_stage": _funding_stage(budget),
        "country_preset": preset_name,
    }


def build_feature_row(inputs: dict[str, Any], preset_name: str = "tunisia") -> dict[str, Any]:
    """Derive every model feature from user answers and documented sector assumptions."""
    norm = _normalized_inputs(inputs, preset_name)
    preset = get_preset(preset_name)
    profile = get_sector_profile(norm["sector"])
    market_factor = float(preset.get("market_factor", 1.0))
    target_market = norm["target_market"]

    target_size_adj = {"b2c": 0.55, "b2b": -0.15, "b2b2c": 0.2, "marketplace": 0.35}[target_market]
    target_fit_adj = {"b2c": 0.45, "b2b": 0.55, "b2b2c": 0.15, "marketplace": 0.25}[target_market]

    market_size_score = clamp(
        float(profile.get("market_size", 6.5)) + 1.2 * (market_factor - 1) + target_size_adj,
        1,
        10,
    )
    market_growth_score = clamp(
        float(profile.get("growth", 5.0)) + (0.3 if market_factor >= 0.95 else 0.0),
        1,
        10,
    )
    audience_fit = clamp(
        5.7
        + target_fit_adj
        + (0.2 if norm["target_customers"] else -0.1)
        + (0.25 if norm["city_tier"] == 1 else 0.0)
        + (0.2 if norm["differentiation"] >= 6 else -0.1),
        1,
        10,
    )
    operational_complexity = clamp(
        float(profile.get("operational", 6.0))
        - min(norm["team_size"], 15) * 0.09
        + (0.3 if norm["timeline_months"] < 5 else 0.0),
        1,
        10,
    )
    regulatory_risk = clamp(
        float(profile.get("regulatory", 5.0))
        + (0.35 if preset_name == "tunisia" else 0.0)
        + (0.25 if norm["sector"] in {"fintech", "healthcare", "real_estate"} else 0.0),
        1,
        10,
    )
    timing_score = clamp(
        float(profile.get("growth", 5.0)) * 0.7
        + 2.4
        - max(0, norm["timeline_months"] - 12) * 0.16
        + (0.3 if norm["timeline_months"] <= 8 else 0.0),
        1,
        10,
    )
    budget_ratio = norm["budget"] / max(float(preset.get("typical_budget", 30000)), 1)
    team_readiness = clamp(
        3.1
        + min(norm["team_size"], 15) * 0.28
        + min(norm["founder_experience_years"], 15) * 0.18
        + (0.45 if budget_ratio >= 1 else -0.35),
        1,
        10,
    )

    row = {
        **norm,
        "market_size_score": round(market_size_score, 3),
        "market_growth_score": round(market_growth_score, 3),
        "audience_fit": round(audience_fit, 3),
        "operational_complexity": round(operational_complexity, 3),
        "regulatory_risk": round(regulatory_risk, 3),
        "timing_score": round(timing_score, 3),
        "team_readiness": round(team_readiness, 3),
    }
    return row


def _subscores(feature_row: dict[str, Any], preset_name: str) -> dict[str, float]:
    preset = get_preset(preset_name)
    profile = get_sector_profile(feature_row["sector"])
    required_capital = (
        float(preset.get("typical_budget", 30000))
        * max(0.45, float(profile.get("capital_intensity", 5.5)) / 5.5)
    )
    capital_ratio = feature_row["budget"] / max(required_capital, 1)
    startup_cost = clamp(3.0 + 6.5 * min(capital_ratio / 1.2, 1.0), 1, 10)
    scalability = clamp(
        float(profile.get("scalability", 5.5))
        + _SCALABILITY_ADJUST.get(feature_row["revenue_model"], 0.0)
        + (0.5 if feature_row["target_market"] in {"b2c", "marketplace"} else 0.1),
        1,
        10,
    )
    competition_position = clamp(
        11 - feature_row["competitor_intensity"]
        + 0.12 * (feature_row["differentiation"] - 5),
        1,
        10,
    )
    operational_feasibility = clamp(
        11 - feature_row["operational_complexity"] + 0.08 * (feature_row["team_readiness"] - 5),
        1,
        10,
    )
    return {
        "market_size_demand": round(feature_row["market_size_score"], 2),
        "market_growth_trend": round(feature_row["market_growth_score"], 2),
        "target_audience_fit": round(feature_row["audience_fit"], 2),
        "competition": round(competition_position, 2),
        "differentiation": round(feature_row["differentiation"], 2),
        "startup_cost": round(startup_cost, 2),
        "operational_feasibility": round(operational_feasibility, 2),
        "revenue_model": round(_REVENUE_MODEL_SCORE.get(feature_row["revenue_model"], 5.5), 2),
        "scalability": round(scalability, 2),
        "timing": round(feature_row["timing_score"], 2),
        "regulatory_risk": round(clamp(11 - feature_row["regulatory_risk"], 1, 10), 2),
        "team_readiness": round(feature_row["team_readiness"], 2),
    }


def _weighted_score(subscores: dict[str, float]) -> float:
    """Weighted rubric on a 0–10 scale (weights sum to 1)."""
    return sum(float(subscores[item["key"]]) * float(item["weight"]) for item in CRITERIA)


def _feature_frame(feature_row: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{column: feature_row.get(column) for column in FEATURE_COLUMNS}])


def _predict_models(feature_row: dict[str, Any]) -> dict[str, Any]:
    models = load_models()
    frame = _feature_frame(feature_row)
    out: dict[str, Any] = {
        "available": bool(models.get("available")),
        "success_probability": None,
        "roi_model": None,
        "breakeven_model": None,
        "segment": None,
    }
    if not models.get("available"):
        return out
    try:
        out["success_probability"] = float(models["classifier"].predict_proba(frame)[0, 1])
    except Exception:
        out["success_probability"] = None
    try:
        out["roi_model"] = float(models["roi_model"].predict(frame)[0])
    except Exception:
        out["roi_model"] = None
    try:
        out["breakeven_model"] = float(models["breakeven_model"].predict(frame)[0])
    except Exception:
        out["breakeven_model"] = None
    kmeans = models.get("kmeans")
    if kmeans:
        try:
            from sklearn.preprocessing import StandardScaler  # noqa: F401

            cluster_input = pd.DataFrame(
                [{name: feature_row.get(name, 0) for name in kmeans["features"]}]
            )
            scaled = kmeans["scaler"].transform(cluster_input)
            cluster_id = int(kmeans["model"].predict(scaled)[0])
            out["segment"] = {
                "cluster_id": cluster_id,
                "name": kmeans.get("descriptions", {}).get(str(cluster_id), f"Segment {cluster_id + 1}"),
            }
        except Exception:
            pass
    return out


def _financials(feature_row: dict[str, Any], predictions: dict[str, Any], preset_name: str) -> dict[str, Any]:
    preset = get_preset(preset_name)
    profile = get_sector_profile(feature_row["sector"])
    city_factor = float(preset.get("city_factor", {}).get(str(feature_row["city_tier"]), 1.0))
    monthly_revenue = feature_row["price"] * feature_row["expected_customers"]
    gross_margin = clamp(
        float(profile.get("gross_margin", 0.5))
        + 0.016 * (feature_row["differentiation"] - 5.5)
        - 0.010 * (feature_row["competitor_intensity"] - 5.5),
        0.10,
        0.92,
    )
    fixed_costs = (
        float(preset.get("base_fixed_monthly", 800))
        + feature_row["team_size"] * float(preset.get("avg_monthly_salary", 1200))
        + float(preset.get("rent_monthly", 800)) * city_factor
    )
    monthly_profit = monthly_revenue * gross_margin - fixed_costs - feature_row["marketing_spend"]
    capital = max(feature_row["budget"], 1.0)
    roi_formula = monthly_profit * 12 / capital * 100
    breakeven_formula = capital / monthly_profit + 2.0 if monthly_profit > 30 else 48.0

    if predictions["roi_model"] is not None:
        roi_estimate = 0.68 * predictions["roi_model"] + 0.32 * roi_formula
    else:
        roi_estimate = roi_formula
    if predictions["breakeven_model"] is not None:
        breakeven = 0.60 * predictions["breakeven_model"] + 0.40 * breakeven_formula
    else:
        breakeven = breakeven_formula
    roi_estimate = clamp(roi_estimate, -100, 700)
    breakeven = clamp(breakeven, 1, 60)

    scenarios: dict[str, dict[str, float]] = {}
    for name, revenue_factor, cost_factor in (
        ("pessimistic", 0.72, 1.12),
        ("realistic", 1.0, 1.0),
        ("optimistic", 1.27, 0.94),
    ):
        scenario_profit = (
            monthly_revenue * revenue_factor * gross_margin
            - fixed_costs * cost_factor
            - feature_row["marketing_spend"] * (0.85 if name == "optimistic" else 1.05 if name == "pessimistic" else 1.0)
        )
        scenarios[name] = {
            "monthly_profit": float(scenario_profit),
            "annual_profit": float(scenario_profit * 12),
            "roi_percent": float(clamp(scenario_profit * 12 / capital * 100, -100, 700)),
            "breakeven_months": float(
                clamp(capital / scenario_profit + 2.0, 1, 60) if scenario_profit > 30 else 60
            ),
        }

    currency = currency_for(preset_name)
    return {
        "currency": currency,
        "monthly_revenue": float(monthly_revenue),
        "annual_revenue": float(monthly_revenue * 12),
        "gross_margin_pct": float(gross_margin * 100),
        "monthly_fixed_costs": float(fixed_costs),
        "monthly_marketing": float(feature_row["marketing_spend"]),
        "monthly_profit": float(monthly_profit),
        "annual_profit": float(monthly_profit * 12),
        "capital": float(capital),
        "roi_formula": float(roi_formula),
        "roi_model": predictions.get("roi_model"),
        "roi_percent": float(roi_estimate),
        "breakeven_formula": float(breakeven_formula),
        "breakeven_model": predictions.get("breakeven_model"),
        "breakeven_months": float(breakeven),
        "scenarios": scenarios,
        "assumptions": {
            "price": float(feature_row["price"]),
            "expected_customers": float(feature_row["expected_customers"]),
            "gross_margin_pct": float(gross_margin * 100),
            "fixed_costs": float(fixed_costs),
            "marketing_spend": float(feature_row["marketing_spend"]),
            "city_factor": city_factor,
        },
    }


def _confidence(predictions: dict[str, Any], feature_row: dict[str, Any], metadata: dict[str, Any]) -> float:
    probability = predictions.get("success_probability")
    if probability is None:
        return 48.0
    auc = 0.72
    try:
        auc = float(metadata["classification"]["test_metrics"]["roc_auc"])
    except (KeyError, TypeError, ValueError):
        pass
    signal = 52.0 + 38.0 * (2.0 * abs(probability - 0.5))
    model_quality = 0.80 + 0.20 * clamp((auc - 0.5) / 0.35, 0, 1)
    completeness = 0.86
    if feature_row.get("target_customers"):
        completeness += 0.04
    if feature_row.get("location"):
        completeness += 0.03
    if feature_row.get("description"):
        completeness += 0.04
    if feature_row.get("founder_experience_years") is not None:
        completeness += 0.03
    limits = get_config().get("scoring", {}).get("confidence", {})
    return float(clamp(signal * model_quality * completeness, limits.get("min", 35), limits.get("max", 96)))


def _local_explanations(
    feature_row: dict[str, Any],
    classifier: Pipeline,
    metadata: dict[str, Any],
    models: dict[str, Any],
    language: str,
) -> tuple[list[dict[str, Any]], str]:
    """Return local SHAP contributions, or a rubric-based fallback."""
    frame = _feature_frame(feature_row)
    try:
        import shap

        preprocessor = classifier.named_steps["preprocessor"]
        transformed = preprocessor.transform(frame)
        estimator = get_estimator(classifier)
        if hasattr(estimator, "feature_importances_"):
            explainer = shap.TreeExplainer(estimator)
        else:
            background = models.get("shap_background", {}).get("background")
            if background is None:
                raise RuntimeError("No SHAP background available")
            explainer = shap.LinearExplainer(estimator, background)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[0, :, -1]
        elif values.ndim == 2:
            values = values[0]
        names = list(preprocessor.get_feature_names_out())
        grouped: dict[str, float] = {}
        for name, value in zip(names, values):
            raw = raw_feature_from_transformed(str(name))
            grouped[raw] = grouped.get(raw, 0.0) + float(value)
        contributions = [
            {
                "feature": feature,
                "label": _feature_label(feature, language),
                "contribution": float(value),
                "direction": "positive" if value >= 0 else "negative",
            }
            for feature, value in grouped.items()
        ]
        contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        return contributions[:8], "SHAP"
    except Exception:
        pass

    # Transparent fallback: compare each rubric score with the weighted average.
    sub = _subscores(feature_row, feature_row["country_preset"])
    weights = {item["key"]: float(item["weight"]) for item in CRITERIA}
    average = sum(sub[key] * weights[key] for key in sub) / sum(weights.values())
    fallback = [
        {
            "feature": key,
            "label": t(next(item["label"] for item in CRITERIA if item["key"] == key), language),
            "contribution": float((sub[key] - average) * weights[key]),
            "direction": "positive" if sub[key] >= average else "negative",
        }
        for key in sub
        if key in weights
    ]
    fallback.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return fallback[:8], "rubric"


_FEATURE_TRANSLATIONS: dict[str, dict[str, str]] = {
    "sector": {"en": "Sector", "fr": "Secteur", "ar": "القطاع"},
    "target_market": {"en": "Target market", "fr": "Marché cible", "ar": "السوق المستهدف"},
    "revenue_model": {"en": "Revenue model", "fr": "Modèle de revenus", "ar": "نموذج الإيرادات"},
    "funding_stage": {"en": "Funding stage", "fr": "Stade de financement", "ar": "مرحلة التمويل"},
    "country_preset": {"en": "Country", "fr": "Pays", "ar": "البلد"},
    "city_tier": {"en": "City size", "fr": "Taille de la ville", "ar": "حجم المدينة"},
    "budget": {"en": "Budget", "fr": "Budget", "ar": "الميزانية"},
    "team_size": {"en": "Team size", "fr": "Taille de l'équipe", "ar": "حجم الفريق"},
    "price": {"en": "Price", "fr": "Prix", "ar": "السعر"},
    "competitor_intensity": {"en": "Competition intensity", "fr": "Intensité concurrentielle", "ar": "شدة المنافسة"},
    "differentiation": {"en": "Differentiation", "fr": "Différenciation", "ar": "التميّز"},
    "market_size_score": {"en": "Market size", "fr": "Taille du marché", "ar": "حجم السوق"},
    "market_growth_score": {"en": "Market growth", "fr": "Croissance du marché", "ar": "نمو السوق"},
    "audience_fit": {"en": "Audience fit", "fr": "Adéquation à la cible", "ar": "ملاءمة الجمهور"},
    "operational_complexity": {"en": "Operational complexity", "fr": "Complexité opérationnelle", "ar": "التعقيد التشغيلي"},
    "regulatory_risk": {"en": "Regulatory risk", "fr": "Risque réglementaire", "ar": "المخاطر التنظيمية"},
    "timing_score": {"en": "Timing", "fr": "Timing", "ar": "التوقيت"},
    "marketing_spend": {"en": "Marketing spend", "fr": "Dépenses marketing", "ar": "مصروف التسويق"},
    "expected_customers": {"en": "Expected customers", "fr": "Clients attendus", "ar": "العملاء المتوقعون"},
    "founder_experience_years": {"en": "Founder experience", "fr": "Expérience du fondateur", "ar": "خبرة المؤسس"},
    "team_readiness": {"en": "Team readiness", "fr": "Préparation de l'équipe", "ar": "جاهزية الفريق"},
}


def _feature_label(feature: str, language: str) -> str:
    entry = _FEATURE_TRANSLATIONS.get(feature)
    if entry:
        return entry.get(language, entry["en"])
    return feature_display_name(feature).replace("_", " ").title()


def _swot(
    sub: dict[str, float], feature_row: dict[str, Any], financials: dict[str, Any], language: str
) -> dict[str, list[str]]:
    labelled = sorted(
        ((item, sub[item["key"]]) for item in CRITERIA),
        key=lambda pair: pair[1],
        reverse=True,
    )
    strengths = [
        t("swot_strength_line", language, criterion=t(item["label"], language), score=round(score, 1))
        for item, score in labelled[:3]
        if score >= 6.3
    ][:3]
    weaknesses = [
        t("swot_weakness_line", language, criterion=t(item["label"], language), score=round(score, 1))
        for item, score in reversed(labelled[-3:])
        if score <= 6.0
    ][:3]
    opportunities: list[str] = []
    if sub["market_growth_trend"] >= 6:
        opportunities.append(t("swot_opportunity_growth", language, score=sub["market_growth_trend"]))
    if sub["timing"] >= 6:
        opportunities.append(t("swot_opportunity_timing", language))
    if feature_row["target_market"] in {"b2c", "b2b", "marketplace"}:
        opportunities.append(
            t("swot_opportunity_segment", language, market=feature_row["target_market"].upper())
        )
    if financials["monthly_profit"] > 0:
        opportunities.append(t("swot_opportunity_profit", language))
    if not opportunities:
        opportunities.append(t("swot_opportunity_generic", language))

    threats: list[str] = []
    if sub["competition"] <= 5.2:
        threats.append(t("swot_threat_competition", language, score=feature_row["competitor_intensity"]))
    if sub["regulatory_risk"] <= 5.5:
        threats.append(t("swot_threat_regulation", language))
    if sub["startup_cost"] <= 5.5:
        threats.append(t("swot_threat_capital", language))
    if sub["operational_feasibility"] <= 5.5:
        threats.append(t("swot_threat_operations", language))
    if sub["team_readiness"] <= 5.0:
        threats.append(t("swot_threat_team", language))
    if not threats:
        threats.append(t("swot_threat_generic", language))
    return {
        "strengths": strengths or [t("swot_strength_generic", language)],
        "weaknesses": weaknesses or [t("swot_weakness_generic", language)],
        "opportunities": opportunities[:4],
        "threats": threats[:4],
    }


def _risks(sub: dict[str, float], language: str) -> list[str]:
    candidates = [
        ("competition", 5.5, "risk_competition"),
        ("regulatory_risk", 5.5, "risk_regulation"),
        ("startup_cost", 5.5, "risk_capital"),
        ("operational_feasibility", 5.5, "risk_operations"),
        ("team_readiness", 5.2, "risk_team"),
        ("revenue_model", 5.5, "risk_revenue"),
        ("timing", 5.2, "risk_timing"),
        ("market_growth_trend", 5.0, "risk_growth"),
        ("differentiation", 5.2, "risk_differentiation"),
    ]
    risks = [t(key, language) for criterion, threshold, key in candidates if sub[criterion] < threshold]
    return risks[:6] or [t("risk_generic", language)]


def _action_specs(feature_row: dict[str, Any], sub: dict[str, float]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if sub["differentiation"] < 7:
        specs.append(
            {
                "key": "differentiation",
                "label": "action_differentiation",
                "changes": {"differentiation": min(10, feature_row["differentiation"] + 1.8)},
            }
        )
    if sub["competition"] < 6:
        specs.append(
            {
                "key": "competition",
                "label": "action_competition",
                "changes": {"competitor_intensity": max(1, feature_row["competitor_intensity"] - 1.6)},
            }
        )
    if sub["startup_cost"] < 6:
        specs.append(
            {
                "key": "capital",
                "label": "action_budget",
                "changes": {"budget": feature_row["budget"] * 1.3},
            }
        )
    if sub["team_readiness"] < 6.5:
        specs.append(
            {
                "key": "team",
                "label": "action_team",
                "changes": {
                    "team_size": feature_row["team_size"] + 1,
                    "founder_experience_years": feature_row["founder_experience_years"] + 1,
                },
            }
        )
    if sub["revenue_model"] < 7:
        specs.append(
            {
                "key": "revenue",
                "label": "action_revenue",
                "changes": {"revenue_model": "hybrid", "price": feature_row["price"] * 1.08},
            }
        )
    if sub["regulatory_risk"] < 6:
        specs.append(
            {
                "key": "regulation",
                "label": "action_regulation",
                "changes": {"differentiation": min(10, feature_row["differentiation"] + 0.7)},
                "flat_roi": -300.0,
            }
        )
    if sub["market_growth_trend"] < 5.5:
        specs.append(
            {
                "key": "growth",
                "label": "action_growth",
                "changes": {"target_market": "b2b" if feature_row["target_market"] == "b2c" else "b2c"},
            }
        )
    if sub["operational_feasibility"] < 5.5:
        specs.append(
            {
                "key": "operations",
                "label": "action_operations",
                "changes": {"team_size": feature_row["team_size"] + 1},
            }
        )
    if sub["timing"] < 5.5:
        specs.append(
            {
                "key": "timing",
                "label": "action_timing",
                "changes": {"timeline_months": max(2, feature_row["timeline_months"] - 3)},
            }
        )
    if sub["target_audience_fit"] < 6:
        specs.append(
            {
                "key": "audience",
                "label": "action_audience",
                "changes": {"target_customers": "validated niche segment"},
            }
        )
    return specs[:8]


def _raw_inputs_from_feature_row(feature_row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "description",
        "sector",
        "target_market",
        "target_customers",
        "location",
        "city_tier",
        "budget",
        "team_size",
        "price",
        "expected_customers",
        "marketing_spend",
        "timeline_months",
        "competitor_intensity",
        "differentiation",
        "revenue_model",
        "founder_experience_years",
        "country_preset",
    ]
    return {key: feature_row.get(key) for key in keys}


def _score_core(inputs: dict[str, Any], preset_name: str, language: str) -> dict[str, Any]:
    feature_row = build_feature_row(inputs, preset_name)
    sub = _subscores(feature_row, preset_name)
    weighted = _weighted_score(sub)
    predictions = _predict_models(feature_row)
    metadata = load_models().get("metadata", {})
    probability = predictions.get("success_probability")
    if probability is None:
        probability = float(1 / (1 + math.exp(-(weighted - 5.4) * 1.15)))
        predictions["success_probability"] = probability
        predictions["available"] = False
    ml_weight = float(get_config().get("scoring", {}).get("ml_weight", 0.55))
    overall = clamp(
        100 * (ml_weight * probability + (1 - ml_weight) * (weighted / 10)),
        0,
        100,
    )
    financials = _financials(feature_row, predictions, preset_name)
    confidence = _confidence(predictions, feature_row, metadata)
    return {
        "feature_row": feature_row,
        "subscores": sub,
        "weighted_rubric": weighted,
        "predictions": predictions,
        "success_probability": probability,
        "overall_score": overall,
        "financials": financials,
        "confidence": confidence,
        "metadata": metadata,
    }


def _verdict(score: float, language: str) -> tuple[str, str]:
    thresholds = get_config().get("scoring", {}).get("verdict_thresholds", {})
    succeed = float(thresholds.get("succeed", 68))
    risky = float(thresholds.get("risky", 45))
    if score >= succeed:
        return "succeed", t("verdict_succeed", language)
    if score >= risky:
        return "risky", t("verdict_risky", language)
    return "unlikely", t("verdict_unlikely", language)


def _gtm(feature_row: dict[str, Any], sub: dict[str, float], language: str) -> list[str]:
    market = feature_row["target_market"]
    location = feature_row["location"] or t("your_market", language)
    suggestions = [
        t(f"gtm_{market}", language, location=location),
        t("gtm_validate", language),
    ]
    if sub["competition"] <= 6:
        suggestions.append(t("gtm_niche", language))
    if sub["market_growth_trend"] >= 6:
        suggestions.append(t("gtm_growth", language))
    if feature_row["revenue_model"] in {"subscription", "freemium", "hybrid"}:
        suggestions.append(t("gtm_subscription", language))
    return suggestions[:5]


def _target_segment(feature_row: dict[str, Any], language: str) -> str:
    market = feature_row["target_market"]
    location = feature_row["location"] or t("your_market", language)
    return t(f"segment_{market}", language, location=location, sector=feature_row["sector_label"])


def _actions(
    feature_row: dict[str, Any],
    sub: dict[str, float],
    preset_name: str,
    language: str,
    base: dict[str, Any],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    raw_inputs = _raw_inputs_from_feature_row(feature_row)
    for spec in _action_specs(feature_row, sub):
        changed = dict(raw_inputs)
        changed.update(spec["changes"])
        try:
            simulated = _score_core(changed, preset_name, language)
            score_gain = simulated["overall_score"] - base["overall_score"]
            roi_gain = simulated["financials"]["roi_percent"] - base["financials"]["roi_percent"]
        except Exception:
            score_gain, roi_gain = 0.0, 0.0
        if "flat_roi" in spec:
            roi_gain += float(spec["flat_roi"])
        actions.append(
            {
                "key": spec["key"],
                "action": t(spec["label"], language),
                "score_gain": round(float(score_gain), 2),
                "roi_gain": round(float(roi_gain), 1),
                "priority": "high" if score_gain >= 2 else "medium" if score_gain >= 0.8 else "low",
                "cost_note": t(f"cost_note_{spec['key']}", language),
            }
        )
    actions.sort(key=lambda item: (item["score_gain"], item["roi_gain"]), reverse=True)
    return actions


def evaluate_idea(
    inputs: dict[str, Any],
    preset_name: str = "tunisia",
    language: str = "en",
    *,
    include_actions: bool = True,
    include_explanations: bool = True,
) -> dict[str, Any]:
    """Evaluate one idea and return a fully serializable report dictionary."""
    language = language if language in {"en", "fr", "ar"} else "en"
    if preset_name not in preset_names():
        preset_name = preset_names()[0]
    core = _score_core(inputs, preset_name, language)
    feature_row = core["feature_row"]
    sub = core["subscores"]
    financials = core["financials"]
    verdict_key, verdict = _verdict(core["overall_score"], language)
    models = load_models()
    explanations, source = ([], "rubric")
    if include_explanations and models.get("classifier") is not None:
        explanations, source = _local_explanations(
            feature_row, models["classifier"], core["metadata"], models, language
        )
    positives = [
        item["label"] for item in explanations if item["direction"] == "positive" and item["contribution"] > 0
    ][:3]
    negatives = [
        item["label"] for item in explanations if item["direction"] == "negative" and item["contribution"] < 0
    ][:3]
    if not positives:
        positives = [
            t(item["label"], language)
            for item, score in sorted(
                ((item, sub[item["key"]]) for item in CRITERIA), key=lambda pair: pair[1], reverse=True
            )[:3]
        ]
    if not negatives:
        negatives = [
            t(item["label"], language)
            for item, score in sorted(((item, sub[item["key"]]) for item in CRITERIA), key=lambda pair: pair[1])[:3]
        ]

    criterion_rows = [
        {
            "key": item["key"],
            "label": t(item["label"], language),
            "score": float(sub[item["key"]]),
            "weight": float(item["weight"]),
        }
        for item in CRITERIA
    ]
    swot = _swot(sub, feature_row, financials, language)
    risks = _risks(sub, language)
    actions = _actions(feature_row, sub, preset_name, language, core) if include_actions else []
    currency = financials["currency"]
    report = {
        "report_id": None,
        "mode": "idea",
        "language": language,
        "bot_name": get_bot_name(),
        "preset": preset_name,
        "currency": currency,
        "created_inputs": {
            key: value
            for key, value in feature_row.items()
            if key
            in {
                "description",
                "sector",
                "target_market",
                "target_customers",
                "location",
                "city_tier",
                "budget",
                "team_size",
                "price",
                "expected_customers",
                "marketing_spend",
                "timeline_months",
                "competitor_intensity",
                "differentiation",
                "revenue_model",
                "founder_experience_years",
            }
        },
        "title": (feature_row["description"][:70] + "…") if len(feature_row["description"]) > 70 else (feature_row["description"] or feature_row["sector_label"]),
        "sector": feature_row["sector"],
        "sector_label": feature_row["sector_label"],
        "location": feature_row["location"],
        "overall_score": round(float(core["overall_score"]), 1),
        "weighted_rubric": round(float(core["weighted_rubric"] * 10), 1),
        "success_probability": round(float(core["success_probability"] * 100), 1),
        "confidence": round(float(core["confidence"]), 1),
        "verdict_key": verdict_key,
        "verdict": verdict,
        "subscores": sub,
        "criteria": criterion_rows,
        "financials": financials,
        "scenarios": financials["scenarios"],
        "swot": swot,
        "key_risks": risks,
        "modifications": actions,
        "target_segment": _target_segment(feature_row, language),
        "go_to_market": _gtm(feature_row, sub, language),
        "explanations": explanations,
        "explanation_source": source,
        "positive_drivers": positives,
        "negative_drivers": negatives,
        "segment": core["predictions"].get("segment"),
        "model_available": bool(core["predictions"].get("available")),
        "disclaimer": t("disclaimer", language),
    }
    return report


def simulate_what_if(
    inputs: dict[str, Any],
    changes: dict[str, Any],
    preset_name: str = "tunisia",
    language: str = "en",
) -> dict[str, Any]:
    """Score a modified set of assumptions (same engine, no report side effects)."""
    modified = dict(inputs)
    modified.update(changes)
    base = evaluate_idea(inputs, preset_name, language, include_actions=False, include_explanations=False)
    scenario = evaluate_idea(modified, preset_name, language, include_actions=False, include_explanations=False)
    return {
        "base": base,
        "scenario": scenario,
        "delta_score": round(scenario["overall_score"] - base["overall_score"], 1),
        "delta_probability": round(scenario["success_probability"] - base["success_probability"], 1),
        "delta_roi": round(scenario["financials"]["roi_percent"] - base["financials"]["roi_percent"], 1),
        "delta_profit": round(
            scenario["financials"]["monthly_profit"] - base["financials"]["monthly_profit"], 1
        ),
    }


def compare_ideas(
    ideas: list[dict[str, Any]], preset_name: str = "tunisia", language: str = "en"
) -> list[dict[str, Any]]:
    return [evaluate_idea(idea, preset_name, language) for idea in ideas[:3]]
