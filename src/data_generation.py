"""Generate a documented, realistic synthetic dataset for the ML course project.

No real Kaggle data is bundled. The generator is explicit about being synthetic
and uses sector-level assumptions from ``configs/markets.yaml`` (margins,
competition, capital intensity, growth, risks...). The resulting CSV uses the
same schema expected by :mod:`src.preprocessing`, so a real dataset with these
columns can be loaded instead.

Schema
------
Features: sector, target_market, revenue_model, funding_stage, country_preset,
city_tier, budget, team_size, price, competitor_intensity, differentiation,
market_size_score, market_growth_score, audience_fit, operational_complexity,
regulatory_risk, timing_score, marketing_spend, expected_customers,
founder_experience_years, team_readiness.
Targets: success (binary), roi_percent (regression), breakeven_months (regression).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    ROOT_DIR,
    ensure_directories,
    get_markets,
    get_paths,
    get_preset,
    get_sector_profile,
    sector_names,
)
from .utils import clamp

TARGET_MARKETS = ("b2c", "b2b", "b2b2c", "marketplace")
REVENUE_MODELS = ("subscription", "one_time", "commission", "freemium", "advertising", "hybrid")
FUNDING_STAGES = ("bootstrapped", "friends_family", "seed", "series_a")

_TARGET_MARKET_WEIGHTS: dict[str, tuple[float, float, float, float]] = {
    "food_beverage": (0.82, 0.08, 0.04, 0.06),
    "retail": (0.72, 0.12, 0.04, 0.12),
    "ecommerce": (0.70, 0.16, 0.03, 0.11),
    "saas": (0.25, 0.62, 0.05, 0.08),
    "fintech": (0.32, 0.58, 0.04, 0.06),
    "education": (0.55, 0.36, 0.05, 0.04),
    "healthcare": (0.54, 0.36, 0.06, 0.04),
    "logistics": (0.22, 0.67, 0.04, 0.07),
    "tourism": (0.78, 0.13, 0.03, 0.06),
    "renewable_energy": (0.10, 0.76, 0.08, 0.06),
    "creative_services": (0.24, 0.67, 0.04, 0.05),
    "manufacturing": (0.15, 0.72, 0.07, 0.06),
    "agriculture": (0.36, 0.50, 0.07, 0.07),
    "real_estate": (0.34, 0.48, 0.04, 0.14),
    "fitness_wellness": (0.76, 0.15, 0.04, 0.05),
    "other": (0.48, 0.36, 0.08, 0.08),
}

_REVENUE_MODEL_WEIGHTS: dict[str, tuple[float, float, float, float, float, float]] = {
    # subscription, one_time, commission, freemium, advertising, hybrid
    "food_beverage": (0.04, 0.72, 0.03, 0.02, 0.01, 0.18),
    "retail": (0.03, 0.76, 0.03, 0.01, 0.01, 0.16),
    "ecommerce": (0.08, 0.60, 0.10, 0.04, 0.03, 0.15),
    "saas": (0.62, 0.05, 0.03, 0.18, 0.02, 0.10),
    "fintech": (0.38, 0.08, 0.24, 0.10, 0.02, 0.18),
    "education": (0.28, 0.42, 0.04, 0.12, 0.02, 0.12),
    "healthcare": (0.24, 0.50, 0.05, 0.04, 0.01, 0.16),
    "logistics": (0.05, 0.48, 0.34, 0.02, 0.02, 0.09),
    "tourism": (0.04, 0.64, 0.16, 0.02, 0.02, 0.12),
    "renewable_energy": (0.12, 0.55, 0.08, 0.02, 0.01, 0.22),
    "creative_services": (0.20, 0.40, 0.05, 0.03, 0.02, 0.30),
    "manufacturing": (0.04, 0.66, 0.04, 0.01, 0.01, 0.24),
    "agriculture": (0.02, 0.70, 0.06, 0.01, 0.01, 0.20),
    "real_estate": (0.03, 0.42, 0.42, 0.01, 0.02, 0.10),
    "fitness_wellness": (0.38, 0.40, 0.04, 0.08, 0.02, 0.08),
    "other": (0.08, 0.52, 0.10, 0.05, 0.03, 0.22),
}

_REVENUE_MODEL_STRENGTH = {
    "subscription": 8.4,
    "hybrid": 7.8,
    "commission": 7.0,
    "freemium": 6.1,
    "one_time": 5.4,
    "advertising": 4.7,
}


def _weights(values: tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr / arr.sum()


def _sector_choice(rng: np.random.Generator, country: str) -> str:
    preset = get_preset(country)
    weights = preset.get("sector_weights", {})
    keys = list(weights.keys()) or sector_names()
    probs = _weights(tuple(float(weights.get(key, 1.0)) for key in keys))
    return str(rng.choice(keys, p=probs))


def _funding_stage(budget: float) -> str:
    if budget < 15_000:
        return "bootstrapped"
    if budget < 45_000:
        return "friends_family"
    if budget < 150_000:
        return "seed"
    return "series_a"


def _round_price(value: float) -> float:
    if value >= 500:
        return float(round(value / 10) * 10)
    if value >= 50:
        return float(round(value / 5) * 5)
    return float(max(1.0, round(value, 1)))


def generate_synthetic_dataset(n_rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Create a reproducible synthetic dataset of business ideas."""
    if n_rows < 100:
        raise ValueError("n_rows must be at least 100")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    sectors = sector_names()
    countries = ("tunisia", "generic")
    country_probs = np.array([0.64, 0.36])

    for _ in range(n_rows):
        country = str(rng.choice(countries, p=country_probs))
        preset = get_preset(country)
        sector = _sector_choice(rng, country)
        profile = get_sector_profile(sector)
        sector_weights = preset.get("sector_weights", {})
        cities = preset.get("cities", {}).get("en", preset.get("cities", ["Capital"]))
        city_tier = int(rng.choice([1, 2, 3], p=[0.46, 0.34, 0.20]))
        location = str(rng.choice(cities[: max(1, min(len(cities), 5))]))

        market_weights = _TARGET_MARKET_WEIGHTS.get(sector, _TARGET_MARKET_WEIGHTS["other"])
        target_market = str(rng.choice(TARGET_MARKETS, p=_weights(market_weights)))
        revenue_model = str(
            rng.choice(REVENUE_MODELS, p=_weights(_REVENUE_MODEL_WEIGHTS.get(sector, _REVENUE_MODEL_WEIGHTS["other"])))
        )

        capital_factor = float(profile.get("capital_intensity", 5.5)) / 5.5
        typical_budget = float(preset.get("typical_budget", 30_000)) * max(0.45, capital_factor)
        budget = float(np.clip(rng.lognormal(np.log(typical_budget), 0.62), 2_500, 1_200_000))
        team_size = int(np.clip(round(rng.lognormal(np.log(float(preset.get("typical_team_size", 3))), 0.55)), 1, 45))
        founder_experience_years = float(np.clip(rng.gamma(2.2, 2.4), 0, 30))

        price = _round_price(float(profile.get("typical_price", 80)) * float(rng.lognormal(0, 0.42)))
        expected_customers = float(
            np.clip(
                float(profile.get("typical_customers", 150)) * float(rng.lognormal(0, 0.52)),
                3,
                25_000,
            )
        )
        expected_customers = round(expected_customers)

        monthly_revenue = price * expected_customers
        marketing_spend = float(
            np.clip(
                monthly_revenue * rng.uniform(0.025, 0.14)
                + float(preset.get("typical_marketing", 600)) * rng.uniform(0.4, 1.2),
                50,
                500_000,
            )
        )

        market_size_score = float(
            np.clip(
                float(profile.get("market_size", 6.5))
                + float(preset.get("market_factor", 1.0)) * 1.2
                - 1.2
                + (0.45 if target_market in {"b2c", "marketplace"} else 0.0)
                + rng.normal(0, 1.05),
                1,
                10,
            )
        )
        growth_score = float(np.clip(float(profile.get("growth", 5.0)) + rng.normal(0, 1.1), 1, 10))
        audience_fit = float(
            np.clip(
                5.8
                + (0.6 if target_market in {"b2c", "b2b"} else 0.1)
                + (0.25 if city_tier == 1 else 0.0)
                + rng.normal(0, 1.4),
                1,
                10,
            )
        )
        competitor_intensity = float(
            np.clip(float(profile.get("competition", 7.0)) + rng.normal(0, 1.35), 1, 10)
        )
        differentiation = float(np.clip(rng.normal(5.8, 1.95), 1, 10))
        operational_complexity = float(
            np.clip(
                float(profile.get("operational", 6.0))
                - min(team_size, 12) * 0.08
                + rng.normal(0, 1.15),
                1,
                10,
            )
        )
        regulatory_risk = float(
            np.clip(
                float(profile.get("regulatory", 5.0))
                + (0.35 if country == "tunisia" else 0.0)
                + rng.normal(0, 1.1),
                1,
                10,
            )
        )
        timing_score = float(
            np.clip(float(profile.get("growth", 5.0)) * 0.72 + 2.4 + rng.normal(0, 1.15), 1, 10)
        )
        budget_ratio = budget / max(typical_budget, 1)
        team_readiness = float(
            np.clip(
                3.2
                + min(team_size, 15) * 0.27
                + min(founder_experience_years, 15) * 0.18
                + (0.5 if budget_ratio >= 1 else -0.4)
                + rng.normal(0, 1.0),
                1,
                10,
            )
        )

        revenue_strength = _REVENUE_MODEL_STRENGTH.get(revenue_model, 5.5)
        target_effect = {"b2c": 0.10, "b2b": 0.18, "b2b2c": 0.06, "marketplace": 0.12}[target_market]
        country_effect = -0.08 if country == "tunisia" else 0.08

        # Latent success model: sector assumptions + observable execution variables.
        z = (
            -0.78
            + 0.31 * (market_size_score - 5.5)
            + 0.22 * (growth_score - 5.5)
            + 0.27 * (audience_fit - 5.5)
            - 0.28 * (competitor_intensity - 5.5)
            + 0.36 * (differentiation - 5.5)
            + 0.25 * (revenue_strength - 5.5)
            + 0.20 * (timing_score - 5.5)
            - 0.16 * (regulatory_risk - 5.5)
            - 0.14 * (operational_complexity - 5.5)
            + 0.27 * (team_readiness - 5.5)
            + 0.12 * np.log1p(founder_experience_years)
            + 0.08 * np.log1p(budget / 10_000)
            + target_effect
            + country_effect
            + rng.normal(0, 0.55)
        )

        # Unit economics used to create realistic ROI/break-even targets.
        gross_margin = float(
            np.clip(
                float(profile.get("gross_margin", 0.5))
                + 0.018 * (differentiation - 5.5)
                - 0.012 * (competitor_intensity - 5.5)
                + rng.normal(0, 0.035),
                0.10,
                0.92,
            )
        )
        city_factor = float(preset.get("city_factor", {}).get(str(city_tier), 1.0))
        fixed_costs = (
            float(preset.get("base_fixed_monthly", 800))
            + team_size * float(preset.get("avg_monthly_salary", 1200))
            + float(preset.get("rent_monthly", 800)) * city_factor
        )
        monthly_profit = monthly_revenue * gross_margin - fixed_costs - marketing_spend
        monthly_profit *= float(np.clip(1 + 0.08 * (differentiation - 5.5), 0.65, 1.4))
        annual_profit = monthly_profit * 12
        capital = max(budget, 1)
        roi_percent = float(np.clip(annual_profit / capital * 100 + rng.normal(0, 18), -95, 650))
        if monthly_profit > 50:
            breakeven_months = float(np.clip(capital / monthly_profit + 2.5 + rng.normal(0, 2.2), 1, 60))
        else:
            breakeven_months = float(np.clip(36 + abs(monthly_profit) / 350 + rng.normal(0, 6), 18, 60))

        profit_signal = float(np.clip(monthly_profit / max(capital / 18, 250), -2.0, 2.5))
        z += 0.30 * profit_signal + rng.normal(0, 0.28)
        success_probability = 1.0 / (1.0 + np.exp(-z))
        success = int(rng.random() < success_probability)

        rows.append(
            {
                "idea_id": f"IDEA-{len(rows) + 1:06d}",
                "sector": sector,
                "target_market": target_market,
                "revenue_model": revenue_model,
                "funding_stage": _funding_stage(budget),
                "country_preset": country,
                "country": "Tunisia" if country == "tunisia" else "Generic",
                "city": location,
                "city_tier": city_tier,
                "budget": round(budget, 2),
                "team_size": team_size,
                "price": round(price, 2),
                "competitor_intensity": round(competitor_intensity, 2),
                "differentiation": round(differentiation, 2),
                "market_size_score": round(market_size_score, 2),
                "market_growth_score": round(growth_score, 2),
                "audience_fit": round(audience_fit, 2),
                "operational_complexity": round(operational_complexity, 2),
                "regulatory_risk": round(regulatory_risk, 2),
                "timing_score": round(timing_score, 2),
                "marketing_spend": round(marketing_spend, 2),
                "expected_customers": expected_customers,
                "founder_experience_years": round(founder_experience_years, 2),
                "team_readiness": round(team_readiness, 2),
                "success": success,
                "roi_percent": round(roi_percent, 2),
                "breakeven_months": round(breakeven_months, 2),
            }
        )

    df = pd.DataFrame(rows)

    # A little realistic missingness so the preprocessing pipeline is meaningful.
    for column, fraction in {
        "budget": 0.006,
        "team_size": 0.005,
        "founder_experience_years": 0.008,
        "marketing_spend": 0.004,
    }.items():
        mask = rng.random(len(df)) < fraction
        df.loc[mask, column] = np.nan

    return df


def generate_business_history(
    months: int = 24,
    sector: str = "ecommerce",
    preset_name: str = "tunisia",
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a monthly operating history used as a demo dataset for Mode B."""
    rng = np.random.default_rng(seed)
    profile = get_sector_profile(sector)
    preset = get_preset(preset_name)
    end = pd.Timestamp.today().normalize().replace(day=1)
    dates = pd.date_range(end=end, periods=months, freq="MS")

    base_customers = 120.0
    growth = 0.035
    price = float(profile.get("typical_price", 70))
    margin = float(profile.get("gross_margin", 0.5))
    fixed = (
        float(preset.get("base_fixed_monthly", 750))
        + float(preset.get("typical_team_size", 3)) * float(preset.get("avg_monthly_salary", 1200))
        + float(preset.get("rent_monthly", 800))
    )

    records: list[dict[str, Any]] = []
    customers = base_customers
    for index, month in enumerate(dates):
        seasonal = 1 + 0.09 * np.sin((index + 2) / 12 * 2 * np.pi)
        customers = max(8, customers * (1 + growth + rng.normal(0, 0.018)))
        effective_customers = customers * seasonal
        revenue = price * effective_customers * (1 + rng.normal(0, 0.035))
        cogs = revenue * (1 - margin)
        marketing = revenue * (0.07 + 0.015 * rng.random())
        cost = fixed + cogs + marketing + revenue * rng.uniform(0.015, 0.035)
        new_customers = max(1, round(customers * (0.13 + rng.normal(0, 0.02))))
        churn = float(np.clip(rng.normal(0.045, 0.008), 0.015, 0.11))
        records.append(
            {
                "month": month.strftime("%Y-%m-%d"),
                "revenue": round(float(revenue), 2),
                "cost": round(float(cost), 2),
                "customers": round(float(effective_customers), 1),
                "new_customers": int(new_customers),
                "marketing_spend": round(float(marketing), 2),
                "price": round(price, 2),
                "churn_rate": round(churn, 4),
            }
        )
    return pd.DataFrame(records)


def save_dataset(df: pd.DataFrame, output: Path | None = None, seed: int = 42) -> Path:
    ensure_directories()
    paths = get_paths()
    output = output or (paths.data / "synthetic_businesses.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8")
    metadata = {
        "synthetic": True,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "success_rate": round(float(df["success"].mean()), 4),
        "note": (
            "Synthetic dataset generated from documented sector assumptions in configs/markets.yaml. "
            "It is not observed market data."
        ),
    }
    metadata_path = paths.data / "synthetic_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return output


def write_sample_business_file() -> Path:
    ensure_directories()
    path = get_paths().data / "sample_business.csv"
    df = generate_business_history()
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic CEO AI dataset.")
    parser.add_argument("--rows", type=int, default=6000, help="Number of synthetic business ideas.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-sample-business", action="store_true")
    args = parser.parse_args()

    df = generate_synthetic_dataset(n_rows=args.rows, seed=args.seed)
    output = save_dataset(df, output=args.output, seed=args.seed)
    sample = None if args.no_sample_business else write_sample_business_file()
    print(f"Saved {len(df):,} synthetic rows to {output}")
    print(f"Success rate: {df['success'].mean():.1%}")
    print(f"Mean ROI: {df['roi_percent'].mean():.1f}% | Mean break-even: {df['breakeven_months'].mean():.1f} months")
    if sample:
        print(f"Demo business history: {sample}")


if __name__ == "__main__":
    main()
