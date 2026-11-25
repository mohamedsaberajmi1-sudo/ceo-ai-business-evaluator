"""Dataset loading, validation and reusable scikit-learn preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_FEATURES: list[str] = [
    "sector",
    "target_market",
    "revenue_model",
    "funding_stage",
    "country_preset",
]

NUMERIC_FEATURES: list[str] = [
    "city_tier",
    "budget",
    "team_size",
    "price",
    "competitor_intensity",
    "differentiation",
    "market_size_score",
    "market_growth_score",
    "audience_fit",
    "operational_complexity",
    "regulatory_risk",
    "timing_score",
    "marketing_spend",
    "expected_customers",
    "founder_experience_years",
    "team_readiness",
]

FEATURE_COLUMNS: list[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES
CLASSIFICATION_TARGET = "success"
REGRESSION_TARGETS = ("roi_percent", "breakeven_months")
TARGET_COLUMNS = (CLASSIFICATION_TARGET, *REGRESSION_TARGETS)

REQUIRED_COLUMNS = set(FEATURE_COLUMNS) | set(TARGET_COLUMNS)


def build_preprocessor(*, scale: bool = True) -> ColumnTransformer:
    """Build the shared preprocessing transformer.

    Numeric values are median-imputed (and optionally standardized); categorical
    values are mode-imputed and one-hot encoded with unknown categories ignored.
    """
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_classification_pipeline(estimator: Any, *, scale: bool = True) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor(scale=scale)), ("classifier", estimator)])


def build_regression_pipeline(estimator: Any, *, scale: bool = True) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor(scale=scale)), ("regressor", estimator)])


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {column: str(column).strip().lower().replace(" ", "_") for column in df.columns}
    return df.rename(columns=mapping)


def load_dataset(
    path: str | Path | None = None,
    *,
    require_targets: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    """Load a CSV/Excel dataset using the shared schema.

    If *path* is omitted, ``data/synthetic_businesses.csv`` is used. If
    ``data/real_businesses.csv`` exists it is not loaded implicitly; pass it
    explicitly with ``--dataset`` to train on real data.
    """
    from .config import get_paths

    paths = get_paths()
    if path is None:
        path = paths.data / "synthetic_businesses.csv"
    path = Path(path)
    if not path.is_absolute():
        path = paths.data.parent / path
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run `python -m src.data_generation` or pass --dataset."
        )

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df = _normalize_columns(df)

    required = set(FEATURE_COLUMNS)
    if require_targets:
        required |= set(TARGET_COLUMNS)
    missing = sorted(required - set(df.columns))
    if missing and strict:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing)
            + ". See README.md for the schema; synthetic data already matches it."
        )

    for column in CATEGORICAL_FEATURES:
        if column in df.columns:
            df[column] = df[column].astype("string").fillna("other").astype(str)
    for column in NUMERIC_FEATURES + list(TARGET_COLUMNS):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def validate_feature_frame(df: pd.DataFrame) -> None:
    """Raise a clear error if prediction rows do not contain all feature columns."""
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(missing)}")


def split_xy(
    df: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X, y and the original rows aligned with X (useful for reports)."""
    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")
    clean = df.dropna(subset=[target]).reset_index(drop=True)
    X = clean[FEATURE_COLUMNS].copy()
    y = clean[target].astype(float if target != CLASSIFICATION_TARGET else int).copy()
    return X, y, clean


def feature_display_name(raw_name: str) -> str:
    """Turn ``cat__sector_saas`` or ``num__budget`` into a readable feature."""
    name = raw_name
    if name.startswith("num__"):
        name = name[5:]
    elif name.startswith("cat__"):
        name = name[5:]
    for prefix in CATEGORICAL_FEATURES:
        if name.startswith(prefix + "_"):
            value = name[len(prefix) + 1 :]
            return f"{prefix}: {value}"
    return name.replace("_", " ")


def raw_feature_from_transformed(raw_name: str) -> str:
    name = raw_name
    if name.startswith("num__"):
        return name[5:]
    if name.startswith("cat__"):
        name = name[5:]
    for prefix in CATEGORICAL_FEATURES:
        if name == prefix or name.startswith(prefix + "_"):
            return prefix
    return name


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        raise ValueError("Pipeline has no preprocessor step")
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        return []


def get_estimator(pipeline: Pipeline) -> Any:
    for name in ("classifier", "regressor", "model"):
        if name in pipeline.named_steps:
            return pipeline.named_steps[name]
    return pipeline.steps[-1][1]


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = None
    return out
