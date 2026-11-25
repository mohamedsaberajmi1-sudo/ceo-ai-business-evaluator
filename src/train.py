"""Train, tune, compare and persist all ML models for CEO AI.

Run with:
    python -m src.train
    python -m src.train --quick
    python -m src.train --dataset data/real_businesses.csv

Artifacts saved in ``models/``:
    classifier.joblib, regression_roi.joblib, regression_breakeven.joblib,
    kmeans.joblib, metadata.json, model_comparison.csv,
    feature_importance_classification.csv, feature_importance_*.csv,
    shap_background.joblib, roc_curve.csv, confusion_matrix.csv,
    test_predictions_*.csv
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ensure_directories, get_markets, get_paths
from .data_generation import generate_synthetic_dataset, save_dataset
from .preprocessing import (
    CATEGORICAL_FEATURES,
    CLASSIFICATION_TARGET,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    REGRESSION_TARGETS,
    build_classification_pipeline,
    build_regression_pipeline,
    feature_display_name,
    get_estimator,
    load_dataset,
    raw_feature_from_transformed,
    transformed_feature_names,
)
from .utils import clamp, json_safe

RANDOM_STATE = 42
CLUSTER_FEATURES = [
    "market_size_score",
    "market_growth_score",
    "audience_fit",
    "competitor_intensity",
    "differentiation",
    "budget",
    "team_readiness",
]


def _optional_xgboost() -> tuple[Any | None, Any | None]:
    try:
        from xgboost import XGBClassifier, XGBRegressor

        return XGBClassifier, XGBRegressor
    except Exception:  # pragma: no cover - optional dependency
        return None, None


def _classification_models(seed: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "Logistic Regression (baseline)": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=240,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=170,
            learning_rate=0.06,
            max_depth=3,
            subsample=0.9,
            random_state=seed,
        ),
    }
    xgb_cls, _ = _optional_xgboost()
    if xgb_cls is not None:
        models["XGBoost"] = xgb_cls(
            n_estimators=240,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=seed,
        )
    return models


def _regression_models(seed: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "Ridge Regression (baseline)": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=220,
            min_samples_leaf=3,
            max_features=0.7,
            n_jobs=-1,
            random_state=seed,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=160,
            learning_rate=0.06,
            max_depth=3,
            subsample=0.9,
            random_state=seed,
        ),
    }
    _, xgb_reg = _optional_xgboost()
    if xgb_reg is not None:
        models["XGBoost Regressor"] = xgb_reg(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=-1,
            random_state=seed,
        )
    return models


def _search_space(model: Any) -> tuple[dict[str, list[Any]], str]:
    name = type(model).__name__
    if name == "LogisticRegression":
        return {"classifier__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}, "roc_auc"
    if name == "RandomForestClassifier":
        return (
            {
                "classifier__n_estimators": [200, 320, 450],
                "classifier__max_depth": [None, 9, 15, 24],
                "classifier__min_samples_leaf": [1, 2, 4, 8],
                "classifier__max_features": ["sqrt", 0.35, 0.6],
            },
            "roc_auc",
        )
    if name == "GradientBoostingClassifier":
        return (
            {
                "classifier__n_estimators": [120, 180, 260, 340],
                "classifier__learning_rate": [0.03, 0.05, 0.08, 0.12],
                "classifier__max_depth": [2, 3, 4],
                "classifier__subsample": [0.75, 0.9, 1.0],
            },
            "roc_auc",
        )
    if name == "XGBClassifier":
        return (
            {
                "classifier__n_estimators": [180, 280, 400],
                "classifier__max_depth": [3, 4, 5, 7],
                "classifier__learning_rate": [0.03, 0.05, 0.08],
                "classifier__subsample": [0.75, 0.9, 1.0],
                "classifier__colsample_bytree": [0.7, 0.9, 1.0],
            },
            "roc_auc",
        )
    if name == "Ridge":
        return {"regressor__alpha": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]}, "neg_mean_absolute_error"
    if name == "RandomForestRegressor":
        return (
            {
                "regressor__n_estimators": [180, 300, 420],
                "regressor__max_depth": [None, 9, 15, 24],
                "regressor__min_samples_leaf": [1, 2, 4, 8],
                "regressor__max_features": [0.5, 0.75, 1.0],
            },
            "neg_mean_absolute_error",
        )
    if name == "GradientBoostingRegressor":
        return (
            {
                "regressor__n_estimators": [120, 180, 260, 340],
                "regressor__learning_rate": [0.03, 0.05, 0.08, 0.12],
                "regressor__max_depth": [2, 3, 4],
                "regressor__subsample": [0.75, 0.9, 1.0],
            },
            "neg_mean_absolute_error",
        )
    if name == "XGBRegressor":
        return (
            {
                "regressor__n_estimators": [180, 280, 400],
                "regressor__max_depth": [3, 4, 5, 7],
                "regressor__learning_rate": [0.03, 0.05, 0.08],
                "regressor__subsample": [0.75, 0.9, 1.0],
                "regressor__colsample_bytree": [0.7, 0.9, 1.0],
            },
            "neg_mean_absolute_error",
        )
    return {}, "neg_mean_absolute_error"


def _evaluate_classifier_cv(
    X: pd.DataFrame, y: pd.Series, seed: int
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    for name, estimator in _classification_models(seed).items():
        pipeline = build_classification_pipeline(estimator, scale=True)
        start = time.perf_counter()
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1)
        elapsed = time.perf_counter() - start
        pipeline.fit(X, y)
        fitted[name] = pipeline
        row: dict[str, Any] = {"model": name, "kind": "classification", "fit_seconds": round(elapsed, 2)}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        rows.append(row)
        print(
            f"  [CV] {name:<30} ROC-AUC={row['roc_auc_mean']:.4f} "
            f"F1={row['f1_mean']:.4f} accuracy={row['accuracy_mean']:.4f}"
        )
    return pd.DataFrame(rows), fitted


def _evaluate_regressor_cv(
    X: pd.DataFrame, y: pd.Series, seed: int
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    cv = KFold(n_splits=3, shuffle=True, random_state=seed)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    for name, estimator in _regression_models(seed).items():
        pipeline = build_regression_pipeline(estimator, scale=True)
        start = time.perf_counter()
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1)
        elapsed = time.perf_counter() - start
        pipeline.fit(X, y)
        fitted[name] = pipeline
        row: dict[str, Any] = {"model": name, "kind": "regression", "fit_seconds": round(elapsed, 2)}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            if metric in {"mae", "rmse"}:
                values = -values
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        rows.append(row)
        print(
            f"  [CV] {name:<30} MAE={row['mae_mean']:.2f} "
            f"RMSE={row['rmse_mean']:.2f} R2={row['r2_mean']:.4f}"
        )
    return pd.DataFrame(rows), fitted


def _cross_validate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    kind: str,
    seed: int,
    splits: int = 3,
) -> dict[str, float]:
    """Return mean/std CV metrics for an already-configured pipeline.

    Used to report the tuned estimator honestly instead of copying the
    baseline row and only patching one metric.
    """
    if kind == "classification":
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        }
    else:
        cv = KFold(n_splits=splits, shuffle=True, random_state=seed)
        scoring = {
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        }
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1)
    row: dict[str, float] = {}
    for metric in scoring:
        values = scores[f"test_{metric}"]
        if metric in {"mae", "rmse"}:
            values = -values
        row[f"{metric}_mean"] = float(np.mean(values))
        row[f"{metric}_std"] = float(np.std(values))
    return row


def _tune(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
    n_iter: int,
    cv_splits: int,
    kind: str,
) -> tuple[Pipeline, dict[str, Any]]:
    estimator = get_estimator(pipeline)
    space, scoring = _search_space(estimator)
    if not space:
        return pipeline, {}
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed) if kind == "classification" else KFold(
        n_splits=cv_splits, shuffle=True, random_state=seed
    )
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=space,
        n_iter=min(n_iter, int(np.prod([len(v) for v in space.values()]))),
        scoring=scoring,
        cv=cv,
        random_state=seed,
        n_jobs=1,
        refit=True,
        return_train_score=False,
    )
    start = time.perf_counter()
    search.fit(X, y)
    elapsed = time.perf_counter() - start
    print(f"  [TUNE] {type(estimator).__name__}: best {search.best_score_:.4f} in {elapsed:.1f}s")
    print(f"         {search.best_params_}")
    details = {
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "scoring": scoring,
        "seconds": round(elapsed, 2),
        "candidates": int(search.n_iter),
    }
    return search.best_estimator_, details


def _metrics_classification(y_true: pd.Series, proba: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "classification_report": classification_report(y_true, pred, output_dict=True, zero_division=0),
    }


def _metrics_regression(y_true: pd.Series, pred: np.ndarray) -> dict[str, Any]:
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    non_zero = np.abs(np.asarray(y_true, dtype=float)) > 1e-8
    mape = float(np.mean(np.abs((np.asarray(y_true)[non_zero] - pred[non_zero]) / np.asarray(y_true)[non_zero])) * 100) if non_zero.any() else None
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, pred)),
        "mape": mape,
    }


def _permutation_importance_table(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    seed: int,
    scoring: str,
    n_repeats: int = 5,
) -> pd.DataFrame:
    result = permutation_importance(
        pipeline, X_test, y_test, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=1
    )
    table = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return table


def _shap_analysis(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    model_dir: Path,
    *,
    seed: int,
) -> pd.DataFrame | None:
    """Compute global SHAP importance when the installed SHAP version supports it."""
    try:
        import shap

        preprocessor = pipeline.named_steps["preprocessor"]
        sample = X_train.sample(min(160, len(X_train)), random_state=seed)
        transformed = preprocessor.transform(sample)
        model = get_estimator(pipeline)
        background = preprocessor.transform(
            X_train.sample(min(120, len(X_train)), random_state=seed + 1)
        )
        joblib.dump({"background": background, "feature_names": transformed_feature_names(pipeline)}, model_dir / "shap_background.joblib")

        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, background)
        else:
            return None
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        names = transformed_feature_names(pipeline)
        importance = np.abs(values).mean(axis=0)
        table = pd.DataFrame({"transformed_feature": names[: len(importance)], "mean_abs_shap": importance})
        table["feature"] = table["transformed_feature"].map(raw_feature_from_transformed)
        table["display"] = table["transformed_feature"].map(feature_display_name)
        table = table.groupby(["feature", "display"], as_index=False)["mean_abs_shap"].sum()
        table.sort_values("mean_abs_shap", ascending=False, inplace=True)
        table.to_csv(model_dir / "shap_importance.csv", index=False)
        print(f"  [SHAP] global importance saved ({len(table)} raw features)")
        return table
    except Exception as exc:  # SHAP is explainability, never a hard dependency.
        print(f"  [SHAP] skipped: {exc}")
        return None


def _fit_kmeans(df: pd.DataFrame, model_dir: Path, seed: int) -> dict[str, Any]:
    available = [column for column in CLUSTER_FEATURES if column in df.columns]
    cluster_df = df[available].apply(pd.to_numeric, errors="coerce").fillna(df[available].median(numeric_only=True))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df)
    n_clusters = min(5, max(2, len(df) // 500))
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    labels = model.fit_predict(X_scaled)
    profiles = (
        df.assign(cluster=labels)
        .groupby("cluster")[available + ["success", "roi_percent"]]
        .mean()
        .round(3)
    )
    quality = (
        profiles["market_size_score"]
        + profiles["market_growth_score"]
        + profiles["audience_fit"]
        + profiles["differentiation"]
        + profiles["team_readiness"]
        - profiles["competitor_intensity"]
    ).sort_values(ascending=False)
    names = [
        "High-potential growth play",
        "Strong differentiated niche",
        "Balanced opportunity",
        "Competitive execution play",
        "High-risk early concept",
    ]
    descriptions: dict[str, str] = {}
    for rank, cluster_id in enumerate(quality.index):
        label = names[min(rank, len(names) - 1)]
        descriptions[str(int(cluster_id))] = label
        profiles.loc[cluster_id, "segment_name"] = label
    joblib.dump(
        {
            "scaler": scaler,
            "model": model,
            "features": available,
            "profiles": profiles.reset_index().to_dict(orient="records"),
            "descriptions": descriptions,
        },
        model_dir / "kmeans.joblib",
    )
    print(f"  [KMeans] {n_clusters} market segments saved")
    return {
        "n_clusters": n_clusters,
        "features": available,
        "profiles": json_safe(profiles.reset_index().to_dict(orient="records")),
        "descriptions": descriptions,
    }


def run_training(
    dataset_path: str | Path | None = None,
    *,
    quick: bool = False,
    seed: int = 42,
    rows: int = 6000,
) -> dict[str, Any]:
    ensure_directories()
    paths = get_paths()
    model_dir = paths.models
    model_dir.mkdir(parents=True, exist_ok=True)

    if dataset_path is None and not (paths.data / "synthetic_businesses.csv").exists():
        print("[DATA] Synthetic dataset not found, generating it…")
        save_dataset(generate_synthetic_dataset(rows, seed), seed=seed)
    df = load_dataset(dataset_path, require_targets=True)
    print(f"[DATA] {len(df):,} rows loaded from {dataset_path or 'data/synthetic_businesses.csv'}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[CLASSIFICATION_TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )
    print(f"[SPLIT] train={len(X_train):,} test={len(X_test):,} positive_rate={y.mean():.1%}")

    print("\n[CLASSIFICATION] Baseline cross-validation")
    comparison_cls, fitted_cls = _evaluate_classifier_cv(X_train, y_train, seed)
    best_name = str(comparison_cls.sort_values("roc_auc_mean", ascending=False).iloc[0]["model"])
    print(f"[CLASSIFICATION] Best baseline: {best_name}")
    best_baseline = fitted_cls[best_name]

    print("[CLASSIFICATION] Hyperparameter tuning")
    tuned_cls, tuning_cls = _tune(
        best_baseline,
        X_train,
        y_train,
        seed=seed,
        n_iter=6 if quick else 12,
        cv_splits=2 if quick else 3,
        kind="classification",
    )
    tuned_name = f"Tuned {best_name}"
    tuned_cv = _cross_validate_pipeline(
        tuned_cls, X_train, y_train, kind="classification", seed=seed
    )
    tuned_row = pd.DataFrame(
        [
            {
                "model": tuned_name,
                "kind": "classification",
                "fit_seconds": tuning_cls.get("seconds", 0.0),
                **tuned_cv,
            }
        ]
    )
    comparison_cls = pd.concat([comparison_cls, tuned_row], ignore_index=True)
    comparison_cls["kind"] = "classification"

    tuned_cls.fit(X_train, y_train)
    proba = tuned_cls.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    cls_metrics = _metrics_classification(y_test, proba, pred)
    print(
        f"[CLASSIFICATION] Test: accuracy={cls_metrics['accuracy']:.4f} "
        f"F1={cls_metrics['f1']:.4f} ROC-AUC={cls_metrics['roc_auc']:.4f}"
    )

    fpr, tpr, _ = roc_curve(y_test, proba)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(model_dir / "roc_curve.csv", index=False)
    pd.DataFrame(cls_metrics["confusion_matrix"]).to_csv(model_dir / "confusion_matrix.csv", index=False)
    pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "y_pred": pred,
            "proba_success": proba,
        }
    ).to_csv(model_dir / "test_predictions_classification.csv", index=False)

    joblib.dump(tuned_cls, model_dir / "classifier.joblib")
    permutation_cls = _permutation_importance_table(
        tuned_cls, X_test, y_test, seed=seed, scoring="roc_auc"
    )
    permutation_cls.to_csv(model_dir / "feature_importance_classification.csv", index=False)
    shap_cls = _shap_analysis(tuned_cls, X_train, model_dir, seed=seed)

    print("\n[REGRESSION] ROI and break-even")
    regression_results: dict[str, Any] = {}
    all_comparison: list[pd.DataFrame] = [comparison_cls]
    for target in REGRESSION_TARGETS:
        print(f"\n  Target: {target}")
        y_reg = df[target].astype(float)
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(
            X, y_reg, test_size=0.20, random_state=seed
        )
        comparison_reg, fitted_reg = _evaluate_regressor_cv(Xr_train, yr_train, seed)
        best_reg_name = str(comparison_reg.sort_values("mae_mean").iloc[0]["model"])
        print(f"  Best baseline: {best_reg_name}")
        tuned_reg, tuning_reg = _tune(
            fitted_reg[best_reg_name],
            Xr_train,
            yr_train,
            seed=seed,
            n_iter=4 if quick else 9,
            cv_splits=2 if quick else 3,
            kind="regression",
        )
        tuned_reg_name = f"Tuned {best_reg_name}"
        tuned_reg_cv = _cross_validate_pipeline(
            tuned_reg, Xr_train, yr_train, kind="regression", seed=seed
        )
        reg_row = pd.DataFrame(
            [
                {
                    "model": tuned_reg_name,
                    "kind": "regression",
                    "fit_seconds": tuning_reg.get("seconds", 0.0),
                    **tuned_reg_cv,
                }
            ]
        )
        comparison_reg = pd.concat([comparison_reg, reg_row], ignore_index=True)
        comparison_reg["kind"] = "regression"
        comparison_reg["target"] = target
        all_comparison.append(comparison_reg)

        tuned_reg.fit(Xr_train, yr_train)
        reg_pred = tuned_reg.predict(Xr_test)
        if target == "roi_percent":
            reg_pred = np.clip(reg_pred, -100, 700)
        else:
            reg_pred = np.clip(reg_pred, 1, 60)
        reg_metrics = _metrics_regression(yr_test, reg_pred)
        print(
            f"  Test {target}: MAE={reg_metrics['mae']:.2f} "
            f"RMSE={reg_metrics['rmse']:.2f} R2={reg_metrics['r2']:.4f}"
        )
        suffix = "roi" if target == "roi_percent" else "breakeven"
        joblib.dump(tuned_reg, model_dir / f"regression_{suffix}.joblib")
        importance_reg = _permutation_importance_table(
            tuned_reg, Xr_test, yr_test, seed=seed, scoring="neg_mean_absolute_error"
        )
        importance_reg.to_csv(model_dir / f"feature_importance_{suffix}.csv", index=False)
        pd.DataFrame(
            {
                "y_true": yr_test.to_numpy(),
                "y_pred": reg_pred,
            }
        ).to_csv(model_dir / f"test_predictions_{suffix}.csv", index=False)
        regression_results[target] = {
            "best_model": tuned_reg_name,
            "metrics": reg_metrics,
            "tuning": tuning_reg,
            "comparison": json_safe(comparison_reg.to_dict(orient="records")),
        }

    print("\n[KMEANS] Market segmentation")
    kmeans_info = _fit_kmeans(df, model_dir, seed)

    comparison = pd.concat(all_comparison, ignore_index=True)
    comparison.to_csv(model_dir / "model_comparison.csv", index=False)

    metadata = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "version": "1.0.0",
        "random_state": seed,
        "dataset": {
            "path": str(dataset_path or "data/synthetic_businesses.csv"),
            "rows": int(len(df)),
            "synthetic": dataset_path is None,
            "success_rate": float(df[CLASSIFICATION_TARGET].mean()),
            "columns": list(df.columns),
        },
        "features": {
            "all": FEATURE_COLUMNS,
            "categorical": CATEGORICAL_FEATURES,
            "numeric": NUMERIC_FEATURES,
        },
        "classification": {
            "target": CLASSIFICATION_TARGET,
            "best_model": tuned_name,
            "cross_validation": json_safe(
                comparison_cls.sort_values("roc_auc_mean", ascending=False).to_dict(orient="records")
            ),
            "test_metrics": cls_metrics,
            "tuning": tuning_cls,
            "permutation_importance": json_safe(permutation_cls.to_dict(orient="records")),
            "shap_available": shap_cls is not None,
        },
        "regression": {target: details for target, details in regression_results.items()},
        "clustering": kmeans_info,
        "test_size": 0.20,
    }
    with (model_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(metadata), handle, ensure_ascii=False, indent=2)
    with (model_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            json_safe(
                {
                    "dataset": metadata["dataset"],
                    "classification": cls_metrics,
                    "regression": {k: v["metrics"] for k, v in regression_results.items()},
                    "model_comparison": comparison.to_dict(orient="records"),
                    "best_models": {
                        "classification": tuned_name,
                        **{k: v["best_model"] for k, v in regression_results.items()},
                    },
                }
            ),
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("\n[DONE] Artifacts written to", model_dir)
    print(f"  Best classifier: {tuned_name} | ROC-AUC={cls_metrics['roc_auc']:.3f}")
    for target, details in regression_results.items():
        print(
            f"  {target}: {details['best_model']} | "
            f"MAE={details['metrics']['mae']:.2f} | R2={details['metrics']['r2']:.3f}"
        )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all CEO AI models.")
    parser.add_argument("--dataset", type=Path, default=None, help="Optional CSV/XLSX with the same schema.")
    parser.add_argument("--quick", action="store_true", help="Fewer tuning candidates for a fast smoke run.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--rows", type=int, default=6000, help="Rows to generate if data is missing.")
    args = parser.parse_args()
    run_training(args.dataset, quick=args.quick, seed=args.seed, rows=args.rows)


if __name__ == "__main__":
    main()
