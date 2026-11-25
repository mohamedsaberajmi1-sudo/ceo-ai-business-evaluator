import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split

from src.data_generation import generate_synthetic_dataset
from src.preprocessing import (
    CLASSIFICATION_TARGET,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_classification_pipeline,
    build_preprocessor,
    build_regression_pipeline,
    load_dataset,
    split_xy,
)


def test_synthetic_dataset_shape_and_schema():
    df = generate_synthetic_dataset(n_rows=250, seed=11)
    assert len(df) == 250
    assert set(FEATURE_COLUMNS).issubset(df.columns)
    assert set(df[CLASSIFICATION_TARGET].unique()).issubset({0, 1})
    assert df["roi_percent"].notna().all()
    assert df["breakeven_months"].notna().all()
    assert df["sector"].nunique() > 3


def test_preprocessor_fits_and_has_no_nan():
    df = generate_synthetic_dataset(n_rows=300, seed=12)
    X = df[FEATURE_COLUMNS]
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > len(NUMERIC_FEATURES)
    assert np.isfinite(transformed).all()
    names = list(preprocessor.get_feature_names_out())
    assert any(name.startswith("num__") for name in names)
    assert any(name.startswith("cat__") for name in names)


def test_classification_and_regression_pipelines_learn():
    df = generate_synthetic_dataset(n_rows=500, seed=13)
    X, y, _ = split_xy(df, CLASSIFICATION_TARGET)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    classifier = build_classification_pipeline(LogisticRegression(max_iter=1000, random_state=0))
    classifier.fit(X_train, y_train)
    probabilities = classifier.predict_proba(X_test)[:, 1]
    assert probabilities.shape == (len(X_test),)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()

    Xr, yr, _ = split_xy(df, "roi_percent")
    Xr_train, Xr_test, yr_train, _ = train_test_split(Xr, yr, test_size=0.25, random_state=0)
    regressor = build_regression_pipeline(Ridge())
    regressor.fit(Xr_train, yr_train)
    predictions = regressor.predict(Xr_test)
    assert predictions.shape == (len(Xr_test),)
    assert np.isfinite(predictions).all()


def test_load_dataset_from_disk(tmp_path):
    df = generate_synthetic_dataset(n_rows=120, seed=14)
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    loaded = load_dataset(path, require_targets=True)
    assert isinstance(loaded, pd.DataFrame)
    assert len(loaded) == 120
    assert set(FEATURE_COLUMNS).issubset(loaded.columns)
