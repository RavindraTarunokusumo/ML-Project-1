from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    FeatureEngineer,
    InformativeMissingFiller,
    build_preprocessor,
)


def _assemble_pipeline(
    preprocessor,
    model,
    *,
    fill_informative_missing: bool = False,
    feature_engineering: bool = False,
):
    """Build the full pipeline with optional pre-processing steps."""
    steps: list[tuple[str, object]] = []
    if fill_informative_missing:
        steps.append(("fill_missing", InformativeMissingFiller()))
    if feature_engineering:
        steps.append(("engineer", FeatureEngineer()))
    steps.append(("preprocess", preprocessor))
    steps.append(("model", model))
    return Pipeline(steps=steps)


def build_elasticnet_pipeline(
    X: pd.DataFrame,
    *,
    fill_informative_missing: bool = False,
    use_ordinal_encoding: bool = False,
    feature_engineering: bool = False,
    correct_skewness: bool = False,
):
    from sklearn.linear_model import ElasticNet

    preprocessor = build_preprocessor(
        X,
        scale_numeric=True,
        use_pca=True,
        pca_n_components=0.95,
        use_ordinal_encoding=use_ordinal_encoding,
        correct_skewness=correct_skewness,
    )
    model = ElasticNet(random_state=42)
    return _assemble_pipeline(
        preprocessor,
        model,
        fill_informative_missing=fill_informative_missing,
        feature_engineering=feature_engineering,
    )


def build_random_forest_pipeline(
    X: pd.DataFrame,
    *,
    fill_informative_missing: bool = False,
    use_ordinal_encoding: bool = False,
    feature_engineering: bool = False,
    correct_skewness: bool = False,
):
    from sklearn.ensemble import RandomForestRegressor

    preprocessor = build_preprocessor(
        X,
        scale_numeric=False,
        use_pca=False,
        use_ordinal_encoding=use_ordinal_encoding,
        correct_skewness=correct_skewness,
    )
    model = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    )
    return _assemble_pipeline(
        preprocessor,
        model,
        fill_informative_missing=fill_informative_missing,
        feature_engineering=feature_engineering,
    )


def build_xgboost_pipeline(
    X: pd.DataFrame,
    *,
    fill_informative_missing: bool = False,
    use_ordinal_encoding: bool = False,
    feature_engineering: bool = False,
    correct_skewness: bool = False,
):
    try:
        from xgboost import XGBRegressor
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "XGBoost is not installed. "
            "Install with `pip install xgboost`."
        ) from exc

    preprocessor = build_preprocessor(
        X,
        scale_numeric=False,
        use_ordinal_encoding=use_ordinal_encoding,
        correct_skewness=correct_skewness,
    )
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    return _assemble_pipeline(
        preprocessor,
        model,
        fill_informative_missing=fill_informative_missing,
        feature_engineering=feature_engineering,
    )


def build_model_pipeline(
    model_name: str,
    X: pd.DataFrame,
    *,
    fill_informative_missing: bool = False,
    use_ordinal_encoding: bool = False,
    feature_engineering: bool = False,
    correct_skewness: bool = False,
    log_target: bool = False,
):
    kwargs = {
        "fill_informative_missing": fill_informative_missing,
        "use_ordinal_encoding": use_ordinal_encoding,
        "feature_engineering": feature_engineering,
        "correct_skewness": correct_skewness,
    }
    name = model_name.lower()
    if name in {"elasticnet", "enet", "elastic_net"}:
        pipeline = build_elasticnet_pipeline(X, **kwargs)
    elif name in {"randomforest", "random_forest", "rf"}:
        pipeline = build_random_forest_pipeline(X, **kwargs)
    elif name in {"xgboost", "xgb"}:
        pipeline = build_xgboost_pipeline(X, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if log_target:
        return TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
        )
    return pipeline
