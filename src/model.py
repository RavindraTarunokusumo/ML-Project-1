from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessing import InformativeMissingFiller, build_preprocessor


def _assemble_pipeline(
    preprocessor,
    model,
    *,
    fill_informative_missing: bool = False,
):
    """Build the full pipeline with optional pre-processing steps."""
    steps: list[tuple[str, object]] = []
    if fill_informative_missing:
        steps.append(("fill_missing", InformativeMissingFiller()))
    steps.append(("preprocess", preprocessor))
    steps.append(("model", model))
    return Pipeline(steps=steps)


def build_elasticnet_pipeline(
    X: pd.DataFrame,
    *,
    fill_informative_missing: bool = False,
):
    from sklearn.linear_model import ElasticNet

    preprocessor = build_preprocessor(
        X, scale_numeric=True, use_pca=True
    )
    model = ElasticNet(random_state=42)
    return _assemble_pipeline(
        preprocessor,
        model,
        fill_informative_missing=fill_informative_missing,
    )


def build_random_forest_pipeline(
    X: pd.DataFrame,
    *,
    fill_informative_missing: bool = False,
):
    from sklearn.ensemble import RandomForestRegressor

    preprocessor = build_preprocessor(
        X, scale_numeric=False, use_pca=False
    )
    model = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    )
    return _assemble_pipeline(
        preprocessor,
        model,
        fill_informative_missing=fill_informative_missing,
    )


def build_xgboost_pipeline(
    X: pd.DataFrame,
    *,
    fill_informative_missing: bool = False,
):
    try:
        from xgboost import XGBRegressor
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "XGBoost is not installed. "
            "Install with `pip install xgboost`."
        ) from exc

    preprocessor = build_preprocessor(
        X, scale_numeric=False, use_pca=False
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
    )


def build_model_pipeline(
    model_name: str,
    X: pd.DataFrame,
    *,
    fill_informative_missing: bool = False,
):
    kwargs = {
        "fill_informative_missing": fill_informative_missing,
    }
    name = model_name.lower()
    if name in {"elasticnet", "enet", "elastic_net"}:
        return build_elasticnet_pipeline(X, **kwargs)
    if name in {"randomforest", "random_forest", "rf"}:
        return build_random_forest_pipeline(X, **kwargs)
    if name in {"xgboost", "xgb"}:
        return build_xgboost_pipeline(X, **kwargs)
    raise ValueError(f"Unknown model name: {model_name}")
