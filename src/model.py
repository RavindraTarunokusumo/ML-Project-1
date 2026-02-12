from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    FeatureEngineer,
    InformativeMissingFiller,
    NeighborhoodTargetEncoder,
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
        steps.append(("target_encode", NeighborhoodTargetEncoder()))
    steps.append(("preprocess", preprocessor))
    steps.append(("model", model))
    return Pipeline(steps=steps)


def _get_x_for_preprocessor(
    X: pd.DataFrame,
    fill_informative_missing: bool,
    feature_engineering: bool,
) -> pd.DataFrame:
    """Apply early transformations to X so build_preprocessor sees correct columns/types."""
    X_p = X.copy()
    if fill_informative_missing:
        X_p = InformativeMissingFiller().transform(X_p)
    if feature_engineering:
        X_p = FeatureEngineer().transform(X_p)
        # Placeholder for columns added by NeighborhoodTargetEncoder
        X_p["NeighMedianPrice"] = 0.0
    return X_p


def build_elasticnet_pipeline(
    X: pd.DataFrame,
    *,
    fill_informative_missing: bool = False,
    use_ordinal_encoding: bool = False,
    feature_engineering: bool = False,
    correct_skewness: bool = False,
):
    from sklearn.linear_model import ElasticNet

    X_for_prep = _get_x_for_preprocessor(
        X, fill_informative_missing, feature_engineering
    )
    preprocessor = build_preprocessor(
        X_for_prep,
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

    X_for_prep = _get_x_for_preprocessor(
        X, fill_informative_missing, feature_engineering
    )
    preprocessor = build_preprocessor(
        X_for_prep,
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

    X_for_prep = _get_x_for_preprocessor(
        X, fill_informative_missing, feature_engineering
    )
    preprocessor = build_preprocessor(
        X_for_prep,
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


class _CatBoostRegressorWrapper(BaseEstimator, RegressorMixin):
    """Thin sklearn-compatible wrapper around CatBoostRegressor.

    Needed because CatBoost <=1.2 lacks the __sklearn_tags__
    protocol required by scikit-learn >=1.6.
    """

    def __init__(
        self,
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=0,
        **kwargs,
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.verbose = verbose
        self._extra_kwargs = kwargs

    def fit(self, X, y, **fit_params):
        from catboost import CatBoostRegressor

        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            verbose=self.verbose,
            **self._extra_kwargs,
        )
        self.model_.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def build_catboost_pipeline(
    X: pd.DataFrame,
    *,
    fill_informative_missing: bool = False,
    use_ordinal_encoding: bool = False,
    feature_engineering: bool = False,
    correct_skewness: bool = False,
):
    try:
        import catboost  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "CatBoost is not installed. "
            "Install with `pip install catboost`."
        ) from exc

    X_for_prep = _get_x_for_preprocessor(
        X, fill_informative_missing, feature_engineering
    )
    preprocessor = build_preprocessor(
        X_for_prep,
        scale_numeric=False,
        use_ordinal_encoding=use_ordinal_encoding,
        correct_skewness=correct_skewness,
    )
    model = _CatBoostRegressorWrapper(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=0,
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
    elif name in {"catboost", "cb"}:
        pipeline = build_catboost_pipeline(X, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if log_target:
        return TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
        )
    return pipeline


def save_model(
    model: object, model_name: str, base_path: str = "model"
) -> Path:
    """Save a trained model/pipeline to the specified directory."""
    dir_path = Path(base_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    file_path = dir_path / f"{model_name.lower()}.joblib"
    joblib.dump(model, file_path)
    return file_path
