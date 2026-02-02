import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor


def build_elasticnet_pipeline(X: pd.DataFrame):
    from sklearn.linear_model import ElasticNet

    preprocessor = build_preprocessor(X, scale_numeric=True)
    model = ElasticNet(random_state=42)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def build_random_forest_pipeline(X: pd.DataFrame):
    from sklearn.ensemble import RandomForestRegressor

    preprocessor = build_preprocessor(X, scale_numeric=False)
    model = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def build_xgboost_pipeline(X: pd.DataFrame):
    try:
        from xgboost import XGBRegressor
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "XGBoost is not installed. Install with `pip install xgboost`."
        ) from exc

    preprocessor = build_preprocessor(X, scale_numeric=False)
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
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def build_model_pipeline(model_name: str, X: pd.DataFrame):
    name = model_name.lower()
    if name in {"elasticnet", "enet", "elastic_net"}:
        return build_elasticnet_pipeline(X)
    if name in {"randomforest", "random_forest", "rf"}:
        return build_random_forest_pipeline(X)
    if name in {"xgboost", "xgb"}:
        return build_xgboost_pipeline(X)
    raise ValueError(f"Unknown model name: {model_name}")