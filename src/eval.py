from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline


def evaluate_holdout(
    model, X: pd.DataFrame, y: pd.Series
) -> dict[str, float]:
    """Score a pre-fitted model on a holdout set.

    Returns dict with keys: rmse, mae, r2.
    """
    y_pred = model.predict(X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "mae": float(mean_absolute_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
    }


def validate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> float:
    """Cross-validation returning average RMSE.

    Note: this re-fits the model on each CV fold.
    For evaluating a pre-fitted model, use evaluate_holdout().
    """
    neg_rmse_scores = cross_val_score(
        model,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
    )
    return float(-neg_rmse_scores.mean())


def run_grid_search(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: dict | None = None,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
):
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid.fit(X, y)

    return grid
