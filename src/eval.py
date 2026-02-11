from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.pipeline import Pipeline


def print_scores(label, scores):
    """Print a scores dict with consistent RMSE / MAE / MAPE / R^2 formatting."""
    # ANSI Color Codes
    BOLD = "\033[1m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    print(f"\n{BOLD}{BLUE}{'=' * 60}{RESET}")
    print(f"{BOLD}{label}{RESET}")
    for name, s in scores.items():
        print(
            f"  {CYAN}{name.upper():<15}{RESET}: "
            f"{GREEN}RMSE={RESET}{YELLOW}{s['rmse']:>10.2f}{RESET}  "
            f"{GREEN}MAE={RESET}{YELLOW}{s['mae']:>10.2f}{RESET}  "
            f"{GREEN}MAPE={RESET}{YELLOW}{s['mape']:>8.2%}{RESET}  "
            f"{GREEN}R^2={RESET}{YELLOW}{s['r2']:>8.4f}{RESET}"
        )
    print(f"{BOLD}{BLUE}{'=' * 60}{RESET}\n")


def evaluate_holdout(
    model, X: pd.DataFrame, y: pd.Series
) -> dict[str, float]:
    """Score a pre-fitted model on a holdout set.

    Returns dict with keys: rmse, mae, mape, r2.
    """
    y_pred = model.predict(X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "mae": float(mean_absolute_error(y, y_pred)),
        "mape": float(mean_absolute_percentage_error(y, y_pred)),
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


def run_randomized_search(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    param_distributions: dict | None = None,
    n_iter: int = 100,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
    random_state: int = 42,
):
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        random_state=random_state,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y)

    return search
