import warnings
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from src.model import build_model_pipeline


def validate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> float:
    """Perform cross-validation and return the average RMSE."""
    try:
        neg_mse_scores = cross_val_score(
            model, X, y, scoring="neg_mean_squared_error", cv=cv, n_jobs=-1
        )
        # return average RMSE
        rmse_scores = (-neg_mse_scores) ** 0.5
        avg_rmse = rmse_scores.mean()
        return avg_rmse
    except Exception as e:
        raise ValueError(f"Error during model validation: {e}")

def run_grid_search(
    model_name: str,    
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: dict | None = None,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
):
    try:
        pipeline = build_model_pipeline(model_name, X)
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        
        # Suppress warnings during grid search
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid.fit(X, y)
        
        return grid
    except Exception as e:
        raise ValueError(f"Error during GridSearchCV for {model_name}: {e}")
        
    
