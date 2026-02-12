# main.py
# Main script to run model training, evaluation, and optimization
# Make sure device have enough memory and compute power for the tasks
# Or adjust PARAM_GRID for lighter runs

import argparse

import pandas as pd
from tqdm import tqdm

from src.data import DataLoader
from src.eval import (
    evaluate_cv,
    evaluate_holdout,
    print_scores,
    run_grid_search,
    run_randomized_search,
)
from src.model import build_model_pipeline, save_model
from src.preprocessing import SplitConfig, split_data, split_features_target

# Hyperparameter grids for GridSearchCV / RandomizedSearchCV
PARAM_GRID = {
    "elasticnet": {
        "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "model__l1_ratio": [0.1, 0.5, 0.9],
        "model__max_iter": [1000, 2000, 5000, 10000],
    },
    "random_forest": {
        "model__n_estimators": [100, 200, 300, 400, 500],
        "model__max_depth": [1, 3, 5, 7],
        "model__min_samples_split": [2, 5, 10, 20],
    },
    "xgboost": {
        "model__n_estimators": [100, 300, 500, 700],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.3],
        "model__max_depth": [1, 3, 5, 7],
        "model__subsample": [0.2, 0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.2, 0.6, 0.8, 1.0],
    },
    "catboost": {
        "model__iterations": [100, 300, 500, 700],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.3],
        "model__depth": [1, 3, 5, 7],
        "model__l2_leaf_reg": [1, 3, 5, 7],
    },
}


def _prefix_param_grid(grid: dict, prefix: str) -> dict:
    """Prefix all keys in a param grid dict (for TransformedTargetRegressor)."""
    return {f"{prefix}{k}": v for k, v in grid.items()}

def main(args) -> None:
    # Configuration
    dataset = args.dataset  # OpenML dataset name
    target_col = args.target_col
    split_cfg = SplitConfig(
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=True, # Shuffle before splitting for better generalization
    )

    # Pipeline construction kwargs (from CLI flags)
    pipeline_kwargs = {
        "fill_informative_missing": args.fill_missing,
        "use_ordinal_encoding": args.ordinal,
        "feature_engineering": args.engineer,
        "correct_skewness": args.correct_skew,
        "log_target": args.log_target,
    }

    # 1. model and data loading
    model_names = [
        "elasticnet", "random_forest", "xgboost", "catboost",
    ]
    df = DataLoader.load_data_from_openml(dataset_name=dataset)

    # 2. data splitting
    X, y = split_features_target(df, target_col=target_col)
    train, val, test = split_data(X, y, cfg=split_cfg)

    # Prepare training data (combine train + val if CV is used)
    if args.cv > 1:
        X_train_cv = pd.concat([train[0], val[0]])
        y_train_cv = pd.concat([train[1], val[1]])
        print(f"Using {args.cv}-fold CV on {len(X_train_cv)} samples.")
    else:
        X_train_cv, y_train_cv = train[0], train[1]
        print(f"Using holdout validation on {len(val[0])} samples.")

    ### ---------- Basic Run (w/o GridSearch) ---------- ###

    scores_val = {}
    scores_eval = {}

    for model_name in tqdm(model_names, desc="Training models"):
        pipeline = build_model_pipeline(
            model_name, X_train_cv, **pipeline_kwargs
        )
        
        if args.cv > 1:
            # Cross-validation mode
            model_scores = evaluate_cv(
                pipeline, X_train_cv, y_train_cv, cv=args.cv
            )
            # Still fit once on full training data for final test evaluation
            pipeline.fit(X_train_cv, y_train_cv)
        else:
            # Holdout mode
            pipeline.fit(train[0], train[1])
            model_scores = evaluate_holdout(pipeline, val[0], val[1])
            scores_val[model_name] = model_scores

        print(f"{model_name} val/CV RMSE: {model_scores['rmse']:.2f}")

        # Final test evaluation
        scores_eval[model_name] = evaluate_holdout(
            pipeline, test[0], test[1]
        )
        
        # Save model
        save_path = save_model(pipeline, model_name)
        print(f"Saved {model_name} to {save_path}")

    if scores_val:
        print_scores("Validation Scores:", scores_val)
    print_scores("Final Evaluation Scores:", scores_eval)

    ### ---------- Optimized Run (w/ GridSearch) ---------- ###

    if args.optimize:
        scores_val = {}
        scores_eval = {}

        for model_name in tqdm(
            model_names, desc="Optimizing models"
        ):
            estimator = build_model_pipeline(
                model_name, X_train_cv, **pipeline_kwargs
            )
            param_grid = PARAM_GRID.get(model_name)
            if param_grid and args.log_target:
                param_grid = _prefix_param_grid(
                    param_grid, "regressor__"
                )

            if model_name in {"xgboost", "catboost"}:
                search = run_randomized_search(
                    estimator,
                    X_train_cv,
                    y_train_cv,
                    param_distributions=param_grid,
                    n_iter=150,
                    cv=args.cv if args.cv > 1 else 5,
                )
            else:
                search = run_grid_search(
                    estimator,
                    X_train_cv,
                    y_train_cv,
                    param_grid=param_grid,
                    cv=args.cv if args.cv > 1 else 5,
                )

            best = search.best_estimator_

            # Evaluate on validation set (holdout or CV)
            if args.cv > 1:
                val_scores = evaluate_cv(
                    best, X_train_cv, y_train_cv, cv=args.cv
                )
            else:
                val_scores = evaluate_holdout(best, val[0], val[1])
                scores_val[model_name] = val_scores
            
            print(f"{model_name} val/CV RMSE: {val_scores['rmse']:.2f}")

            scores_eval[model_name] = evaluate_holdout(
                best, test[0], test[1]
            )
            
            # Save optimized model
            save_path = save_model(best, f"{model_name}_optimized")
            print(f"Saved optimized {model_name} to {save_path}")

        if scores_val:
            print_scores(
                "Validation Scores (Optimized):", scores_val
                )
        print_scores(
            "Final Evaluation Scores (Optimized):", scores_eval
        )


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Run model training and evaluation"
    )
    args.add_argument(
        "--dataset",
        type=str,
        default="house_prices",
        help="OpenML dataset name",
    )
    args.add_argument(
        "--target_col",
        type=str,
        default="SalePrice",
        help="Target column name",
    )
    args.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help="Validation set size (relative)",
    )
    args.add_argument(
        "--test_size",
        type=float,
        default=0.15,
        help="Test set size (relative)",
    )
    args.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility",
    )
    args.add_argument(
        "--optimize",
        action="store_true",
        help="Whether to run GridSearch optimization",
    )
    args.add_argument(
        "--cv",
        type=int,
        default=1,
        help="Number of folds for k-fold cross-validation (1 = holdout)",
    )
    args.add_argument(
        "--log_target",
        action="store_true",
        help="Apply log1p transform to target variable",
    )
    args.add_argument(
        "--fill_missing",
        action="store_true",
        help="Fill informative missing values (NaN = absent)",
    )
    args.add_argument(
        "--ordinal",
        action="store_true",
        help="Use ordinal encoding for quality features",
    )
    args.add_argument(
        "--engineer",
        action="store_true",
        help="Add engineered features (TotalSF, TotalBath, etc.)",
    )
    args.add_argument(
        "--correct_skew",
        action="store_true",
        help="Apply log1p to skewed numeric features",
    )
    args = args.parse_args()
    main(args)
