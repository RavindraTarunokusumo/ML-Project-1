# main.py
# Main script to run model training, evaluation, and optimization
# Make sure device have enough memory and compute power for the tasks
# Or adjust PARAM_GRID for lighter runs

import argparse

from tqdm import tqdm

from src.data import DataLoader
from src.eval import evaluate_holdout, run_grid_search
from src.model import build_model_pipeline
from src.preprocessing import SplitConfig, split_data, split_features_target

PARAM_GRID = {
    "elasticnet": {
        "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "model__l1_ratio": [0.1, 0.5, 0.9],
        "model__max_iter": [1000, 2000, 5000, 10000],
    },
    "random_forest": {
        "model__n_estimators": [100, 200, 300, 400, 500],
        "model__max_depth": [None, 10, 20, 30, 40],
        "model__min_samples_split": [2, 5, 10, 20],
    },
    "xgboost": {
        "model__n_estimators": [100, 300, 500, 1000],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        "model__max_depth": [3, 6, 9, 12],
        "model__subsample": [0.2, 0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.2, 0.6, 0.8, 1.0],
    },
}


def _prefix_param_grid(grid: dict, prefix: str) -> dict:
    """Prefix all keys in a param grid dict (for TransformedTargetRegressor)."""
    return {f"{prefix}{k}": v for k, v in grid.items()}


def _print_scores(label, scores):
    """Print a scores dict with consistent RMSE / MAE / R^2 formatting."""
    print(f"\n{'=' * 50}")
    print(label)
    for name, s in scores.items():
        print(
            f"  {name}: "
            f"RMSE={s['rmse']:.2f}  "
            f"MAE={s['mae']:.2f}  "
            f"R^2={s['r2']:.4f}"
        )
    print("=" * 50 + "\n")


def main(args) -> None:
    # Configuration
    dataset = args.dataset  # OpenML dataset name
    target_col = args.target_col
    split_cfg = SplitConfig(
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # 1. model and data loading
    model_names = ["elasticnet", "random_forest", "xgboost"]
    df = DataLoader.load_data_from_openml(dataset_name=dataset)

    # 2. data splitting
    X, y = split_features_target(df, target_col=target_col)
    train, val, test = split_data(X, y, cfg=split_cfg)

    ### ---------- Basic Run (w/o GridSearch) ---------- ###

    scores_basic = {}

    for model_name in tqdm(model_names, desc="Training models"):
        pipeline = build_model_pipeline(model_name, train[0])
        pipeline.fit(train[0], train[1])

        # Evaluate on validation set (holdout, no re-fitting)
        if args.val_size > 0:
            val_scores = evaluate_holdout(
                pipeline, val[0], val[1]
            )
            print(
                f"{model_name} val RMSE: {val_scores['rmse']:.2f}"
            )

        # Final test evaluation
        scores_basic[model_name] = evaluate_holdout(
            pipeline, test[0], test[1]
        )

    _print_scores("Final Evaluation Scores:", scores_basic)

    ### ---------- Optimized Run (w/ GridSearch) ---------- ###

    if args.optimize:
        scores_optim = {}

        for model_name in tqdm(
            model_names, desc="Optimizing models"
        ):
            estimator = build_model_pipeline(model_name, train[0])
            param_grid = PARAM_GRID.get(model_name)
            grid_search = run_grid_search(
                estimator,
                train[0],
                train[1],
                param_grid=param_grid,
                cv=5,
            )

            best = grid_search.best_estimator_

            # Evaluate on validation set (holdout)
            if args.val_size > 0:
                val_scores = evaluate_holdout(
                    best, val[0], val[1]
                )
                print(
                    f"{model_name} val RMSE: "
                    f"{val_scores['rmse']:.2f}"
                )

            scores_optim[model_name] = evaluate_holdout(
                best, test[0], test[1]
            )

        _print_scores(
            "Final Evaluation Scores (Optimized):", scores_optim
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
    args = args.parse_args()
    main(args)
