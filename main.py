# main.py
# Main script to run model training, evaluation, and optimization
# Make sure device have enough memory and compute power for the tasks
# Or adjust PARAM_GRID for lighter runs

import argparse
from tqdm import tqdm

from src.preprocessing import (
    split_features_target, 
    split_data, 
    SplitConfig
)
from src.data import DataLoader
from src.model import build_model_pipeline
from src.eval import (
    validate_model,
    run_grid_search
)

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


def main(args) -> None:
    # Configuration
    dataset = args.dataset  # OpenML dataset name
    target_col = args.target_col
    split_cfg = SplitConfig(val_size=args.val_size, test_size=args.test_size, random_state=args.random_state)
    
    # 1. model and data loading
    model_names = ["elasticnet", "random_forest", "xgboost"]
    df = DataLoader.load_data_from_openml(dataset_name=dataset)
    
    # 2. data splitting
    X, y = split_features_target(df, target_col=target_col)
    train, val, test = split_data(X, y, cfg=split_cfg) # df[X, y]
    
    # Tally scores
    scores_basic = {
        "validation": {},
        "evaluation": {}
    }
    
    ### ---------- Basic Run (w/o GridSearch) ---------- ###
    
    # 3.1 models pipeline
    for model_name in tqdm(model_names, desc="Training models"):
        pipeline = build_model_pipeline(model_name, train[0])
        
        # 4.1 model training
        pipeline.fit(train[0], train[1])
        
        # 5.1 model evaluation on validation set
        if args.val_size > 0:
            val_score = validate_model(pipeline, val[0], val[1], cv=5)   
            scores_basic["validation"][model_name] = val_score
            print(f"{model_name} validation RMSE: {val_score:.4f}")

        # 6.1 final test evaluation
        eval_score = pipeline.score(test[0], test[1])
        scores_basic["evaluation"][model_name] = eval_score
        
    # 8.1 final scores output
    print("\n" + "="*40)
    print("Final Evaluation Scores:")
    for model_name, score in scores_basic["evaluation"].items():
        print(f"{model_name}: R^2 = {score:.4f}")
    print("="*40 + "\n")
    
    
    ### ---------- Optimized Run (w/ GridSearch) ---------- ###
    
    # 3.2 models pipeline
    if args.optimize:
        for model_name in tqdm(model_names, desc="Optimizing models"):
            pipeline = build_model_pipeline(model_name, train[0])
            
            # 4.2 model training with GridSearchCV
            grid_search = run_grid_search(
                model_name, train[0], train[1],
                param_grid=PARAM_GRID.get(model_name, None),
                cv=5, scoring="neg_mean_squared_error"
            )
            
            # 5.2 model evaluation on validation set
            val_score = validate_model(grid_search.best_estimator_, val[0], val[1], cv=5)   
            
            # 6.2 final test evaluation
            eval_score = grid_search.best_estimator_.score(test[0], test[1])
            
            # 7.2 save scores
            scores_basic["validation"][model_name] = val_score
            scores_basic["evaluation"][model_name] = eval_score
        
        # 8.2 final scores output
        print("\n" + "="*40)
        print("Final Evaluation Scores (Optimized):")
        for model_name, score in scores_basic["evaluation"].items():
            print(f"{model_name}: R^2 = {score:.4f}")
        print("="*40 + "\n")
    
if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Run model training and evaluation")
    args.add_argument("--dataset", type=str, default="house_prices", help="OpenML dataset name")
    args.add_argument("--target_col", type=str, default="SalePrice", help="Target column name")
    args.add_argument("--val_size", type=float, default=0.4, help="Validation set size (relative)")
    args.add_argument("--test_size", type=float, default=0.2, help="Test set size (relative)")
    args.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    args.add_argument("--optimize", action="store_true", help="Whether to run GridSearch optimization")
    args = args.parse_args()
    main(args)
