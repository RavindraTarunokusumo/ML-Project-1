# TODO

## 1) Improve preprocessing
### Requirements:
- Keep preprocessing in `src/preprocessing.py` and avoid data leakage.
- Preserve compatibility with ElasticNet, RandomForest, and XGBoost pipelines.
S### ubtasks:
- Add optional log transforms for skewed numeric features and `SalePrice`.
- Add feature engineering helpers (e.g., total bathrooms, house age).
- Add configurable imputation strategies (median vs. KNN).
### Notes:
- Prefer opt-in flags for transformations to avoid changing baseline results.

## 2) Logging and data persistence (JSON)
### Requirements:
- Store experiment metadata and metrics in JSON.
- Keep files under `data/` or a new `artifacts/` folder.
### Subtasks:
- Define a lightweight run schema (model name, params, scores, timestamp).
- Save `GridSearchCV` best params and CV scores.
- Add utility to load and summarize past runs.
### Notes:
- Avoid storing large arrays unless necessary; summarize where possible.

## 3) Train-Val plotting
### Requirements:
- Plot learning curves or metric-by-epoch/iteration when available.
### Subtasks:
- Add a helper to plot training vs. validation scores.
- Support both CV folds and holdout splits.
### Notes:
- For non-iterative models, approximate using increasing training sizes.

## 4) Eval plotting
### Requirements:
- Add visualization beyond ASCII curve in `src/eval.py`.
### Subtasks:
- Add residual plots and predicted vs. actual scatter plots.
- Add feature importance plots for tree-based models.
### Notes:
- Keep plotting optional to avoid forcing matplotlib dependency in core code.

## 5) Demo notebook
### Requirements:
- Provide an end-to-end notebook that runs the full pipeline.
### Subtasks:
- Add a notebook that loads data, runs CV, grid search, and evaluation.
- Include a short summary of results and best model.
### Notes:
- Keep runtime under ~10 minutes on a laptop.

## 6) Neural Nets (optional 02_exploration.ipynb)
### Requirements: 
- Only if time allows; treat as exploratory.
### Subtasks:
- Add a baseline MLPRegressor or Keras model (optional dependency).
- Compare against tree/linear baselines.
### Notes:
- Clearly mark optional dependencies and steps.
