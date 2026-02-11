# Project Overview

This is a machine learning project for regression tasks. It trains and evaluates three different models: ElasticNet, RandomForest, and XGBoost. The project is structured to allow for easy data loading, preprocessing, model training, and evaluation.

# Building and Running

## Dependencies

The project requires Python >=3.11. The dependencies are not yet listed in the `pyproject.toml` file.

**TODO:** Add the following dependencies to the `pyproject.toml` file:

- `pandas`
- `scikit-learn`
- `xgboost`
- `tqdm`
- `openml`

The dependencies are managed by `uv`. To install the dependencies, first add them to the `pyproject.toml` file and then run:

```bash
uv sync
```

## Running the project

The main entry point for the project is `main.py`. You can run it from the command line with the following command:

```bash
python main.py [OPTIONS]
```

### Options

- `--dataset`: The name of the OpenML dataset to use (default: `house_prices`).
- `--target_col`: The name of the target column in the dataset (default: `SalePrice`).
- `--val_size`: The proportion of the data to use for the validation set (default: `0.4`).
- `--test_size`: The proportion of the data to use for the test set (default: `0.2`).
- `--random_state`: The random state for reproducibility (default: `42`).
- `--optimize`: If set, the script will run a grid search to find the best hyperparameters for each model.

### Examples

**Run with default settings:**

```bash
python main.py
```

**Run with a different dataset and target column:**

```bash
python main.py --dataset "california_housing" --target_col "MedHouseVal"
```

**Run with hyperparameter optimization:**

```bash
python main.py --optimize
```

# Development Conventions

The project uses `ruff` for linting. The configuration is in the `pyproject.toml` file.
