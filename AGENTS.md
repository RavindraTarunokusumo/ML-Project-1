# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the project’s Python modules:
- `src/data.py` for data loading (CSV/OpenML).
- `src/preprocessing.py` for feature/target splitting and preprocessing setup.
- `src/model.py` for model initialization and pipelines.
- `src/eval.py` for cross-validation, ASCII score plotting, and grid search.
- `notebooks/` holds Jupyter notebooks for experimentation (e.g., `notebooks/01_exploration.ipynb`).
- `data/` is present for datasets and artifacts (currently empty).
- `main.py` is a simple entry point for quick sanity checks.

## Build, Test, and Development Commands
- `python main.py` runs the minimal entry point and verifies the environment.
- `python -m ruff check .` runs lint checks using the rules in `pyproject.toml`.
- `python -m ruff format .` (optional) formats code if you choose to use Ruff’s formatter.

## Coding Style & Naming Conventions
- Python 3.11+ (see `pyproject.toml`).
- Indentation: 4 spaces, no tabs.
- Line length: 79 characters (Ruff config).
- Naming: `snake_case.py` for modules, `snake_case` for functions/variables, `CapWords` for classes (PEP 8 style).
- Keep imports organized; Ruff enforces import sorting (`I` rules).

## Testing Guidelines
- No test framework or `tests/` directory exists yet.
- If tests are added, place them in `tests/` and name files `test_*.py`.
- Prefer `pytest` conventions if you introduce automated tests.

## Commit & Pull Request Guidelines
- Git history currently shows a single commit (“Initial commit”), so no established convention.
- Use short, imperative commit messages (e.g., “Add data loader”).
- PRs should include a concise summary, relevant context, and any notebook outputs or figures when applicable.

## Security & Configuration Tips
- Keep large datasets and credentials out of Git; use `data/` for local-only files.
- Avoid committing `.venv/` changes; it’s a local environment directory.
