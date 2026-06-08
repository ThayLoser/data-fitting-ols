# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-08

### Added
- Comprehensive English and Vietnamese `README.md` files containing project information, technology stack, and IMDB dataset links.
- Final project report (`report.pdf`).
- Initial codebase including core OLS, Ridge, and Lasso regressions implemented from scratch.
- `DataPipeline` class for automated data preprocessing (KNN Imputation, Winsorization, Box-Cox, One-hot encoding).
- Residual analysis, cross-validation, and Monte Carlo simulations.
- Jupyter Notebooks (`part1_notebook.ipynb`, `part2_notebook.ipynb`) for experiments and model evaluation.

### Fixed
- Matrix multiplication performance and correctness in `part1/utils.py`.

### Refactored
- Removed inline execution and added missing module imports in `part2/model_comparison.py` to make it modular.
