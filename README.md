<div align="center">

*Read this in other languages: [Tiếng Việt](README-vi.md)*

# Data Fitting with Ordinary Least Squares (OLS)

[![Python version](https://img.shields.io/badge/Python->=3.8-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

An end-to-end regression project built from scratch, featuring mathematical rigor, data pipelines, and advanced statistical analysis.

[Project Overview](#project-overview) • [Features](#features) • [Getting started](#getting-started) • [Repository Structure](#repository-structure)

</div>

## Project Information

- **Course**: MTH00051 - Toán Ứng Dụng và Thống Kê (Applied Mathematics and Statistics)
- **Institution**: Ho Chi Minh City University of Science (HCMUS)
- **Authors**:
  - Nguyễn Anh Thái
  - Nguyễn Đình Tuấn
  - Nguyễn Huỳnh Gia Bảo
  - Vòng Sau Hậu
  - Lương Nhật Tân

## Project Overview

This project provides a comprehensive implementation of Ordinary Least Squares (OLS) regression entirely from scratch, utilizing core linear algebra principles. Designed for Applied Mathematics and Statistics, the repository is split into two primary components: a mathematically rigorous foundational implementation, and a practical application predicting IMDB movie scores through an advanced data preprocessing pipeline.

> [!NOTE]  
> This project focuses on understanding the inner workings of regression models by minimizing the reliance on high-level machine learning libraries for core calculations.

## Features

- **Algorithm Implementations from Scratch**: Core OLS, Ridge, and Lasso regressions built using raw matrix operations.
- **Statistical Inference**: Calculates standard errors, t-statistics, p-values, and 95% confidence intervals directly from the Hat matrix.
- **Advanced Diagnostics**: Extensive residual analysis, cross-validation, and Monte Carlo simulations to validate estimator properties and model robustness.
- **Robust Data Pipeline**: An automated preprocessing class handling KNN imputation, Winsorization, Box-Cox transformations, and categorical encoding without data leakage.
- **Model Comparison**: Benchmarks custom implementations against standard libraries like `scikit-learn` and `statsmodels`.

## Technology Stack

- **Core Data Processing**: Python (NumPy, pandas, SciPy)
- **Machine Learning & Statistics**: scikit-learn, statsmodels
- **Data Visualization**: Matplotlib, Seaborn
- **Development Environment**: Jupyter Notebook

## Dataset

This project utilizes the **IMDB 5000 Movie Dataset** for its practical application in Part 2. You can download the dataset from Kaggle to run the pipeline yourself:

- [IMDB 5000 Movie Dataset on Kaggle](https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset)

## Getting started

You need to have Python installed on your local machine to run this project.

### Use your local environment

1. **Clone the repository** to your local machine (if not already done).

2. **Install the required dependencies** using the provided `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

3. **Explore the notebooks**:
   - Navigate to `part1/` and open `part1_notebook.ipynb` to explore the from-scratch implementations, mathematical proofs, and Monte Carlo simulations.
   - Navigate to `part2/` and open `part2_notebook.ipynb` to review the end-to-end data pipeline applied to the IMDB movie dataset.

> [!TIP]  
> Use JupyterLab or Visual Studio Code with the Python extension to smoothly interact with the `.ipynb` notebooks.

## Repository Structure

The codebase strictly follows the required project structure:

- `part1/`: Contains the core mathematical implementations (`ols.py`, `metrics.py`, `ridge_lasso.py`, etc.) and theoretical validations.
- `part2/`: Features the robust `DataPipeline` class (`data_pipeline.py`) and real-world application scripts (`model_comparison.py`, `summary_utils.py`).
- `report/`: Houses the final comprehensive project report (`report.pdf`) detailing methodologies, mathematical proofs, and experimental results.
