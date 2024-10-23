# SVM Classifier with Outlier Removal and Grid Search Optimization

## Overview

This is a personal project developed by **Daniel Northcott**, an engineering student at the University of Arizona, with minors in Computer Science and Mathematics, set to graduate in December 2025. The project demonstrates the use of **Support Vector Machines (SVM)** for classification on a user-selected dataset. It includes techniques such as outlier removal using **Isolation Forest**, data preprocessing with **StandardScaler**, and parameter optimization.

## Features

- **Support Vector Machine (SVM)**: The project uses a Radial Basis Function (RBF) kernel SVM for classification.
- **Outlier Removal**: Implements outlier detection and removal using the **Isolation Forest** algorithm, with a tunable contamination parameter.
- **Data Preprocessing**: Uses `StandardScaler` to standardize features by removing the mean and scaling to unit variance.
- **Model Optimization**: Optimizes SVM parameters (`C` and `gamma`) using Grid Search (optional in the future development).
- **Evaluation Metrics**: Outputs the **confusion matrix** and **accuracy score** for model evaluation.

## Requirements

To run this project, you'll need:

- Python 3.x
- `numpy`
- `scikit-learn`
- `matplotlib` (for future visualization extensions, if needed)

You can install the required libraries using the following:

```bash
pip install numpy scikit-learn matplotlib
