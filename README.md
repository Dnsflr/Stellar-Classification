# Stellar Classification with RandomForest and SVM

## Project Overview

This project focuses on classifying stellar objects (stars, galaxies, and quasars) based on astronomical features extracted from observational data. Utilizing machine learning techniques, specifically RandomForest and Support Vector Machines (SVM), this project aims to build robust models for accurate stellar classification.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Methodology](#methodology)
  - [Data Loading and Initial Exploration](#data-loading-and-initial-exploration)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training and Evaluation](#model-training-and-evaluation)
  - [Feature Importance](#feature-importance)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The project uses a dataset containing various features of celestial objects. The target variable is `class`, which categorizes objects into 'GALAXY', 'QSO' (Quasar), or 'STAR'.

## Features

The dataset includes the following features, which are crucial for classification:

**Categorical Features:**
* `class`: The type of stellar object (GALAXY, QSO, STAR). This is the target variable.

**Numerical Features:**
* `alpha`: Right Ascension angle (J2000)
* `delta`: Declination angle (J2000)
* `u`, `g`, `r`, `i`, `z`: Photometric magnitudes in different filters (ultraviolet, green, red, infrared, and another infrared band respectively). These represent the brightness of the object in various wavelengths.
* `redshift`: A measure of how much the light from the object has been stretched due to the expansion of the universe.
* `run_id`: Run number used to identify the specific scan.
* `field_id`: Field number used to identify the specific field.
* `spec_obj_id`: Unique ID used for optical spectroscopic surveys.
* `mjd`: Modified Julian Date, used to specify the time of observation.
* `fiber_id`: Optical fiber ID that took the spectroscopic measurement.

## Methodology

The project follows a standard machine learning pipeline:

### Data Loading and Initial Exploration

The dataset is loaded using pandas, and initial checks are performed to understand its structure, identify missing values, and analyze the distribution of features.

### Data Preprocessing

* **Handling Categorical Features:** The `class` variable, being the target, is used directly. Other categorical features are handled as part of the pipeline.
* **Feature Scaling:** Numerical features are scaled using `StandardScaler` to ensure that features with larger values do not dominate the learning process.
* **Column Transformer:** A `ColumnTransformer` is used to apply different preprocessing steps to numerical and categorical features within a single pipeline.

### Model Training and Evaluation

Two powerful classification algorithms are employed:

1.  **RandomForestClassifier:** An ensemble learning method that builds multiple decision trees and merges their predictions.
    * **Hyperparameter Tuning:** `GridSearchCV` with `StratifiedKFold` cross-validation is used to find the optimal hyperparameters for the RandomForest model, including `n_estimators`, `max_depth`, and `min_samples_split`.

2.  **LinearSVC (Support Vector Machine):** A linear classifier that finds the optimal hyperplane to separate classes.
    * **Hyperparameter Tuning:** `GridSearchCV` is used to tune the `C` parameter (regularization parameter) for the LinearSVC model.

Both models are evaluated based on their accuracy and confusion matrices to understand their performance in classifying different stellar object types.

### Feature Importance

For the best-performing SVM model (after `GridSearchCV`), the coefficients are analyzed to determine the most important features in predicting the type of stellar object. This provides insights into which astronomical measurements are most indicative of an object's classification.

## Results

The project will present:

* The performance metrics (e.g., accuracy) of the optimized RandomForest and SVM models.
* Confusion matrices for both models, illustrating their classification strengths and weaknesses across different stellar classes.
* A visualization of the most important features as determined by the SVM model, highlighting which observational parameters contribute most significantly to the classification.

## Requirements

To run this notebook, you'll need the following Python libraries:

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `scipy`

You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
