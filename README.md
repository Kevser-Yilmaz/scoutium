
# Scoutium Player Performance Classification and Analysis

This project aims to classify and analyze the performance of football players using the **Scoutium** dataset. The analysis involves multiple machine learning models to predict player potential based on various attributes. Key stages include data preprocessing, feature engineering, model training and evaluation, and interpretability using SHAP (SHapley Additive exPlanations) values to gain insights into feature importance.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Modeling and Evaluation](#modeling-and-evaluation)
- [Feature Importance and Interpretability](#feature-importance-and-interpretability)
- [Author](#author)

## Project Overview

The main objectives of this project are as follows:
1. **Data Preprocessing and Cleaning**: Filter out irrelevant player positions and labels, handle categorical data, and prepare the dataset for modeling.
2. **Feature Engineering**: Generate a player-level attribute table to create a structured dataset.
3. **Modeling**: Train and evaluate multiple machine learning classifiers to predict player performance potential.
4. **Interpretability**: Use SHAP values to analyze feature importance, providing insights into the model’s decision-making process.

The project ultimately aims to provide a robust model for classifying players based on their performance potential, supporting scouts and analysts in their evaluations.

## Dataset

This analysis uses two main datasets from **Scoutium**:
- **scoutium_attributes.csv**: Contains player attributes such as agility, passing, and game intelligence.
- **scoutium_potential_labels.csv**: Includes potential labels for each player, indicating their projected performance level.

The datasets are merged on `task_response_id`, `match_id`, `evaluator_id`, and `player_id`. Only relevant player positions and potential labels are retained for modeling.

## Requirements

To run this project, you’ll need the following libraries:

- Python 3.7+
- Numpy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- LightGBM
- XGBoost
- SHAP



## Modeling and Evaluation

The project implements a variety of machine learning models to classify player potential:

- **Baseline Models**: Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Decision Tree.
- **Ensemble Models**: Random Forest, AdaBoost, Gradient Boosting, XGBoost, and LightGBM.

Each model is evaluated using cross-validation metrics like:
- **Accuracy**: Measures the overall correctness.
- **ROC-AUC**: Area Under the Receiver Operating Characteristic Curve for classification sensitivity.
- **F1 Score**: Balance between precision and recall, useful for imbalanced datasets.

To compare models, run:

```python
base_models(X, y, scoring="accuracy")
base_models(X, y, scoring="roc_auc")
base_models(X, y, scoring="f1")
```

## Feature Importance and Interpretability

To understand which features contribute most to player performance prediction, **SHAP** is used to interpret model outputs. SHAP values provide a detailed breakdown of feature influence on predictions.

### Visualizations Include:
1. **Feature Importance Plot**: Highlights the most influential features.
2. **Summary Plot**: Shows feature impact distribution across all samples.
3. **Waterfall Plot**: Visualizes individual prediction explanations.
4. **Force Plot**: Interactive visualization for detailed instance-level interpretation.

Run the following to generate SHAP visualizations:

```python
explainer = shap.Explainer(model)
shap_values = explainer(X)
shap.summary_plot(shap_values[:, :, 0], X)
```

These insights are invaluable for identifying key player attributes and understanding model decisions, aiding scouts and analysts in making more informed evaluations.

## Author
 
  Kaggle Profile: [kuvars](https://www.kaggle.com/kuvars)

