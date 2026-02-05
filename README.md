🌸 Iris Flower Classification – Machine Learning Project
📌 Project Overview

This project demonstrates an end-to-end machine learning workflow for classifying Iris flowers into three species using their physical measurements.
It covers data exploration, model building, evaluation, optimization, and deployment readiness, following industry-standard practices.

🎯 Objective
To build a supervised machine learning model that accurately predicts the species of an Iris flower based on:
Sepal length
Sepal width
Petal length
Petal width

📊 Dataset
Source: UCI Machine Learning Repository
Records: 150 samples
Features: 4 numerical features
Target: Iris species
Iris-setosa
Iris-versicolor
Iris-virginica
The dataset is clean, well-structured, and balanced across classes.

🛠️ Technologies Used
Programming Language: Python
Libraries:
pandas, numpy
matplotlib, seaborn
scikit-learn
joblib
Environment: Jupyter Notebook (VS Code)
Version Control: Git & GitHub

🔍 Exploratory Data Analysis (EDA)
Key insights from EDA:
The dataset is balanced, with equal samples for each class.
Petal length and petal width are the most discriminative features.
Iris-setosa is clearly separable, while minor overlap exists between versicolor and virginica.
No significant missing values or extreme outliers.

⚙️ Data Preprocessing
Features and target variable separated.
Train–test split performed (80–20) with stratification.
Feature scaling applied using StandardScaler to support distance-based models.

🤖 Model Building
Multiple models were trained and compared:
Logistic Regression
K-Nearest Neighbors (KNN)
Decision Tree
All models achieved similar accuracy due to the simplicity and separability of the dataset.

🧠 Final Model Selection
Logistic Regression was selected as the final model due to:
High and stable accuracy
Strong generalization performance
Interpretability of model coefficients
Suitability for linearly separable data

📈 Advanced Evaluation
To enhance robustness and professionalism:
Cross-validation was applied to ensure model stability.
Hyperparameter tuning (GridSearchCV) optimized model performance.
Feature importance analysis confirmed that petal features contribute most to predictions.

📊 Model Evaluation
Accuracy ≈ 93%
Confusion matrix analysis shows:
Perfect classification for Iris-setosa
Minor confusion between versicolor and virginica

💾 Model Persistence
The trained model and preprocessing steps were saved using joblib:
scaler.joblib
logistic_regression_best_model.joblib
This ensures consistent preprocessing and enables future deployment without retraining.
