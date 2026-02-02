# Mobile Price Classification using Machine Learning

## a. Problem Statement
The objective of this project is to build and compare multiple machine learning
classification models to predict the price range of a mobile phone based on its
technical specifications. The predicted price range is a categorical variable
representing different cost segments. The project also includes an interactive
Streamlit web application for model evaluation and demonstration.

---

## b. Dataset Description
The dataset used for this project is the **Mobile Price Classification Dataset**
sourced from Kaggle. It contains technical specifications of mobile phones such as
battery power, RAM, internal memory, screen resolution, and connectivity features.

- Number of instances: 2000+
- Number of features: 20
- Target variable: `price_range` (0 = Low, 1 = Medium, 2 = High, 3 = Very High)
- All features are numerical, making the dataset suitable for multiple
classification algorithms.

---

## c. Models Used and Evaluation Metrics

The following six classification models were implemented on the same dataset:

- Logistic Regression  
- Decision Tree Classifier  
- K-Nearest Neighbors (kNN)  
- Naive Bayes Classifier  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)

Each model was evaluated using the following metrics:
Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation
Coefficient (MCC).

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.95 | 0.99 | 0.95 | 0.95 | 0.95 | 0.94 |
| Decision Tree | 0.86 | 0.92 | 0.86 | 0.86 | 0.86 | 0.81 |
| kNN | 0.91 | 0.97 | 0.91 | 0.91 | 0.91 | 0.88 |
| Naive Bayes | 0.82 | 0.90 | 0.82 | 0.82 | 0.82 | 0.76 |
| Random Forest (Ensemble) | 0.94 | 0.99 | 0.94 | 0.94 | 0.94 | 0.93 |
| XGBoost (Ensemble) | 0.96 | 0.99 | 0.96 | 0.96 | 0.96 | 0.95 |

---

## Model Performance Observations

| ML Model Name | Observation about Model Performance |
|--------------|--------------------------------------|
| Logistic Regression | Performed well due to linear separability and effective feature scaling. |
| Decision Tree | Easy to interpret but showed slight overfitting compared to ensemble models. |
| kNN | Achieved good performance but was sensitive to feature scaling and choice of k. |
| Naive Bayes | Fast and simple but underperformed due to independence assumptions. |
| Random Forest (Ensemble) | Delivered strong performance with reduced overfitting through bagging. |
| XGBoost (Ensemble) | Achieved the best overall performance due to boosting and optimized learning. |

---

## Streamlit Application Features
The deployed Streamlit application includes the following features:
- CSV dataset upload option
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix visualization

---

## Deployment
The application is deployed using **Streamlit Community Cloud**.
The repository contains:
- `app.py`
- `requirements.txt`
- `README.md`
- Model implementation code

---

## Execution Environment
The assignment was executed on **BITS Virtual Lab**, and a screenshot of execution
has been captured as proof, as required.
