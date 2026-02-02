import streamlit as st
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# Page config
# -----------------------------------
st.set_page_config(page_title="Mobile Price Classification", layout="wide")

st.title("📱 Mobile Price Classification App")
st.write("Evaluate multiple ML models and predict mobile price range")

# -----------------------------------
# Load models safely
# -----------------------------------
MODEL_DIR = "model"

@st.cache_resource
def load_models():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    models = {
        "Logistic Regression": joblib.load(f"{MODEL_DIR}/logistic_regression.pkl"),
        "Decision Tree": joblib.load(f"{MODEL_DIR}/decision_tree.pkl"),
        "KNN": joblib.load(f"{MODEL_DIR}/knn.pkl"),
        "Naive Bayes": joblib.load(f"{MODEL_DIR}/naive_bayes.pkl"),
        "Random Forest": joblib.load(f"{MODEL_DIR}/random_forest.pkl"),
        "XGBoost": joblib.load(f"{MODEL_DIR}/xgboost.pkl"),
    }
    return scaler, models

scaler, models = load_models()

# -----------------------------------
# Dataset upload
# -----------------------------------
uploaded_file = st.file_uploader(
    "Upload Mobile Price CSV (must contain price_range column)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("👆 Please upload a dataset to continue")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -----------------------------------
# Feature / target split
# -----------------------------------
if "price_range" not in df.columns:
    st.error("❌ Dataset must contain 'price_range' column")
    st.stop()

X = df.drop("price_range", axis=1)
y = df["price_range"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------
# Model selection
# -----------------------------------
model_name = st.selectbox(
    "Select Classification Model",
    list(models.keys())
)

model = models[model_name]

# -----------------------------------
# Predictions
# -----------------------------------
if model_name in ["Logistic Regression", "KNN"]:
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
else:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

# -----------------------------------
# Metrics
# -----------------------------------
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")
mcc = matthews_corrcoef(y_test, y_pred)

st.subheader("📈 Model Performance Metrics")

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{accuracy:.4f}")
c1.metric("AUC", f"{auc:.4f}")
c2.metric("Precision", f"{precision:.4f}")
c2.metric("Recall", f"{recall:.4f}")
c3.metric("F1 Score", f"{f1:.4f}")
c3.metric("MCC", f"{mcc:.4f}")

# -----------------------------------
# Confusion Matrix
# -----------------------------------
st.subheader("🔍 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ======================================================
# EXTRA FEATURE: Custom Mobile Price Prediction
# ======================================================
st.markdown("---")
st.subheader("📱 Test Mobile Price with Custom Inputs")
st.write("Adjust the feature values to predict the mobile price range.")

input_data = {}

for col in X.columns:
    min_val = int(X[col].min())
    max_val = int(X[col].max())
    default_val = int(X[col].mean())

    input_data[col] = st.slider(
        col,
        min_value=min_val,
        max_value=max_val,
        value=default_val
    )

input_df = pd.DataFrame([input_data])

if st.button("Predict Price Range"):
    if model_name in ["Logistic Regression", "KNN"]:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
    else:
        prediction = model.predict(input_df)

    price_map = {
        0: "Low Cost",
        1: "Medium Cost",
        2: "High Cost",
        3: "Very High Cost"
    }

    st.success(
        f"💰 Predicted Mobile Price Range: **{price_map[int(prediction[0])]}**"
    )
