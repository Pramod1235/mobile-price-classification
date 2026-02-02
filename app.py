
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------
# Streamlit Page Config
# -----------------------------------
st.set_page_config(page_title="Mobile Price Classification", layout="wide")

st.title("📱 Mobile Price Classification App")
st.write("Predict mobile price range using multiple ML models")

# -----------------------------------
# Upload Dataset
# -----------------------------------
uploaded_file = st.file_uploader(
    "Upload Mobile Price CSV (with price_range column)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------------
    # Split Features & Target
    # -----------------------------------
    X = df.drop("price_range", axis=1)
    y = df["price_range"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------------
    # Model Selection
    # -----------------------------------
    model_name = st.selectbox(
        "Select Classification Model",
        (
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        )
    )

    # -----------------------------------
    # Model Initialization
    # -----------------------------------
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)

    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

    elif model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)

    elif model_name == "Naive Bayes":
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

    else:  # XGBoost
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=4,
            eval_metric="mlogloss",
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

    # -----------------------------------
    # Evaluation Metrics
    # -----------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_test, y_pred)

    # -----------------------------------
    # Display Metrics
    # -----------------------------------
    st.subheader("📈 Model Performance Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.4f}")
    col1.metric("AUC", f"{auc:.4f}")

    col2.metric("Precision", f"{precision:.4f}")
    col2.metric("Recall", f"{recall:.4f}")

    col3.metric("F1 Score", f"{f1:.4f}")
    col3.metric("MCC", f"{mcc:.4f}")

    # -----------------------------------
    # Confusion Matrix
    # -----------------------------------
    st.subheader("🔍 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

else:
    st.info("👆 Please upload the Mobile Price dataset CSV file")

