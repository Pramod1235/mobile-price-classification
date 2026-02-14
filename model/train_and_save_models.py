import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Create model directory
os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("train.csv")   # <-- ensure this CSV exists locally

X = df.drop("price_range", axis=1)
y = df["price_range"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling (for LR & KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

joblib.dump(scaler, "model/scaler.pkl")

# Models
models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "xgboost": XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        random_state=42
    )
}

# Train & save models
for name, model in models.items():
    if name in ["logistic_regression", "knn"]:
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)

    joblib.dump(model, f"model/{name}.pkl")
    print(f"Saved model/{name}.pkl")

print("✅ All models trained and saved")