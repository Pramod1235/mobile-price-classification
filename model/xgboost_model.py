from xgboost import XGBClassifier

def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(set(y_train)),
        eval_metric="mlogloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
