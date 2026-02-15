from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def train_logistic(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    return model, scaler
