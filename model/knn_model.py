from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def train_knn(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)

    return model, scaler
