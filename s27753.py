# train.py
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def generate_data(n_points1=75, n_points2=75):


    cloud1 = np.random.randn(n_points1, 2) * 1 + np.array([2, 2])
    labels1 = np.zeros(n_points1)

    cloud2 = np.random.randn(n_points2, 2) * 1 + np.array([10, 10])
    labels2 = np.ones(n_points2)

    X = np.vstack((cloud1, cloud2))
    y = np.hstack((labels1, labels2))

    return X, y

def train_model():
    X, y = generate_data()
    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model trained with accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    train_model()
