# train.py
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Generowanie prostego zbioru danych
def generate_data(n_points1=75, n_points2=75):


    # Chmura 1 (wokół punktu [2,2])
    cloud1 = np.random.randn(n_points1, 2) * 1 + np.array([2, 2])
    labels1 = np.zeros(n_points1)

    # Chmura 2 (wokół punktu [5,5])
    cloud2 = np.random.randn(n_points2, 2) * 1 + np.array([10, 10])
    labels2 = np.ones(n_points2)

    # Połączenie danych
    X = np.vstack((cloud1, cloud2))
    y = np.hstack((labels1, labels2))

    return X, y

# Trenowanie prostego modelu regresji logistycznej
def train_model():
    X, y = generate_data()
    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    # Trenowanie modelu
    model = LogisticRegression()
    model.fit(X_train, y_train)
    # Predykcja na zbiorze testowym
    y_pred = model.predict(X_test)
    # Wyliczenie dokładności
    accuracy = accuracy_score(y_test, y_pred)
    # Zapis wyniku
    
    print(f"Model trained with accuracy: {accuracy * 100:.2f}%")
 # --- Wizualizacja ---
    plt.figure(figsize=(6, 6))

    # Punkty treningowe
    plt.scatter(
        X_train[y_train == 0, 0], X_train[y_train == 0, 1],
        color="blue", label="Chmura 1 (train)", alpha=0.6
    )
    plt.scatter(
        X_train[y_train == 1, 0], X_train[y_train == 1, 1],
        color="red", label="Chmura 2 (train)", alpha=0.6
    )

    # Punkty testowe
    plt.scatter(
        X_test[y_test == 0, 0], X_test[y_test == 0, 1],
        color="blue", edgecolor="black", marker="o", label="Chmura 1 (test)"
    )
    plt.scatter(
        X_test[y_test == 1, 0], X_test[y_test == 1, 1],
        color="red", edgecolor="black", marker="o", label="Chmura 2 (test)"
    )

    # Granica decyzyjna
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.RdBu)

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.savefig("plot.png")
    print("Wykres zapisany jako plot.png")
if __name__ == "__main__":
    train_model()
