"""Regressão múltipla treinada manualmente com gradiente descendente."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def main():
    rng = np.random.default_rng(42)
    area = rng.integers(45, 220, size=120).astype(float)
    quartos = rng.integers(1, 5, size=120).astype(float)
    y = 80_000 + 3_200 * area + 35_000 * quartos + rng.normal(0, 35_000, size=120)
    X_original = np.column_stack((area, quartos))

    media, desvio = X_original.mean(axis=0), X_original.std(axis=0)
    X = (X_original - media) / desvio
    X = np.column_stack((np.ones(X.shape[0]), X))
    pesos = np.zeros(X.shape[1])
    taxa, epocas = 0.08, 2_000

    for _ in range(epocas):
        erro = X @ pesos - y
        gradiente = (2 / len(y)) * X.T @ erro
        pesos -= taxa * gradiente

    previsoes = X @ pesos
    referencia = LinearRegression().fit(X_original, y)
    print("Gradiente descendente - regressão múltipla")
    print(f"MSE: {mean_squared_error(y, previsoes):.2f}")
    print(f"Pesos na escala normalizada: {pesos}")
    print(f"MSE do LinearRegression: {mean_squared_error(y, referencia.predict(X_original)):.2f}")
    exemplo = (np.array([[100.0, 3.0]]) - media) / desvio
    print(f"Previsão para 100 m² e 3 quartos: R$ {(np.r_[1, exemplo[0]] @ pesos):,.2f}")


if __name__ == "__main__":
    main()
