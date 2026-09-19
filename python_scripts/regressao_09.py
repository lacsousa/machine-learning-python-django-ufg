"""Regressão linear múltipla para prever preço a partir de área e quartos."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    rng = np.random.default_rng(42)
    area = rng.integers(45, 220, size=120)
    quartos = rng.integers(1, 5, size=120)
    preco = 80_000 + 3_200 * area + 35_000 * quartos + rng.normal(0, 35_000, size=120)
    X = np.column_stack((area, quartos))

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, preco, test_size=0.25, random_state=42
    )
    modelo = LinearRegression().fit(X_treino, y_treino)
    previsoes = modelo.predict(X_teste)

    print("Regressão linear múltipla")
    print(f"Preço = {modelo.intercept_:.2f} + {modelo.coef_[0]:.2f}*área + {modelo.coef_[1]:.2f}*quartos")
    print(f"MSE no teste: {mean_squared_error(y_teste, previsoes):.2f}")
    print(f"R² no teste: {r2_score(y_teste, previsoes):.3f}")
    exemplo = np.array([[100, 3]])
    print(f"Previsão para 100 m² e 3 quartos: R$ {modelo.predict(exemplo)[0]:,.2f}")


if __name__ == "__main__":
    main()
