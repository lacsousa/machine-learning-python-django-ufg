"""Regressão polinomial: quando uma reta não representa bem os dados."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


def main():
    rng = np.random.default_rng(42)
    x = np.linspace(-3, 3, 60)
    y = 2 + 1.5 * x + 0.9 * x**2 + rng.normal(0, 1.5, size=x.size)
    X = x.reshape(-1, 1)

    transformador = PolynomialFeatures(degree=2, include_bias=False)
    X_polinomial = transformador.fit_transform(X)
    modelo = LinearRegression().fit(X_polinomial, y)
    previsoes = modelo.predict(X_polinomial)

    print("Regressão polinomial de grau 2")
    print(f"Intercepto: {modelo.intercept_:.3f}")
    print(f"Coeficientes [x, x²]: {modelo.coef_}")
    print(f"MSE: {mean_squared_error(y, previsoes):.3f}")
    print(f"R²: {r2_score(y, previsoes):.3f}")

    x_linha = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)
    y_linha = modelo.predict(transformador.transform(x_linha))
    plt.scatter(x, y, color="black", label="Dados observados")
    plt.plot(x_linha, y_linha, color="crimson", linewidth=2, label="Polinômio grau 2")
    plt.xlabel("Variável de entrada")
    plt.ylabel("Variável alvo")
    plt.title("Regressão polinomial")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
