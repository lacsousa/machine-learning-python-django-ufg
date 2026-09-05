import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ------------------------------------------------------------
# Dados observados
# ------------------------------------------------------------
X_dados = np.array([480, 510, 520, 850, 960, 1200, 1400, 1650, 1700, 1920, 2350])
Y_dados = np.array([98, 110, 200, 210, 280, 265, 300, 287, 325, 300, 290])

# O Scikit-Learn exige uma matriz (n_amostras, n_variáveis)
X = X_dados.reshape(-1, 1)

# ------------------------------------------------------------
# Criação do modelo
# ------------------------------------------------------------
modelo = LinearRegression()

# Treinamento
modelo.fit(X, Y_dados)

# ------------------------------------------------------------
# Coeficientes da reta
# ------------------------------------------------------------
a = modelo.intercept_
b = modelo.coef_[0]

print(f"Intercepto (a): {a:.4f}")
print(f"Inclinação (b): {b:.6f}")

# ------------------------------------------------------------
# Previsões
# ------------------------------------------------------------
Y_predito = modelo.predict(X)

# ------------------------------------------------------------
# Mean Squared Error
# ------------------------------------------------------------
mse = mean_squared_error(Y_dados, Y_predito)
print(f"MSE: {mse:.4f}")

# ------------------------------------------------------------
# Equação da reta
# ------------------------------------------------------------
print(f"\nEquação da reta:")
print(f"Preço = {a:.4f} + {b:.6f} × Área")

# ------------------------------------------------------------
# Gráfico
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(X_dados, Y_dados, color="black", label="Dados observados")
X_linha = np.linspace(0, 2500, 300).reshape(-1, 1)
Y_linha = modelo.predict(X_linha)
plt.plot(X_linha, Y_linha, color="red", linewidth=2, label="Regressão Linear")
plt.xlim(0, 2500)
plt.ylim(0, 400)
plt.xlabel("Área em m²")
plt.ylabel("Preço (1000's R$)")
plt.title("Regressão Linear utilizando Scikit-Learn")
plt.grid(True)
plt.legend()
plt.show()
