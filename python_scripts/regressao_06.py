import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ------------------------------------------------------------
# Leitura do arquivo CSV
# ------------------------------------------------------------
df = pd.read_csv("regressao_06.csv")

# Remove espaços dos nomes das colunas
df.columns = df.columns.str.strip()

# ------------------------------------------------------------
# Dados observados
# X = Life expectancy
# Y = Adult Mortality
# ------------------------------------------------------------
X_dados = df["Life expectancy"].values
Y_dados = df["Adult Mortality"].values

# Remove linhas com valores ausentes
dados = pd.DataFrame({
    "X": X_dados,
    "Y": Y_dados
}).dropna()

X_dados = dados["X"].values
Y_dados = dados["Y"].values

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
print("\nEquação da reta:")
print(f"Adult Mortality = {a:.4f} + {b:.6f} × Life expectancy")

# ------------------------------------------------------------
# Gráfico
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(X_dados, Y_dados, color="black", label="Dados observados")
X_linha = np.linspace(X_dados.min(), X_dados.max(), 300).reshape(-1, 1)
Y_linha = modelo.predict(X_linha)
plt.plot(X_linha, Y_linha, color="red", linewidth=2, label="Regressão Linear")
plt.xlabel("Life expectancy")
plt.ylabel("Adult Mortality")
plt.title("Regressão Linear utilizando Scikit-Learn")
plt.grid(True)
plt.legend()
plt.show()
