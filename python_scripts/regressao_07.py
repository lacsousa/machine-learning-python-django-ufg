import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ==========================================================
# DADOS
# ==========================================================
X = np.array([480, 510, 520, 850, 960, 1200, 1400, 1650, 1700, 1920, 2350], dtype=float)
Y = np.array([98, 110, 200, 210, 280, 265, 300, 287, 325, 300, 290], dtype=float)
m = len(X)

# ==========================================================
# NORMALIZAÇÃO
# (Facilita o Gradiente Descendente)
# ==========================================================
media = np.mean(X)
desvio = np.std(X)
Xn = (X - media) / desvio

# ==========================================================
# PARÂMETROS INICIAIS
# ==========================================================
a = 0.0
b = 0.0
taxa = 0.05
epocas = 1000
historico_custo = []

# ==========================================================
# GRADIENTE DESCENDENTE
# ==========================================================
for epoca in range(epocas):
    # previsão
    Y_pred = a + b * Xn
    # erro
    erro = Y_pred - Y
    # função de custo
    J = np.sum(erro ** 2) / (2 * m)
    historico_custo.append(J)

    # gradientes
    da = np.sum(erro) / m
    db = np.sum(erro * Xn) / m

    # atualização
    a = a - taxa * da
    b = b - taxa * db

# ==========================================================
# RESULTADO
# ==========================================================
print("\n==============================")
print("GRADIENTE DESCENDENTE")
print("==============================")
print(f"Intercepto (normalizado): {a:.6f}")
print(f"Inclinação (normalizada): {b:.6f}")

# ==========================================================
# PREVISÕES
# ==========================================================
Y_pred = a + b * Xn
mse = mean_squared_error(Y, Y_pred)
print(f"MSE = {mse:.6f}")

# ==========================================================
# COMPARAÇÃO COM SCIKIT-LEARN
# ==========================================================
modelo = LinearRegression()
modelo.fit(X.reshape(-1, 1), Y)

print("\n==============================")
print("SCIKIT-LEARN")
print("==============================")
print(f"Intercepto = {modelo.intercept_:.6f}")
print(f"Inclinação = {modelo.coef_[0]:.6f}")
print(f"MSE = {mean_squared_error(Y, modelo.predict(X.reshape(-1, 1))):.6f}")

# ==========================================================
# RETA
# ==========================================================
x_plot = np.linspace(min(X), max(X), 300)
x_plot_norm = (x_plot - media) / desvio
y_plot = a + b * x_plot_norm

# ==========================================================
# GRÁFICO 1
# ==========================================================
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color="black", s=70, label="Dados")
plt.plot(x_plot, y_plot, color="red", linewidth=3, label="Gradiente Descendente")
plt.xlabel("Área (m²)")
plt.ylabel("Preço (1000's R$)")
plt.title("Regressão Linear - Gradiente Descendente")
plt.grid(True)
plt.legend()
plt.show()

# ==========================================================
# GRÁFICO DA FUNÇÃO DE CUSTO
# ==========================================================
plt.figure(figsize=(10, 5))
plt.plot(historico_custo, linewidth=2)
plt.xlabel("Época")
plt.ylabel("Função de Custo J")
plt.title("Convergência do Gradiente Descendente")
plt.grid(True)
plt.show()

# ==========================================================
# CONSULTA
# ==========================================================
print("\n==============================")
print("Estimativa de Preço")
print("Digite 0 para sair.")
print("==============================")
while True:
    try:
        area = float(input("\nÁrea (m²): "))
        if area == 0:
            break
        area_norm = (area - media) / desvio
        preco = a + b * area_norm
        print(f"Preço estimado = R$ {preco * 1000:,.2f}")
    except (ValueError, EOFError):
        break
