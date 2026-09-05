import matplotlib.pyplot as plt
import numpy as np

# Dados observados (lotes)
X_dados = np.array([480, 510, 520, 850, 960, 1200, 1400, 1650, 1700, 1920, 2350])
Y_dados = np.array([98, 110, 200, 210, 280, 265, 300, 287, 325, 300, 290])

# -----------------------------------------------------------------
# Procura pelos melhores parâmetros utilizando o Mean Squared Error
# -----------------------------------------------------------------
menor_mse = np.inf
print("Valor Inicial para MSE:", menor_mse)
melhor_a = 0
melhor_b = 0

# Inclinação aproximada
b_estimado = (Y_dados.max() - Y_dados.min()) / (X_dados.max() - X_dados.min())

# Intercepto aproximado
a_estimado = np.mean(Y_dados - b_estimado * X_dados)

print("Estimado para a:", a_estimado)
print("Estimado para b:", b_estimado)

# Valores candidatos para a e b
for a in np.arange(a_estimado - 100, a_estimado + 100, 1):
    for b in np.arange(b_estimado - 0.10, b_estimado + 0.10, 0.001):
        # Preços previstos
        Y_predito = a + b * X_dados

        # Mean Squared Error
        mse = np.mean((Y_dados - Y_predito) ** 2)

        # Guarda os melhores parâmetros
        if mse < menor_mse:
            menor_mse = mse
            melhor_a = a
            melhor_b = b

print(f"Melhor valor de a: {melhor_a:.2f}")
print(f"Melhor valor de b: {melhor_b:.4f}")
print(f"MSE: {menor_mse:.2f}")

# -----------------------------------------------------------------
# Plota os dados
# -----------------------------------------------------------------
plt.plot(X_dados, Y_dados, 'o', color='black', label='Dados observados')

# Reta encontrada
X = np.linspace(0, 2500, 2500)
Y = melhor_a + melhor_b * X

plt.plot(X, Y, '-r', linewidth=2, label='Melhor reta')
plt.xlim(0, 2500)
plt.ylim(0, 400)
plt.xlabel('Área em $m^2$')
plt.ylabel("Preço em 1000's R$")
plt.title('Regressão Linear utilizando MSE')

plt.grid()
plt.legend()
plt.show()
