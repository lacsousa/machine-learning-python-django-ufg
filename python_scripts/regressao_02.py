import matplotlib.pyplot as plt
import numpy as np

# Dados observados (imóveis)
X_dados = np.array([480, 510, 520, 850, 960, 1200, 1400, 1650, 1700, 1920, 2350])
Y_dados = np.array([92, 96.5, 98, 147.5, 164, 200, 230, 267.5, 275, 308, 372.5])

# Plota os imóveis observados
plt.plot(X_dados, Y_dados, 'o', color='black')

# Parâmetros da regressão linear
a = 20 # Valor de partida
b = 0.15 # Quanto o preço aumenta para cada m² adicional

# Valores de X para desenhar a reta
X = np.linspace(0, 2500, 2500)

# Equação da regressão linear
Y = a + b * X

# Plota a reta da regressão
plt.plot(X, Y, '-r')

# Configuração do gráfico
plt.xlim(0, 2500)
plt.ylim(0, 400)

plt.xlabel('Área em $m^2$')
plt.ylabel("Preço em 1000's R$")
plt.title('Preço estimado do imóvel')

plt.grid()
plt.show()
