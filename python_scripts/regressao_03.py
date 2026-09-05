import matplotlib.pyplot as plt
import numpy as np

# Dados observados (lotes)
X_dados = np.array([480, 510, 520, 850, 960, 1200, 1400, 1650, 1700, 1920, 2350])
Y_dados = np.array([98, 110, 200, 210, 280, 265, 300, 287, 325, 300, 290])

# Plota os lotes observados
plt.plot(X_dados, Y_dados, 'o', color='black')

# Configuração do gráfico
plt.xlim(0, 2500)
plt.ylim(0, 400)
plt.xlabel('Área em $m^2$')
plt.ylabel("Preço em 1000's R$")
plt.title('Preço estimado do imóvel')

#
## Ajustando manualmente os parâmetros da Regressão Linear
#
# Parâmetros da regressão linear
# Nota: No slide da aula propõe-se encontrar os valores corretos:
# a = ?? # Valor de partida
# b = ?.?? # Quanto o preço aumenta para cada m² adicional
# Conforme calculado analiticamente no slide 24:
a = 120.23986 # Valor de partida
b = 0.09914044 # Quanto o preço aumenta para cada m² adicional

# Valores de X para desenhar a reta
X = np.linspace(0, 2500, 2500)

# Equação da regressão linear
Y = a + b * X

# Plota a reta da regressão
plt.plot(X, Y, '-r')

plt.grid()
plt.show()
