import matplotlib.pyplot as plt
import numpy as np

# Valor de partida (preço base do imóvel)
a = 0

# Quanto o preço aumenta para cada m² adicional
b = 0.8

# Área do imóvel (m²)
X = np.linspace(0, 1000, 1000)

# Preço estimado do imóvel (em milhares de reais)
Y = a + b * X

# Plota a reta da regressão
plt.plot(X, Y, '-r')

# Configuração dos eixos
plt.xlim(0, 1000)
plt.ylim(0, 2000)
plt.xticks(np.arange(0, 1100, step=100))

# Rótulos
plt.xlabel('Área em $m^2$')
plt.ylabel("Preço em 1000's R$")
plt.title('Preço estimado do imóvel')

# Exibe a grade
plt.grid()

# Mostra o gráfico
plt.show()
