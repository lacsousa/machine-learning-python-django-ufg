"""LDA para classificação e redução supervisionada de dimensionalidade."""

import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    dados = load_wine()
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        dados.data, dados.target, test_size=0.25, random_state=42, stratify=dados.target
    )
    scaler = StandardScaler().fit(X_treino)
    modelo = LinearDiscriminantAnalysis().fit(scaler.transform(X_treino), y_treino)
    print(f"Acurácia no teste: {modelo.score(scaler.transform(X_teste), y_teste):.3f}")

    componentes = modelo.transform(scaler.transform(dados.data))
    for classe, nome in enumerate(dados.target_names):
        pontos = componentes[dados.target == classe]
        plt.scatter(pontos[:, 0], pontos[:, 1], label=nome)
    plt.xlabel("Componente discriminante 1")
    plt.ylabel("Componente discriminante 2")
    plt.title("LDA: projeção que usa os rótulos")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
