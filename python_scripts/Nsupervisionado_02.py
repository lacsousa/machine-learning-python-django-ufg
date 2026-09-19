"""PCA: reduzir dimensões preservando a maior parte da variância."""

import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main():
    dados = load_wine()
    X = StandardScaler().fit_transform(dados.data)
    pca = PCA(n_components=2).fit(X)
    X_reduzido = pca.transform(X)
    print(f"Variância explicada por PC1 e PC2: {pca.explained_variance_ratio_}")
    print(f"Variância preservada: {pca.explained_variance_ratio_.sum():.2%}")

    for classe, nome in enumerate(dados.target_names):
        pontos = X_reduzido[dados.target == classe]
        plt.scatter(pontos[:, 0], pontos[:, 1], label=nome)
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.title("PCA: projeção não supervisionada")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
