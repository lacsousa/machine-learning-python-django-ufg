"""Comparação didática: PCA ignora rótulos; LDA usa rótulos para separar classes."""

import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


def desenhar(eixo, dados, y, nomes, titulo, x_label, y_label):
    for classe, nome in enumerate(nomes):
        pontos = dados[y == classe]
        eixo.scatter(pontos[:, 0], pontos[:, 1], label=nome)
    eixo.set(title=titulo, xlabel=x_label, ylabel=y_label)
    eixo.grid(alpha=0.3)


def main():
    base = load_wine()
    X = StandardScaler().fit_transform(base.data)
    pca = PCA(n_components=2).fit_transform(X)
    lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, base.target)
    print("PCA não recebe y no fit; LDA recebe y e maximiza a separação entre classes.")

    fig, eixos = plt.subplots(1, 2, figsize=(12, 5))
    desenhar(eixos[0], pca, base.target, base.target_names, "PCA", "PC1", "PC2")
    desenhar(eixos[1], lda, base.target, base.target_names, "LDA", "LD1", "LD2")
    eixos[1].legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
