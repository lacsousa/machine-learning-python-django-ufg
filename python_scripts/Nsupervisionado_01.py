"""K-Means: agrupamento e escolha de k por inércia e silhouette score."""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def main():
    X, _ = make_blobs(n_samples=350, centers=4, cluster_std=1.15, random_state=42)
    X = StandardScaler().fit_transform(X)
    candidatos = range(2, 8)
    inercias, silhouettes = [], []
    for k in candidatos:
        modelo = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X)
        inercias.append(modelo.inertia_)
        silhouettes.append(silhouette_score(X, modelo.labels_))

    melhor_k = list(candidatos)[silhouettes.index(max(silhouettes))]
    modelo = KMeans(n_clusters=melhor_k, n_init=20, random_state=42).fit(X)
    print(f"Melhor k pelo silhouette score: {melhor_k} ({max(silhouettes):.3f})")

    fig, eixos = plt.subplots(1, 2, figsize=(12, 4))
    eixos[0].plot(list(candidatos), inercias, marker="o")
    eixos[0].set(title="Método do cotovelo", xlabel="k", ylabel="Inércia")
    eixos[0].grid(alpha=0.3)
    eixos[1].scatter(X[:, 0], X[:, 1], c=modelo.labels_, cmap="tab10")
    eixos[1].scatter(modelo.cluster_centers_[:, 0], modelo.cluster_centers_[:, 1], c="black", marker="X", s=160)
    eixos[1].set(title=f"K-Means com k = {melhor_k}", xlabel="Feature 1", ylabel="Feature 2")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
