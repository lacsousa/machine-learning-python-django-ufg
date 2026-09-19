"""Efeito da profundidade no overfitting de uma árvore de decisão."""

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    dados = load_breast_cancer()
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        dados.data, dados.target, test_size=0.25, random_state=42, stratify=dados.target
    )
    profundidades = range(1, 16)
    treino, teste = [], []
    for profundidade in profundidades:
        modelo = DecisionTreeClassifier(max_depth=profundidade, random_state=42).fit(X_treino, y_treino)
        treino.append(modelo.score(X_treino, y_treino))
        teste.append(modelo.score(X_teste, y_teste))

    melhor = profundidades[teste.index(max(teste))]
    print(f"Melhor profundidade no teste: {melhor} (acurácia {max(teste):.3f})")
    plt.plot(profundidades, treino, marker="o", label="Treino")
    plt.plot(profundidades, teste, marker="o", label="Teste")
    plt.xlabel("max_depth")
    plt.ylabel("Acurácia")
    plt.title("Árvore: ajuste versus generalização")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
