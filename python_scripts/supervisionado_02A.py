"""Árvore de decisão: classificação e visualização de regras."""

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


def main():
    dados = load_breast_cancer()
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        dados.data, dados.target, test_size=0.25, random_state=42, stratify=dados.target
    )
    modelo = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_treino, y_treino)
    previsoes = modelo.predict(X_teste)
    print(f"Acurácia: {accuracy_score(y_teste, previsoes):.3f}")
    print(classification_report(y_teste, previsoes, target_names=dados.target_names))

    plt.figure(figsize=(16, 8))
    plot_tree(modelo, feature_names=dados.feature_names, class_names=dados.target_names,
              filled=True, rounded=True, max_depth=3, fontsize=7)
    plt.title("Árvore de decisão (profundidade máxima = 3)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
