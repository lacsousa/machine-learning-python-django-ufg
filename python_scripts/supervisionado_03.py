"""Random Forest: conjunto de árvores para classificação robusta."""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main():
    dados = load_breast_cancer()
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        dados.data, dados.target, test_size=0.25, random_state=42, stratify=dados.target
    )
    modelo = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(X_treino, y_treino)
    print(f"Acurácia no teste: {modelo.score(X_teste, y_teste):.3f}")
    importancia = pd.Series(modelo.feature_importances_, index=dados.feature_names).sort_values(ascending=False)
    print("\nCinco variáveis mais importantes:")
    print(importancia.head().to_string())


if __name__ == "__main__":
    main()
