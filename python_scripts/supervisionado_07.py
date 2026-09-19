"""Permutation importance: impacto de cada variável na acurácia do modelo."""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


def main():
    dados = load_breast_cancer()
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        dados.data, dados.target, test_size=0.25, random_state=42, stratify=dados.target
    )
    modelo = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(X_treino, y_treino)
    resultado = permutation_importance(modelo, X_teste, y_teste, n_repeats=20, random_state=42, n_jobs=-1)
    importancia = pd.Series(resultado.importances_mean, index=dados.feature_names).sort_values(ascending=False)
    print(f"Acurácia antes do embaralhamento: {modelo.score(X_teste, y_teste):.3f}")
    print("\nMaior queda média de acurácia ao embaralhar:")
    print(importancia.head(10).to_string())


if __name__ == "__main__":
    main()
