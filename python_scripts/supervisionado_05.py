"""Boosting: comparação entre Gradient Boosting e XGBoost, se disponível."""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def main():
    dados = load_breast_cancer()
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        dados.data, dados.target, test_size=0.25, random_state=42, stratify=dados.target
    )
    modelo = GradientBoostingClassifier(random_state=42).fit(X_treino, y_treino)
    print(f"GradientBoosting - acurácia no teste: {modelo.score(X_teste, y_teste):.3f}")
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("XGBoost não instalado. Para compará-lo: uv add xgboost")
        return

    xgb = XGBClassifier(eval_metric="logloss", random_state=42).fit(X_treino, y_treino)
    print(f"XGBoost - acurácia no teste: {xgb.score(X_teste, y_teste):.3f}")


if __name__ == "__main__":
    main()
