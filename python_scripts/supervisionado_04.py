"""SVM com padronização e busca dos hiperparâmetros C e gamma."""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    dados = load_breast_cancer()
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        dados.data, dados.target, test_size=0.25, random_state=42, stratify=dados.target
    )
    pipeline = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf"))])
    busca = GridSearchCV(
        pipeline, {"svm__C": [0.1, 1, 10], "svm__gamma": ["scale", 0.01, 0.1]}, cv=5, n_jobs=-1
    ).fit(X_treino, y_treino)
    print(f"Melhores parâmetros: {busca.best_params_}")
    print(f"Melhor acurácia CV: {busca.best_score_:.3f}")
    print(f"Acurácia no teste: {busca.score(X_teste, y_teste):.3f}")


if __name__ == "__main__":
    main()
