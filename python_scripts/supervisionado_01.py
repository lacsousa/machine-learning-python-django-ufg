"""
supervisionado_01.py
Aprendizado Supervisionado: Classificação Multiclasse com o Dataset Íris
Baseado nas aulas do Prof. Dr. Ronaldo Martins da Costa (UFG / INF)

Etapas cobertas:
1. Carregamento e inspeção do Dataset Íris (features, labels, formato).
2. Visualização gráfica: Dispersão 2D e Projeção 3D com Matplotlib.
3. Particionamento de dados com train_test_split (stratify e random_state).
4. Criação de Pipeline de Machine Learning (StandardScaler + KNeighborsClassifier).
5. Otimização de hiperparâmetros com GridSearchCV (10-Fold Cross-Validation).
6. Avaliação do modelo em dados de teste (Acurácia, Matriz de Confusão, Classification Report).
7. Exemplo de predição pontual com novas amostras.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # Necessário para projeção '3d'
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    # ---------------------------------------------------------
    # 1. CARREGAMENTO E EXPLORAÇÃO DOS DADOS
    # ---------------------------------------------------------
    iris = load_iris()
    X = iris.data
    y = iris.target

    print("\n================ DATASET =================")
    print("Quantidade de amostras:", X.shape[0])
    print("Quantidade de caracteristicas:", X.shape[1])

    print("\n================ CARACTERISTICAS =================")
    for idx, feature_name in enumerate(iris.feature_names):
        print(f"{idx} - {feature_name}")

    print("\nPrimeiras 5 amostras (X):")
    print(X[:5])

    print("\n================ CLASSES =================")
    print(iris.target_names)
    print("Primeiros 5 valores de y:")
    print(y[:5])

    # ---------------------------------------------------------
    # 2. VISUALIZAÇÃO GRÁFICA (2D E 3D)
    # ---------------------------------------------------------
    # Gráfico 2D: Comprimento da Sépala vs. Largura da Sépala
    fig2d, ax2d = plt.subplots(figsize=(8, 6))
    scatter_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    for class_idx, class_name in enumerate(iris.target_names):
        mask = (y == class_idx)
        ax2d.scatter(
            X[mask, 0],
            X[mask, 1],
            label=class_name,
            color=scatter_colors[class_idx],
            edgecolor="k",
            alpha=0.8,
            s=45
        )

    ax2d.set_title("Distribuição Íris (2D)", fontsize=13, fontweight="bold")
    ax2d.set_xlabel("Comprimento da Sépala (cm)")
    ax2d.set_ylabel("Largura da Sépala (cm)")
    ax2d.legend(title="Classes")
    ax2d.grid(True, linestyle="--", alpha=0.5)

    print("\nExibindo gráfico 2D...")
    print("Feche a janela do gráfico para continuar...")
    plt.show()

    # Gráfico 3D: Sépala (Comprimento, Largura) e Pétala (Comprimento)
    fig3d = plt.figure(figsize=(9, 7))
    ax3d = fig3d.add_subplot(111, projection="3d")

    for class_idx, class_name in enumerate(iris.target_names):
        mask = (y == class_idx)
        ax3d.scatter(
            X[mask, 0],
            X[mask, 1],
            X[mask, 2],
            label=class_name,
            color=scatter_colors[class_idx],
            edgecolor="k",
            s=50,
            alpha=0.8
        )

    ax3d.set_title("Distribuição Íris (3D)", fontsize=13, fontweight="bold")
    ax3d.set_xlabel("Comprimento da Sépala (cm)")
    ax3d.set_ylabel("Largura da Sépala (cm)")
    ax3d.set_zlabel("Comprimento da Pétala (cm)")
    ax3d.legend(title="Classes")

    print("\nExibindo gráfico 3D...")
    print("Feche a janela do gráfico 3D para continuar...")
    plt.show()

    # ---------------------------------------------------------
    # 3. DIVISÃO EM TREINAMENTO E TESTE (ESTRATIFICADA)
    # ---------------------------------------------------------
    # test_size=0.15 -> 15% para teste final, 85% para treino/validação cruzada
    # random_state=42 -> reprodutibilidade no embaralhamento
    # stratify=y -> mantém a mesma proporção de classes em ambos os conjuntos
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
        stratify=y
    )

    print("\n================ DIVISÃO DOS DADOS =================")
    print(f"Total de amostras de treino: {X_train.shape[0]}")
    print(f"Total de amostras de teste:  {X_test.shape[0]}")
    print("Distribuição das classes no treino (y_train):", np.bincount(y_train))
    print("Distribuição das classes no teste  (y_test): ", np.bincount(y_test))

    # ---------------------------------------------------------
    # 4. CRIAÇÃO DO PIPELINE
    # ---------------------------------------------------------
    # O pipeline garante que o escalonamento (StandardScaler) seja calculado
    # apenas nos folds de treino de cada iteração da validação cruzada,
    # prevenindo vazamento de dados (data leakage).
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])

    # ---------------------------------------------------------
    # 5. AJUSTE DE HIPERPARÂMETROS COM GRIDSEARCHCV
    # ---------------------------------------------------------
    parametros = {
        "knn__n_neighbors": range(1, 8),
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
        "knn__algorithm": ["auto", "ball_tree", "kd_tree"]
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=parametros,
        cv=10,
        scoring="accuracy",
        n_jobs=-1,
        return_train_score=True
    )

    print("\nIniciando busca em grade (GridSearchCV) com 10 Folds...")
    grid.fit(X_train, y_train)

    print("\n================ RESULTADOS DO GRIDSEARCH =================")
    print("Melhores Hiperparâmetros encontrados:")
    for param, val in grid.best_params_.items():
        print(f"  {param}: {val}")
    print(f"Melhor pontuação média de Treinamento/CV (Acurácia): {grid.best_score_:.4f}")

    # ---------------------------------------------------------
    # 6. AVALIAÇÃO DO MODELO NO CONJUNTO DE TESTE
    # ---------------------------------------------------------
    pred_test = grid.predict(X_test)
    acuracia = accuracy_score(y_test, pred_test)
    matriz_confusao = confusion_matrix(y_test, pred_test)

    print("\n================ AVALIAÇÃO NO CONJUNTO DE TESTE =================")
    print(f"Acurácia no Teste: {acuracia:.4f} ({acuracia * 100:.2f}%)")

    print("\nMatriz de Confusão:")
    print(matriz_confusao)

    print("\nRelatório de Classificação Detalhado:")
    print(classification_report(y_test, pred_test, target_names=iris.target_names))

    # ---------------------------------------------------------
    # 7. PREVISÃO PONTUAL (NOVA AMOSTRA)
    # ---------------------------------------------------------
    # Exemplo similar ao apresentado em aula: prevendo para uma nova flor
    # Suponha uma amostra com características: [5.0, 3.4, 1.5, 0.2]
    amostra_exemplo = np.array([[5.0, 3.4, 1.5, 0.2]])
    pred_amostra = grid.predict(amostra_exemplo)
    prob_amostra = grid.predict_proba(amostra_exemplo)

    print("\n================ PREVISÃO PONTUAL =================")
    print("Características da flor:", amostra_exemplo[0])
    print(f"Classe Predita: {pred_amostra[0]} -> {iris.target_names[pred_amostra[0]]}")
    print("Probabilidades estimadas por classe:")
    for nome_classe, prob in zip(iris.target_names, prob_amostra[0]):
        print(f"  {nome_classe}: {prob * 100:.2f}%")


if __name__ == "__main__":
    main()
