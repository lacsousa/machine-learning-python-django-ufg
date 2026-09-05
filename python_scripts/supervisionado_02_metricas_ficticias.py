"""
supervisionado_02_metricas_ficticias.py
Dissecação Matemática das Métricas de Avaliação em Classificação Multiclasse

Objetivo:
Demonstrar passo a passo, no código, como calcular:
- Matriz de Confusão (Diagonal Principal vs. Fora da Diagonal)
- Decomposição One-vs-Rest: VP, FP, FN, VN para cada classe
- Acurácia (Global, Por Classe, Macro, Micro, Weighted)
- Precisão (Precision: Por Classe, Macro, Micro, Weighted)
- Recall / Sensibilidade (Por Classe, Macro, Micro, Weighted)
- Especificidade (TNR / One-vs-Rest)
- F1-Score (Média Harmônica: Por Classe, Macro, Micro, Weighted)
"""

import numpy as np
import pandas as pd


def print_separator(title=""):
    print("\n" + "=" * 65)
    if title:
        print(f" {title.upper()} ".center(65, "="))
        print("=" * 65)


def main():
    classes = ["Setosa", "Virgínica", "Versicolor"]

    # ---------------------------------------------------------
    # MATRIZ DE CONFUSÃO HIPOTÉTICA
    # Linhas = Classe Predita pelo Modelo
    # Colunas = Classe Real (Verdadeira)
    # ---------------------------------------------------------
    matriz_confusao = np.array([
        # Real: Setosa, Virgínica, Versicolor
        [7, 1, 2],  # Predita: Setosa
        [3, 8, 4],  # Predita: Virgínica
        [5, 9, 6],  # Predita: Versicolor
    ])

    df_confusao = pd.DataFrame(
        matriz_confusao,
        index=[f"Pred: {c}" for c in classes],
        columns=[f"Real: {c}" for c in classes],
    )

    print_separator("Matriz de Confusão Multiclasse")
    print(df_confusao)

    total_amostras = matriz_confusao.sum()
    print(f"\nTotal de Elementos na Avaliação: {total_amostras}")

    # ---------------------------------------------------------
    # 1. DECOMPOSIÇÃO ONE-VS-REST PARA A CLASSE SETOSA
    # ---------------------------------------------------------
    idx_setosa = 0

    # Verdadeiro Positivo (VP): Predito Setosa E Real Setosa
    vp_setosa = matriz_confusao[idx_setosa, idx_setosa]

    # Falso Positivo (FP): Predito Setosa, mas Real NÃO é Setosa (soma da linha exceto DP)
    fp_setosa = matriz_confusao[idx_setosa, :].sum() - vp_setosa

    # Falso Negativo (FN): Real é Setosa, mas Predito NÃO é Setosa (soma da coluna exceto DP)
    fn_setosa = matriz_confusao[:, idx_setosa].sum() - vp_setosa

    # Verdadeiro Negativo (VN): Nem Predito Setosa, nem Real Setosa (submatriz restante)
    vn_setosa = total_amostras - (vp_setosa + fp_setosa + fn_setosa)

    print_separator("Decomposição One-vs-Rest para a Classe SETOSA")
    print(f"Verdadeiro Positivo (VP): {vp_setosa}")
    print(f"Falso Positivo      (FP): {fp_setosa} (1 + 2)")
    print(f"Falso Negativo      (FN): {fn_setosa} (3 + 5)")
    print(f"Verdadeiro Negativo (VN): {vn_setosa} (8 + 9 + 4 + 6)")
    print(f"Soma Total (VP+FP+FN+VN): {vp_setosa + fp_setosa + fn_setosa + vn_setosa}")

    # ---------------------------------------------------------
    # 2. MÉTRICAS PARA A CLASSE SETOSA
    # ---------------------------------------------------------
    # Precisão: VP / (VP + FP)
    precision_setosa = vp_setosa / (vp_setosa + fp_setosa)

    # Recall (Sensibilidade): VP / (VP + FN)
    recall_setosa = vp_setosa / (vp_setosa + fn_setosa)

    # Especificidade: VN / (VN + FP)
    especificidade_setosa = vn_setosa / (vn_setosa + fp_setosa)

    # Acurácia (One-vs-Rest para Setosa): (VP + VN) / Total
    acuracia_setosa = (vp_setosa + vn_setosa) / total_amostras

    # F1-Score: 2 * (P * R) / (P + R)
    f1_setosa = 2 * (precision_setosa * recall_setosa) / (precision_setosa + recall_setosa)

    print_separator("Métricas Calculadas para a Classe SETOSA")
    print(f"Precisão (Precision):     {precision_setosa * 100:.2f}% -> 7 / (7 + 3)")
    print(f"Recall (Sensibilidade):   {recall_setosa * 100:.2f}% -> 7 / (7 + 8)")
    print(f"Especificidade:           {especificidade_setosa * 100:.2f}% -> 27 / (27 + 3)")
    print(f"Acurácia (One-vs-Rest):   {acuracia_setosa * 100:.2f}% -> (7 + 27) / 45")
    print(f"F1-Score:                 {f1_setosa * 100:.2f}% -> 2 * (70.00 * 46.66) / (70.00 + 46.66)")

    # ---------------------------------------------------------
    # 3. RECALL POR CLASSE E RECALL MACRO
    # ---------------------------------------------------------
    recalls = []
    precisoes = []
    f1s = []
    suportes = []  # quantidade real por classe

    for i, nome in enumerate(classes):
        vp_i = matriz_confusao[i, i]
        fp_i = matriz_confusao[i, :].sum() - vp_i
        fn_i = matriz_confusao[:, i].sum() - vp_i
        total_real_i = matriz_confusao[:, i].sum()

        prec_i = vp_i / (vp_i + fp_i) if (vp_i + fp_i) > 0 else 0
        rec_i = vp_i / (vp_i + fn_i) if (vp_i + fn_i) > 0 else 0
        f1_i = 2 * (prec_i * rec_i) / (prec_i + rec_i) if (prec_i + rec_i) > 0 else 0

        recalls.append(rec_i)
        precisoes.append(prec_i)
        f1s.append(f1_i)
        suportes.append(total_real_i)

    print_separator("Recall Por Classe & Recall Macro")
    for nome, rec, sup in zip(classes, recalls, suportes):
        print(f"Recall {nome:10s}: {rec * 100:.2f}% (Total de amostras reais: {sup})")

    recall_macro = np.mean(recalls)
    print(f"\n--> Recall Macro (Média Simples): {recall_macro * 100:.2f}%")

    # Recall Weighted
    recall_weighted = sum(r * s for r, s in zip(recalls, suportes)) / total_amostras
    print(f"--> Recall Weighted (Média Ponderada pelo suporte): {recall_weighted * 100:.2f}%")

    # Recall Micro = soma(VP) / (soma(FN) + soma(VP))
    soma_vp = np.trace(matriz_confusao)
    soma_fn = sum(matriz_confusao[:, i].sum() - matriz_confusao[i, i] for i in range(3))
    recall_micro = soma_vp / (soma_fn + soma_vp)
    print(f"--> Recall Micro (Soma Global): {recall_micro * 100:.2f}%")

    # ---------------------------------------------------------
    # 4. TABELA RESUMO DE MÉTRICAS
    # ---------------------------------------------------------
    print_separator("Tabela Comparativa de Médias Multiclasse")
    tabela_medias = pd.DataFrame({
        "Macro (Média Simples)": [
            f"{np.mean(precisoes) * 100:.2f}%",
            f"{recall_macro * 100:.2f}%",
            f"{np.mean(f1s) * 100:.2f}%"
        ],
        "Weighted (Ponderada)": [
            f"{(sum(p * s for p, s in zip(precisoes, suportes)) / total_amostras) * 100:.2f}%",
            f"{recall_weighted * 100:.2f}%",
            f"{(sum(f * s for f, s in zip(f1s, suportes)) / total_amostras) * 100:.2f}%"
        ],
        "Micro (Global)": [
            f"{(soma_vp / total_amostras) * 100:.2f}%",
            f"{recall_micro * 100:.2f}%",
            f"{(soma_vp / total_amostras) * 100:.2f}%"
        ]
    }, index=["Precisão", "Recall", "F1-Score"])

    print(tabela_medias)

    # ---------------------------------------------------------
    # 5. ACURÁCIA GLOBAL DO MODELO MULTICLASSE
    # ---------------------------------------------------------
    # Para uma matriz de teste com total de 38 amostras e 21 acertos:
    acertos_globais = 21
    total_teste = 38
    acuracia_global = acertos_globais / total_teste
    print_separator("Acurácia Global do Modelo")
    print(f"Total de acertos na Diagonal Principal: 7 + 8 + 6 = {acertos_globais}")
    print(f"Total de elementos no teste: {total_teste}")
    print(f"Acurácia Global: {acertos_globais} / {total_teste} = {acuracia_global * 100:.2f}%")


if __name__ == "__main__":
    main()
