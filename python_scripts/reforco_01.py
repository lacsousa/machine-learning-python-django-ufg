"""Q-Learning em um gridworld: encontre o objetivo evitando a penalidade."""

import numpy as np


ACOES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
LINHAS, COLUNAS = 4, 4
INICIO, OBJETIVO, PERIGO = (0, 0), (3, 3), (2, 2)


def proximo_estado(estado, acao):
    delta_linha, delta_coluna = ACOES[acao]
    linha = min(max(estado[0] + delta_linha, 0), LINHAS - 1)
    coluna = min(max(estado[1] + delta_coluna, 0), COLUNAS - 1)
    return linha, coluna


def recompensa(estado):
    if estado == OBJETIVO:
        return 10
    if estado == PERIGO:
        return -10
    return -0.2


def main():
    rng = np.random.default_rng(42)
    q = np.zeros((LINHAS, COLUNAS, len(ACOES)))
    alpha, gamma, epsilon = 0.2, 0.95, 0.15

    for _ in range(2_000):
        estado = INICIO
        for _ in range(60):
            acao = rng.integers(len(ACOES)) if rng.random() < epsilon else int(np.argmax(q[estado]))
            novo_estado = proximo_estado(estado, acao)
            r = recompensa(novo_estado)
            q[estado][acao] += alpha * (r + gamma * np.max(q[novo_estado]) - q[estado][acao])
            estado = novo_estado
            if estado in {OBJETIVO, PERIGO}:
                break

    nomes = ["cima", "baixo", "esquerda", "direita"]
    print("Melhor ação aprendida em cada estado (G = objetivo; X = perigo):")
    for linha in range(LINHAS):
        resultado = []
        for coluna in range(COLUNAS):
            estado = (linha, coluna)
            if estado == OBJETIVO:
                resultado.append("G")
            elif estado == PERIGO:
                resultado.append("X")
            else:
                resultado.append(nomes[int(np.argmax(q[estado]))][0].upper())
        print(" ".join(resultado))


if __name__ == "__main__":
    main()
