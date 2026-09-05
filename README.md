# Projeto de AM (Aprendizado de Máquina)

Este repositório contém o projeto desenvolvido para a disciplina de AM (Aprendizado de Máquina) da Pós-Graduação da UFG.

## 🚀 Como começar (Instalação e Execução)

Este projeto utiliza o gerenciador de pacotes e ambientes Python [**`uv`**](https://github.com/astral-sh/uv), que é extremamente rápido e eficiente.

### 1. Como baixar (Clonar o repositório)

Primeiro, faça o clone do repositório para a sua máquina local e acesse o diretório do projeto:

```bash
git clone git@github.com:lacsousa/machine-learning-python-django-ufg.git
cd machine-learning-python-django-ufg
```

### 2. Pré-requisitos

Certifique-se de ter instalado no seu computador:
- **Python** (versão 3.10 ou superior)
- **uv** (Se não tiver o `uv` instalado, instale-o executando: `curl -LsSf https://astral.sh/uv/install.sh | sh` no Mac/Linux ou consulte a [documentação do uv](https://github.com/astral-sh/uv))

### 3. Como configurar o ambiente e instalar dependências

O `uv` facilita muito esse processo. Dentro da pasta do projeto, apenas rode o comando abaixo. Ele irá automaticamente criar o ambiente virtual (equivalente ao `am-env` dos slides) e sincronizar as dependências:

```bash
uv sync
```

### 4. Como executar o projeto

Temos dois ambientes configurados para rodar o projeto:

**Opção A: Scripts Locais (Matplotlib)**
Para manter a organização, nossos códigos fonte locais ficam armazenados na pasta `python_scripts`. Para rodar o script principal:
```bash
uv run python_scripts/main.py
```

**Opção B: Ambiente Web (Django + Plotly)**
Temos um servidor web configurado para hospedar as aplicações em formato de site. Para iniciar o servidor localmente, entre na pasta do Django e execute o servidor:
```bash
cd python_django
uv run python manage.py runserver
```
**Opção C: Documentação e Notas de Estudo (`notes/`)**
Para guias teóricos, aprofundamento conceitual e orientações passo a passo de como estudar e debugar os códigos:
- Consulte a pasta [`notes/`](notes/): contém anotações estruturadas e documentações didáticas dos tópicos vistos em aula.
- Exemplo: [`notes/regressao.md`](notes/regressao.md) traz a dissecação completa dos scripts de regressão linear (do modelo geométrico básico até o Gradiente Descendente e Scikit-Learn).

## 📁 Estrutura do Repositório

```text
projeto-am/
├── notes/             # Anotações teóricas, guias de estudo e roteiros de depuração
│   └── regressao.md   # Guia detalhado da trilha de Regressão Linear (01 a 07)
├── python_django/     # Servidor web Django e aplicação interativa com Plotly
├── python_scripts/    # Scripts Python para experimentos de AM (01 a 07)
├── pyproject.toml     # Gerenciamento de dependências via uv
└── README.md          # Documentação principal do projeto
```

## 📝 Diário de Bordo e Avanços

Esta seção será atualizada progressivamente com os avanços do projeto.

### 05/09/2026 - Criação da Pasta `notes/` e Guia de Regressão Linear
- **Criação da pasta `notes/`**: Estruturada para centralizar o material de apoio, anotações de aula, fórmulas e guias conceituais da disciplina.
- **`notes/regressao.md`**: Criado guia completo cobrindo do `regressao_01.py` ao `regressao_07.py`. Detalha o aprendizado passo a passo, conceitos matemáticos (MSE, OLS, Gradiente Descendente), sugestões de experimentos e técnicas de depuração (*debugging* com `breakpoint()` e IDE).


### 29/08/2026 - Configuração de Bibliotecas e Ambiente Web
Para seguir com a ementa da disciplina, adicionamos suporte a visualização de dados e criamos a estrutura para rodar a aplicação em um navegador web.

**Bibliotecas adicionadas (em ordem de inclusão):**
- **`django`** e **`plotly`**: Instalados no setup inicial para o ambiente web interativo.
- **`numpy`**: Adicionada para permitir a geração e manipulação eficiente de matrizes numéricas, como a criação do vetor de "área do imóvel".
- **`matplotlib`**: Adicionada para nos permitir traçar gráficos 2D nativos (como no `regressao_01.py`).

**Como instalamos com o `uv` (Boas Práticas):**
Em vez de instalarmos os pacotes soltos, utilizamos o gerenciador do projeto:
```bash
uv add django plotly numpy matplotlib
```
Isso garante que todas as bibliotecas e suas versões exatas fiquem salvas no arquivo `pyproject.toml` e no `uv.lock`, dispensando o uso de `requirements.txt`.

**Criação da estrutura Web:**
Em seguida, utilizamos o `uv` para invocar o Django e criar a pasta raiz do servidor e o módulo principal da disciplina (que chamamos de `am`):
```bash
uv run django-admin startproject python_django
cd python_django
uv run python manage.py startapp am
```

### 29/08/2026 - Scripts Iniciais (Regressão Linear)
- **`python_scripts/regressao_01.py`**: Criado o primeiro script isolado (sem dependência web) utilizando `numpy` e `matplotlib`. O script plota um modelo hipotético (simples equação de reta da regressão) para demonstrar a relação linear entre o tamanho de um imóvel e seu preço.

### 29/08/2026 - Setup Inicial
- Configuração do repositório
- Criação do ambiente Python e pasta `python_scripts` na raiz.
- Inicialização do `README.md`
