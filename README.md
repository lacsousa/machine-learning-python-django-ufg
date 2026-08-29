# Projeto de AM (Aprendizado de Máquina)

Este repositório contém o projeto desenvolvido para a disciplina de AM (Aprendizado de Máquina) da Pós-Graduação da UFG.

## 🚀 Como começar (Instalação e Execução)

Este projeto utiliza o gerenciador de pacotes e ambientes Python [**`uv`**](https://github.com/astral-sh/uv), que é extremamente rápido e eficiente.

### 1. Como baixar (Clonar o repositório)

Primeiro, faça o clone do repositório para a sua máquina local e acesse o diretório do projeto:

```bash
git clone <URL_DO_SEU_REPOSITORIO>
cd projeto
```
*(Substitua `<URL_DO_SEU_REPOSITORIO>` pelo link real do GitHub/GitLab quando houver)*

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
*(Após rodar o comando, abra o link http://127.0.0.1:8000/ no seu navegador)*

## 📝 Diário de Bordo e Avanços

Esta seção será atualizada progressivamente com os avanços do projeto.

### 29/08/2026 - Configuração de Bibliotecas e Ambiente Web
Para seguir com a ementa da disciplina, adicionamos suporte a visualização de dados e criamos a estrutura para rodar a aplicação em um navegador web.

**Bibliotecas adicionadas:**
- **`matplotlib`**: Utilizada para gerar gráficos estáticos através dos scripts locais em Python.
- **`django`**: Framework web robusto utilizado para criar o sistema web/site.
- **`plotly`**: Biblioteca para a construção de gráficos dinâmicos e interativos para a web.

**Como instalamos com o `uv` (Boas Práticas):**
Em vez de instalarmos os pacotes soltos (usando `pip install ...`), utilizamos o gerenciador do projeto:
```bash
uv add matplotlib django plotly
```
Isso garante que todas as bibliotecas e suas versões exatas fiquem salvas no arquivo `pyproject.toml`, facilitando a replicação do projeto.

**Criação da estrutura Web:**
Em seguida, utilizamos o `uv` para invocar o Django e criar a pasta raiz do servidor e o módulo principal da disciplina (que chamamos de `am`):
```bash
uv run django-admin startproject python_django
cd python_django
uv run python manage.py startapp am
```

### 29/08/2026 - Setup Inicial
- Configuração do repositório
- Criação do ambiente Python e pasta `python_scripts` na raiz.
- Inicialização do `README.md`
