# 🧠 Code Intelligence RAG

> Um sistema RAG (Retrieval-Augmented Generation) que ingere repositórios de código e responde perguntas inteligentes sobre arquitetura, funcionalidades e fluxos do projeto.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green?style=flat-square&logo=fastapi)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange?style=flat-square&logo=openai)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-purple?style=flat-square)

## 🎯 O que é?

Aponte para qualquer repositório de código e faça perguntas em linguagem natural:

```
Pergunta: "Como funciona a autenticação neste projeto?"

Resposta: "A autenticação utiliza JWT tokens implementados em
`auth.middleware.ts` (linhas 12-45). O fluxo é:
1. Login via POST /auth/login
2. Validação com bcrypt em validatePassword()
3. Token JWT gerado com expiração de 24h
📁 Fontes: auth.middleware.ts, auth.routes.ts, prisma/schema.prisma"
```

## 🏗️ Arquitetura

```
  Repositório  →  Loader  →  Chunker  →  Embedder  →  ChromaDB
                                                          │
  Pergunta  →  Embedding  →  Busca Semântica  ────────────┘
                                    │
                             Top-K Chunks + Pergunta
                                    │
                               GPT-4o-mini
                                    │
                            Resposta + Fontes
```

## 🚀 Quick Start

### 1. Instalar dependências

```bash
pip install -e .
```

### 2. Configurar API key

```bash
copy .env.example .env
# Edite .env e coloque sua OPENAI_API_KEY
```

### 3. Ingerir um repositório

```bash
python ingest.py C:\caminho\do\seu\repo
```

### 4. Fazer perguntas

**Via Streamlit (interface visual):**
```bash
streamlit run app.py
```

**Via API (FastAPI):**
```bash
uvicorn api:app --reload
# Acesse: http://localhost:8000/docs
```

## 🛠️ Stack Tecnológico

| Componente | Tecnologia | Função |
|---|---|---|
| Linguagem | Python 3.11+ | Core do projeto |
| Embeddings | OpenAI text-embedding-3-small | Converte texto em vetores |
| LLM | OpenAI GPT-4o-mini | Gera respostas |
| Vector DB | ChromaDB | Armazena e busca embeddings |
| API | FastAPI | Endpoints REST |
| UI | Streamlit | Interface de chat |

## 📁 Estrutura

```
CodeIntelligence/
├── src/
│   ├── ingestion/        # Pipeline de ingestão
│   │   ├── loader.py     # Carrega arquivos do repo
│   │   ├── chunker.py    # Divide código em chunks
│   │   └── embedder.py   # Gera embeddings
│   ├── retrieval/        # Pipeline de busca
│   │   ├── vector_store.py  # ChromaDB operations
│   │   └── retriever.py     # Busca semântica
│   ├── generation/       # Pipeline de geração
│   │   └── generator.py  # Prompt + LLM
│   └── config.py         # Configurações
├── ingest.py             # CLI de ingestão
├── api.py                # FastAPI server
├── app.py                # Streamlit UI
└── pyproject.toml        # Dependências
```

## 📝 Licença

MIT

---

Feito com ❤️ como projeto de portfólio para AI Engineering
