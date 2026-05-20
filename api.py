"""
🌐 API — Endpoints REST com FastAPI

FastAPI é o framework #1 para APIs em Python. Ele gera
documentação automática (Swagger) e é extremamente rápido.

Depois de rodar, acesse:
  http://localhost:8000/docs  → documentação interativa (Swagger UI)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import validate_config
from src.generation.generator import generate_answer
from src.retrieval.vector_store import get_collection_stats


# ========================================
# Pydantic Models — Validação de dados
# ========================================
# Pydantic garante que os dados recebidos estão no formato correto.
# Se alguém enviar um request sem o campo "query", o FastAPI
# retorna automaticamente um erro 422 explicando o que está faltando.

class QueryRequest(BaseModel):
    """Corpo do request para fazer uma pergunta."""
    query: str
    top_k: int = 5


class SourceInfo(BaseModel):
    """Informação sobre um arquivo fonte."""
    filepath: str
    score: float
    start_line: int | None = None
    end_line: int | None = None
    language: str | None = None


class QueryResponse(BaseModel):
    """Resposta completa de uma pergunta."""
    answer: str
    sources: list[SourceInfo]
    model: str
    tokens_used: int


# ========================================
# App FastAPI
# ========================================
app = FastAPI(
    title="Code Intelligence RAG",
    description="API que responde perguntas sobre repositórios de código usando RAG",
    version="0.1.0",
)

# CORS — permite que o frontend (se houver) acesse a API
# Em produção, você restringiria os origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Health check — verifica se a API está rodando."""
    return {
        "status": "online",
        "service": "Code Intelligence RAG",
        "docs": "/docs",
    }


@app.get("/stats")
def stats():
    """Retorna estatísticas do vector store."""
    return get_collection_stats()


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Faz uma pergunta sobre o código ingerido.

    Exemplo de request:
    ```json
    {
        "query": "Como funciona a autenticação?",
        "top_k": 5
    }
    ```
    """
    if not validate_config():
        raise HTTPException(status_code=500, detail="OpenAI API key não configurada")

    collection_stats = get_collection_stats()
    if collection_stats["total_chunks"] == 0:
        raise HTTPException(
            status_code=400,
            detail="Nenhum código ingerido. Rode: python ingest.py <repo-path>",
        )

    result = generate_answer(query=request.query, top_k=request.top_k)

    sources = [
        SourceInfo(
            filepath=s.filepath,
            score=s.score,
            start_line=s.metadata.get("start_line"),
            end_line=s.metadata.get("end_line"),
            language=s.metadata.get("language"),
        )
        for s in result.sources
    ]

    return QueryResponse(
        answer=result.answer,
        sources=sources,
        model=result.model,
        tokens_used=result.tokens_used,
    )


# Para rodar: uvicorn api:app --reload
