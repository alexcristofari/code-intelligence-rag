"""
🔍 Retriever — Busca semântica por trechos de código relevantes

Este módulo conecta o embedding da pergunta ao vector store,
encontra os chunks mais relevantes e os formata para uso pelo gerador.
"""

from dataclasses import dataclass

from src.ingestion.embedder import embed_query
from src.retrieval.vector_store import query_similar
from src.config import TOP_K


@dataclass
class RetrievalResult:
    """
    Resultado de uma busca semântica.

    Attributes:
        content: O texto do chunk encontrado
        filepath: De qual arquivo veio
        score: Score de similaridade (0 = idêntico, 1 = irrelevante)
        metadata: Informações extras (linha, linguagem, etc.)
    """
    content: str
    filepath: str
    score: float
    metadata: dict


def retrieve(query: str, top_k: int = TOP_K) -> list[RetrievalResult]:
    """
    Busca os trechos de código mais relevantes para uma pergunta.

    O FLUXO COMPLETO:
      1. "Como funciona a autenticação?" (texto)
      2. → embed_query() → [0.7, 0.3, 0.85, ...] (vetor)
      3. → query_similar() → top-5 chunks mais próximos
      4. → RetrievalResult com score e metadados

    SCORE DE SIMILARIDADE:
      Usamos cosine distance (distância por cosseno):
        - 0.0 = vetores idênticos (match perfeito)
        - 0.5 = alguma relação
        - 1.0 = completamente diferentes
      
      Na prática, scores < 0.3 são muito bons.
      Scores > 0.7 provavelmente não são relevantes.

    Args:
        query: Pergunta em linguagem natural
        top_k: Quantos resultados retornar

    Returns:
        Lista de RetrievalResult ordenados por relevância
    """
    # 1. Converte a pergunta em vetor
    query_embedding = embed_query(query)

    # 2. Busca os chunks mais similares
    raw_results = query_similar(query_embedding, n_results=top_k)

    # 3. Formata os resultados
    results: list[RetrievalResult] = []

    # ChromaDB retorna listas dentro de listas (por query)
    # Como sempre fazemos 1 query, pegamos o índice [0]
    if raw_results["ids"] and raw_results["ids"][0]:
        documents = raw_results["documents"][0]
        metadatas = raw_results["metadatas"][0]
        distances = raw_results["distances"][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            results.append(RetrievalResult(
                content=doc,
                filepath=meta.get("filepath", "unknown"),
                score=round(dist, 4),
                metadata=meta,
            ))

    return results


def format_context(results: list[RetrievalResult]) -> str:
    """
    Formata os resultados de retrieval como contexto para o LLM.

    Este é o texto que será inserido no prompt do LLM, entre a
    instrução do sistema e a pergunta do usuário.

    Um bom formato de contexto:
      1. Identifica claramente cada trecho (filepath, linhas)
      2. Separa trechos visualmente
      3. Inclui a linguagem para ajudar o LLM a interpretar
    """
    if not results:
        return "Nenhum trecho de código relevante encontrado."

    context_parts: list[str] = []

    for i, result in enumerate(results, 1):
        start = result.metadata.get("start_line", "?")
        end = result.metadata.get("end_line", "?")
        lang = result.metadata.get("language", "unknown")
        score = result.score

        header = (
            f"--- Trecho {i}/{len(results)} ---\n"
            f"Arquivo: {result.filepath}\n"
            f"Linhas: {start}-{end} | Linguagem: {lang} | Relevância: {score}\n"
        )

        context_parts.append(header + result.content)

    return "\n\n".join(context_parts)
