"""
🔢 Embedder — Transforma texto em vetores numéricos

TERCEIRO passo do pipeline:
  Chunks de código → [EMBEDDER] → Vetores de 1536 dimensões

O QUE SÃO EMBEDDINGS?
  Embedding = representação numérica do SIGNIFICADO de um texto.
  
  Cada texto é convertido em um vetor (lista de números) de 1536 posições.
  Textos com significados PARECIDOS terão vetores PRÓXIMOS no espaço.

  Exemplo simplificado (na realidade são 1536 dimensões):
    "função de login"     → [0.8, 0.2, 0.9, ...]
    "autenticação"        → [0.7, 0.3, 0.85, ...]  ← PRÓXIMO!
    "CSS do botão"        → [0.1, 0.9, 0.1, ...]   ← DISTANTE!

  Isso é o que permite a "busca semântica": buscar por SIGNIFICADO,
  não por palavras exatas. Se você perguntar "como o login funciona?",
  o sistema encontra código sobre autenticação mesmo que a palavra
  "login" não apareça naquele trecho.

POR QUE USAMOS A API DA OPENAI?
  Treinar um modelo de embedding do zero requer milhões de exemplos
  e GPUs caras. A OpenAI já fez isso e oferece via API por ~$0.02/1M tokens.
  Para nosso projeto, gastaremos centavos.
"""

from openai import OpenAI

from src.config import OPENAI_API_KEY, EMBEDDING_MODEL
from src.ingestion.chunker import Chunk


# Inicializa o client da OpenAI
# O client gerencia conexões, retry, rate limits automaticamente
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """
    Retorna o client OpenAI (singleton pattern).

    CONCEITO - SINGLETON:
      Queremos apenas UMA instância do client, não criar um novo
      a cada chamada. Isso é mais eficiente e evita problemas
      de conexão.
    """
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Gera embeddings para uma lista de textos.

    Envia os textos em BATCH (todos de uma vez) para a API.
    Isso é muito mais eficiente do que enviar um por um:
      - 1 request com 100 textos = ~0.5s
      - 100 requests com 1 texto cada = ~30s

    Args:
        texts: Lista de strings para converter em embeddings

    Returns:
        Lista de vetores (cada vetor = lista de 1536 floats)
    """
    client = _get_client()

    # Prevenir falha na API: Truncar textos absurdamente grandes
    # A OpenAI tem um limite estrito de ~8192 tokens.
    # Assumindo o pior caso (1 token = 1 char), 8000 caracteres é um limite 100% seguro.
    safe_texts = [text[:8000] for text in texts]

    # A API aceita até ~8000 tokens por texto e múltiplos textos por request
    response = client.embeddings.create(
        input=safe_texts,
        model=EMBEDDING_MODEL,
    )

    # response.data é uma lista de objetos com .embedding
    # Ordenamos pelo index para garantir que a ordem seja preservada
    embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    return embeddings


def embed_chunks(chunks: list[Chunk], batch_size: int = 50) -> list[dict]:
    """
    Gera embeddings para uma lista de Chunks em batches.

    POR QUE BATCH?
      A API tem limites de tokens por request (~8191 tokens).
      Dividindo em batches de 50, evitamos ultrapassar o limite
      e também podemos mostrar progresso para o usuário.

    Args:
        chunks: Lista de Chunks do chunker
        batch_size: Quantos chunks processar por request da API

    Returns:
        Lista de dicts com {chunk, embedding} prontos para o vector store
    """
    results: list[dict] = []

    # Processa em batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [chunk.content for chunk in batch]

        # Gera embeddings para este batch
        embeddings = generate_embeddings(texts)

        # Combina cada chunk com seu embedding
        for chunk, embedding in zip(batch, embeddings):
            results.append({
                "chunk": chunk,
                "embedding": embedding,
            })

    return results


def embed_query(query: str) -> list[float]:
    """
    Gera embedding para uma pergunta do usuário.

    Quando o usuário faz uma pergunta, precisamos converter ela
    para o MESMO espaço vetorial dos chunks. Assim podemos comparar
    a distância entre a pergunta e cada chunk.

    Args:
        query: Pergunta em linguagem natural

    Returns:
        Vetor de 1536 floats representando a pergunta
    """
    embeddings = generate_embeddings([query])
    return embeddings[0]
