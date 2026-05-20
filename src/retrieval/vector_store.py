"""
💾 Vector Store — Armazena e busca embeddings com Numpy

Substituímos o ChromaDB por uma implementação em Numpy puro 
devido a problemas de compatibilidade de DLLs no Windows com Python 3.14.
Para um projeto desse tamanho (alguns milhares de chunks), Numpy 
é extremamente rápido e não requer dependências complexas!
"""

import json
import numpy as np
from pathlib import Path

from src.config import CHROMA_PERSIST_DIR, COLLECTION_NAME
from src.ingestion.chunker import Chunk

# Garante que o diretório existe
Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

DATA_FILE = Path(CHROMA_PERSIST_DIR) / f"{COLLECTION_NAME}_data.json"
VECTORS_FILE = Path(CHROMA_PERSIST_DIR) / f"{COLLECTION_NAME}_vectors.npy"


def _load_data():
    """Carrega os dados salvos do disco."""
    if DATA_FILE.exists() and VECTORS_FILE.exists():
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        vectors = np.load(VECTORS_FILE)
        return data, vectors
    return {"ids": [], "documents": [], "metadatas": []}, np.array([])


def _save_data(data, vectors):
    """Salva os dados no disco."""
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)
    np.save(VECTORS_FILE, vectors)


def add_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> int:
    """
    Adiciona chunks e seus embeddings ao vector store.
    """
    data, vectors = _load_data()
    
    new_ids = [chunk.chunk_id for chunk in chunks]
    new_docs = [chunk.content for chunk in chunks]
    new_metas = [chunk.metadata for chunk in chunks]
    new_vecs = np.array(embeddings, dtype=np.float32)

    # Filtrar IDs que já existem (upsert simples)
    existing_ids = set(data["ids"])
    indices_to_add = [i for i, id in enumerate(new_ids) if id not in existing_ids]

    if indices_to_add:
        filtered_ids = [new_ids[i] for i in indices_to_add]
        filtered_docs = [new_docs[i] for i in indices_to_add]
        filtered_metas = [new_metas[i] for i in indices_to_add]
        filtered_vecs = new_vecs[indices_to_add]

        data["ids"].extend(filtered_ids)
        data["documents"].extend(filtered_docs)
        data["metadatas"].extend(filtered_metas)
        
        if vectors.size == 0:
            vectors = filtered_vecs
        else:
            vectors = np.vstack((vectors, filtered_vecs))

        _save_data(data, vectors)

    return len(indices_to_add)


def query_similar(
    query_embedding: list[float],
    n_results: int = 5,
    where: dict | None = None,
) -> dict:
    """
    Busca os chunks mais similares a um embedding de query usando Cosine Similarity.
    """
    data, vectors = _load_data()
    
    if vectors.size == 0:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    # Converter query para numpy array
    q_vec = np.array(query_embedding, dtype=np.float32)

    # Calcular Cosine Similarity
    # cosine_similarity = dot(A, B) / (norm(A) * norm(B))
    # Para OpenAI, embeddings já são normalizados (norm=1), mas vamos normalizar pra garantir
    q_norm = np.linalg.norm(q_vec)
    v_norms = np.linalg.norm(vectors, axis=1)
    
    # Previne divisão por zero
    v_norms[v_norms == 0] = 1e-10
    if q_norm == 0:
        q_norm = 1e-10

    similarities = np.dot(vectors, q_vec) / (v_norms * q_norm)
    
    # ChromaDB retorna 'distance' (1 - similarity)
    distances = 1.0 - similarities

    # Ordenar pelos mais próximos (menor distância)
    sorted_indices = np.argsort(distances)

    # Aplicar filtros 'where' se existirem
    filtered_indices = []
    for idx in sorted_indices:
        meta = data["metadatas"][idx]
        if where:
            match = all(meta.get(k) == v for k, v in where.items())
            if not match:
                continue
        filtered_indices.append(idx)
        if len(filtered_indices) == n_results:
            break

    # Formatar como o ChromaDB faria
    result_docs = [data["documents"][i] for i in filtered_indices]
    result_metas = [data["metadatas"][i] for i in filtered_indices]
    result_dists = [float(distances[i]) for i in filtered_indices]
    result_ids = [data["ids"][i] for i in filtered_indices]

    return {
        "ids": [result_ids],
        "documents": [result_docs],
        "metadatas": [result_metas],
        "distances": [result_dists],
    }


def get_collection_stats() -> dict:
    """Retorna estatísticas da coleção."""
    data, _ = _load_data()
    return {
        "total_chunks": len(data["ids"]),
        "collection_name": COLLECTION_NAME,
        "persist_dir": CHROMA_PERSIST_DIR,
    }


def clear_collection() -> None:
    """Limpa toda a coleção."""
    if DATA_FILE.exists():
        DATA_FILE.unlink()
    if VECTORS_FILE.exists():
        VECTORS_FILE.unlink()
