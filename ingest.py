"""
📥 Ingest CLI — Script para ingerir um repositório de código

USO:
  python ingest.py C:\\Users\\alexc\\Desktop\\Matchgame

O que acontece:
  1. Carrega todos os arquivos de código do repositório
  2. Divide cada arquivo em chunks inteligentes
  3. Gera embeddings para cada chunk via OpenAI
  4. Salva tudo no ChromaDB (vector database local)

Depois de rodar, o sistema está pronto para responder perguntas!
"""

import sys
import time

# Rich é uma biblioteca para output bonito no terminal
# Cores, tabelas, barras de progresso, etc.
from rich.console import Console
from rich.table import Table
from rich.progress import track

from src.config import validate_config
from src.ingestion.loader import load_repository, get_repo_stats
from src.ingestion.chunker import chunk_document
from src.ingestion.embedder import embed_chunks
from src.retrieval.vector_store import add_chunks, get_collection_stats, clear_collection


console = Console()


def ingest(repo_path: str, fresh: bool = False) -> None:
    """
    Pipeline completo de ingestão.

    Args:
        repo_path: Caminho do repositório para ingerir
        fresh: Se True, limpa dados anteriores antes de ingerir
    """
    console.print("\n🧠 [bold cyan]Code Intelligence RAG[/bold cyan] — Ingestão\n")

    # Valida configuração
    if not validate_config():
        sys.exit(1)

    # Limpa dados anteriores se solicitado
    if fresh:
        console.print("🗑️  Limpando dados anteriores...")
        clear_collection()

    # ========================================
    # PASSO 1: Carregar arquivos
    # ========================================
    console.print(f"📂 Carregando arquivos de: [green]{repo_path}[/green]")
    start = time.time()

    documents = load_repository(repo_path)

    if not documents:
        console.print("❌ Nenhum arquivo encontrado! Verifique o caminho.")
        sys.exit(1)

    stats = get_repo_stats(documents)
    elapsed = round(time.time() - start, 2)
    console.print(f"   ✅ {stats['total_files']} arquivos carregados em {elapsed}s")

    # Mostra tabela de linguagens encontradas
    lang_table = Table(title="Linguagens Detectadas")
    lang_table.add_column("Linguagem", style="cyan")
    lang_table.add_column("Arquivos", justify="right", style="green")

    for lang, count in stats["languages"].items():
        lang_table.add_row(lang, str(count))

    console.print(lang_table)

    # ========================================
    # PASSO 2: Dividir em chunks
    # ========================================
    console.print("\n✂️  Dividindo em chunks inteligentes...")
    start = time.time()

    all_chunks = []
    for doc in track(documents, description="Chunking..."):
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

    elapsed = round(time.time() - start, 2)
    console.print(f"   ✅ {len(all_chunks)} chunks criados em {elapsed}s")

    # ========================================
    # PASSO 3: Gerar embeddings
    # ========================================
    console.print("\n🔢 Gerando embeddings via OpenAI...")
    start = time.time()

    embedded = embed_chunks(all_chunks)

    elapsed = round(time.time() - start, 2)
    console.print(f"   ✅ {len(embedded)} embeddings gerados em {elapsed}s")

    # ========================================
    # PASSO 4: Salvar no ChromaDB
    # ========================================
    console.print("\n💾 Salvando no ChromaDB...")
    start = time.time()

    chunks_list = [item["chunk"] for item in embedded]
    embeddings_list = [item["embedding"] for item in embedded]
    added = add_chunks(chunks_list, embeddings_list)

    elapsed = round(time.time() - start, 2)
    console.print(f"   ✅ {added} chunks salvos em {elapsed}s")

    # ========================================
    # RESUMO FINAL
    # ========================================
    console.print("\n" + "=" * 50)
    console.print("🎉 [bold green]Ingestão concluída com sucesso![/bold green]")

    final_stats = get_collection_stats()
    summary = Table(title="Resumo")
    summary.add_column("Métrica", style="cyan")
    summary.add_column("Valor", justify="right", style="green")
    summary.add_row("Arquivos processados", str(stats["total_files"]))
    summary.add_row("Total de linhas", str(stats["total_lines"]))
    summary.add_row("Chunks no vector store", str(final_stats["total_chunks"]))
    summary.add_row("Tamanho do código", f"{stats['total_size_kb']} KB")

    console.print(summary)
    console.print("\n💡 Agora rode: [bold]streamlit run app.py[/bold] para a interface!")
    console.print("   Ou: [bold]uvicorn api:app --reload[/bold] para a API\n")


if __name__ == "__main__":
    # sys.argv contém os argumentos da linha de comando
    # sys.argv[0] = nome do script
    # sys.argv[1] = primeiro argumento (caminho do repo)
    if len(sys.argv) < 2:
        console.print("❌ Uso: python ingest.py <caminho-do-repositorio>")
        console.print("   Exemplo: python ingest.py C:\\Users\\alexc\\Desktop\\Matchgame")
        sys.exit(1)

    repo_path = sys.argv[1]
    fresh = "--fresh" in sys.argv  # --fresh limpa dados anteriores

    ingest(repo_path, fresh=fresh)
