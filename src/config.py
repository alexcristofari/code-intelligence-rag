"""
Code Intelligence RAG — Configurações Centralizadas

Este arquivo carrega variáveis de ambiente e define constantes
usadas em todo o projeto. Centralizar configuração é uma boa
prática porque:
  1. Evita "magic strings" espalhados pelo código
  2. Facilita mudar valores sem mexer em lógica
  3. Mantém secrets (API keys) seguros no .env
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis do arquivo .env para o ambiente
# Isso permite usar os.getenv() para acessar os valores
load_dotenv()

# ==================================================
# Diretórios
# ==================================================
# Path.resolve() converte para caminho absoluto
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chromadb"))

# ==================================================
# OpenAI
# ==================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Dimensão dos embeddings do text-embedding-3-small
# Cada texto vira um vetor de 1536 números — é assim que
# o computador "entende" o significado semântico do texto
EMBEDDING_DIMENSION = 1536

# ==================================================
# Chunking (como dividimos o código em pedaços)
# ==================================================
# Tamanho máximo de cada chunk em caracteres
# ~1000 chars ≈ ~250 tokens — bom equilíbrio entre contexto e precisão
MAX_CHUNK_SIZE = 1000

# Overlap: quantos caracteres do chunk anterior repetimos no próximo
# Isso evita que informação se perca na "borda" entre chunks
CHUNK_OVERLAP = 200

# ==================================================
# Retrieval (busca)
# ==================================================
# Quantos chunks retornar por busca
# Mais chunks = mais contexto, mas mais tokens (mais caro)
TOP_K = 5

# Nome da coleção no ChromaDB
COLLECTION_NAME = "code_intelligence"

# ==================================================
# Extensões de arquivo suportadas
# ==================================================
# Só processamos estes tipos de arquivo — não faz sentido
# indexar imagens, binários, ou arquivos de lock
SUPPORTED_EXTENSIONS = {
    # JavaScript / TypeScript
    ".js", ".jsx", ".ts", ".tsx",
    # Python
    ".py",
    # Web
    ".html", ".css",
    # Config
    ".json", ".yaml", ".yml", ".toml",
    # Database
    ".prisma", ".sql",
    # Docs
    ".md",
    # Docker
    ".dockerfile",
}

# Diretórios que devemos ignorar durante a ingestão
IGNORED_DIRS = {
    "node_modules",
    ".git",
    "__pycache__",
    "dist",
    "build",
    ".next",
    ".venv",
    "venv",
    ".env",
    "coverage",
}

# Arquivos que devemos ignorar especificamente
IGNORED_FILES = {
    "pnpm-lock.yaml",
    "package-lock.json",
    "yarn.lock",
}

# ==================================================
# Validação
# ==================================================
def validate_config() -> bool:
    """Verifica se as configurações essenciais estão corretas."""
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-your-key-here":
        print("❌ OPENAI_API_KEY não configurada!")
        print("   1. Copie .env.example para .env")
        print("   2. Coloque sua API key no arquivo .env")
        return False
    return True
