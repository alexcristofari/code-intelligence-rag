"""
📂 File Loader — Carrega arquivos de código de um repositório

Este é o PRIMEIRO passo do pipeline RAG:
  Repositório → [LOADER] → Lista de documentos brutos

O loader percorre todos os arquivos de um diretório, filtra os
relevantes (código-fonte, configs, docs) e retorna uma lista
de "documentos" com o conteúdo e metadados de cada arquivo.

CONCEITO IMPORTANTE - "Document":
  No mundo de RAG, um "Document" não é só texto. É texto + metadados.
  Os metadados (filepath, linguagem, etc.) são cruciais para que o
  sistema saiba DE ONDE veio cada informação e possa citar as fontes.
"""

from dataclasses import dataclass, field
from pathlib import Path

from src.config import SUPPORTED_EXTENSIONS, IGNORED_DIRS


@dataclass
class Document:
    """
    Representa um arquivo de código carregado.

    Usamos @dataclass (Python 3.7+) ao invés de um dict porque:
      1. Type hints — IDE sabe exatamente os campos
      2. Autocompletar funciona
      3. Mais legível que doc["content"]
      4. Gera __init__, __repr__, __eq__ automaticamente

    Atributos:
        content: O conteúdo completo do arquivo como string
        filepath: Caminho relativo do arquivo (ex: "src/auth/login.ts")
        language: Linguagem de programação detectada pela extensão
        extension: Extensão do arquivo (ex: ".ts", ".py")
    """
    content: str
    filepath: str
    language: str
    extension: str
    metadata: dict = field(default_factory=dict)


# Mapeamento de extensão → linguagem
# Usado para dar contexto ao LLM sobre qual linguagem é o código
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".js": "javascript",
    ".jsx": "javascript (react)",
    ".ts": "typescript",
    ".tsx": "typescript (react)",
    ".py": "python",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".prisma": "prisma (orm)",
    ".sql": "sql",
    ".md": "markdown",
    ".dockerfile": "dockerfile",
}


def load_repository(repo_path: str) -> list[Document]:
    """
    Carrega todos os arquivos de código de um repositório.

    Como funciona:
      1. Converte o caminho para Path (objeto Python para manipular caminhos)
      2. Usa rglob("*") para percorrer TODOS os arquivos recursivamente
      3. Filtra: só pega extensões suportadas, ignora dirs como node_modules
      4. Lê cada arquivo e cria um Document com conteúdo + metadados

    Args:
        repo_path: Caminho absoluto ou relativo para o repositório

    Returns:
        Lista de Document com todos os arquivos carregados

    Raises:
        FileNotFoundError: Se o caminho não existir
        ValueError: Se o caminho não for um diretório
    """
    path = Path(repo_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Caminho não encontrado: {path}")
    if not path.is_dir():
        raise ValueError(f"O caminho não é um diretório: {path}")

    documents: list[Document] = []

    # rglob("*") = recursive glob — percorre TODAS as subpastas
    # É como o comando `find . -type f` no Linux
    for file_path in sorted(path.rglob("*")):
        # Pula se não for arquivo (pode ser diretório)
        if not file_path.is_file():
            continue

        # Pula se a extensão não é suportada
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        # Pula arquivos ignorados específicos (como lockfiles)
        from src.config import IGNORED_FILES
        if file_path.name in IGNORED_FILES:
            continue

        # Pula se está dentro de um diretório ignorado
        # Exemplo: node_modules/express/index.js → pula
        if _is_in_ignored_dir(file_path, path):
            continue

        # Tenta ler o arquivo como texto (UTF-8)
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except (PermissionError, OSError) as e:
            print(f"  ⚠️  Não consegui ler: {file_path} ({e})")
            continue

        # Pula arquivos vazios ou muito pequenos (< 10 chars)
        if len(content.strip()) < 10:
            continue

        # Calcula o caminho relativo (mais legível que o absoluto)
        # Exemplo: "apps/backend/src/auth/login.ts" ao invés do caminho completo
        relative_path = str(file_path.relative_to(path)).replace("\\", "/")

        # Detecta a linguagem pela extensão
        language = EXTENSION_TO_LANGUAGE.get(file_path.suffix.lower(), "unknown")

        doc = Document(
            content=content,
            filepath=relative_path,
            language=language,
            extension=file_path.suffix.lower(),
            metadata={
                "num_lines": content.count("\n") + 1,
                "size_bytes": len(content.encode("utf-8")),
            },
        )
        documents.append(doc)

    return documents


def _is_in_ignored_dir(file_path: Path, repo_root: Path) -> bool:
    """
    Verifica se um arquivo está dentro de um diretório ignorado.

    Percorre cada "parte" do caminho relativo e checa contra IGNORED_DIRS.
    Exemplo:
      file_path = "repo/node_modules/express/index.js"
      parts = ["node_modules", "express", "index.js"]
      "node_modules" está em IGNORED_DIRS → retorna True
    """
    try:
        relative = file_path.relative_to(repo_root)
        return any(part in IGNORED_DIRS for part in relative.parts)
    except ValueError:
        return False


def get_repo_stats(documents: list[Document]) -> dict:
    """Retorna estatísticas sobre os documentos carregados."""
    languages: dict[str, int] = {}
    total_lines = 0
    total_bytes = 0

    for doc in documents:
        languages[doc.language] = languages.get(doc.language, 0) + 1
        total_lines += doc.metadata.get("num_lines", 0)
        total_bytes += doc.metadata.get("size_bytes", 0)

    return {
        "total_files": len(documents),
        "total_lines": total_lines,
        "total_size_kb": round(total_bytes / 1024, 1),
        "languages": dict(sorted(languages.items(), key=lambda x: -x[1])),
    }
