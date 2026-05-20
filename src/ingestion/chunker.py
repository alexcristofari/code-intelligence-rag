"""
✂️ Code Chunker — Divide código em pedaços inteligentes

SEGUNDO passo do pipeline:
  Documentos brutos → [CHUNKER] → Chunks menores e pesquisáveis

POR QUE PRECISAMOS DE CHUNKING?
  LLMs têm uma "context window" limitada (janela de contexto).
  Não dá pra jogar um repositório inteiro de 50 mil linhas no prompt.
  Precisamos dividir em pedaços menores, para que na hora da busca
  possamos encontrar EXATAMENTE o trecho relevante.

  Analogia: é como um índice de livro. Ao invés de ler o livro inteiro
  para achar a resposta, você vai direto ao capítulo certo.

ESTRATÉGIA DE CHUNKING:
  Chunking "burro" = dividir a cada 500 caracteres (corta no meio de funções)
  Chunking "inteligente" = dividir por blocos lógicos (funções, classes)

  Nós usamos chunking inteligente: detectamos limites de funções/classes
  e dividimos respeitando a estrutura do código.
"""

from dataclasses import dataclass
import re

from src.ingestion.loader import Document
from src.config import MAX_CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    """
    Um pedaço de código pronto para ser convertido em embedding.

    Cada chunk contém:
      - O texto do código
      - Metadados que identificam de onde veio
      - Um ID único para rastreamento no vector store
    """
    chunk_id: str
    content: str
    filepath: str
    language: str
    start_line: int
    end_line: int
    chunk_type: str  # "function", "class", "block", "file_header"
    metadata: dict


def chunk_document(document: Document) -> list[Chunk]:
    """
    Divide um documento de código em chunks inteligentes.

    A estratégia é:
      1. Primeiro, extrair o "header" do arquivo (imports, configs no topo)
      2. Depois, tentar dividir por blocos lógicos (funções, classes)
      3. Se o arquivo for pequeno o suficiente, manter como chunk único
      4. Se um bloco for muito grande, dividir por tamanho com overlap

    Args:
        document: Um Document carregado pelo loader

    Returns:
        Lista de Chunks prontos para embedding
    """
    content = document.content
    lines = content.split("\n")

    # Se o arquivo é pequeno, não precisa dividir
    if len(content) <= MAX_CHUNK_SIZE:
        return [_create_chunk(
            document=document,
            content=content,
            start_line=1,
            end_line=len(lines),
            chunk_type="full_file",
            chunk_index=0,
        )]

    chunks: list[Chunk] = []
    chunk_index = 0

    # Tenta dividir por blocos lógicos (funções, classes, exports)
    blocks = _split_into_logical_blocks(content, document.language)

    if blocks:
        for block in blocks:
            block_content = block["content"]

            # Se o bloco é grande demais, subdivide por tamanho
            if len(block_content) > MAX_CHUNK_SIZE:
                sub_chunks = _split_by_size(block_content, block["start_line"])
                for sub in sub_chunks:
                    # Adiciona contexto: filepath + linguagem no topo de cada chunk
                    # Isso ajuda o LLM a entender o contexto
                    enriched = _enrich_chunk_content(
                        sub["content"], document.filepath, document.language
                    )
                    chunks.append(_create_chunk(
                        document=document,
                        content=enriched,
                        start_line=sub["start_line"],
                        end_line=sub["end_line"],
                        chunk_type=block["type"],
                        chunk_index=chunk_index,
                    ))
                    chunk_index += 1
            else:
                enriched = _enrich_chunk_content(
                    block_content, document.filepath, document.language
                )
                chunks.append(_create_chunk(
                    document=document,
                    content=enriched,
                    start_line=block["start_line"],
                    end_line=block["end_line"],
                    chunk_type=block["type"],
                    chunk_index=chunk_index,
                ))
                chunk_index += 1
    else:
        # Fallback: divide por tamanho com overlap
        size_chunks = _split_by_size(content, start_line=1)
        for sub in size_chunks:
            enriched = _enrich_chunk_content(
                sub["content"], document.filepath, document.language
            )
            chunks.append(_create_chunk(
                document=document,
                content=enriched,
                start_line=sub["start_line"],
                end_line=sub["end_line"],
                chunk_type="block",
                chunk_index=chunk_index,
            ))
            chunk_index += 1

    return chunks


def _split_into_logical_blocks(content: str, language: str) -> list[dict]:
    """
    Detecta limites de funções/classes/exports no código.

    Usa regex para encontrar padrões comuns de definição de blocos.
    Não é perfeito (um parser AST seria melhor), mas funciona bem
    para a maioria dos casos e é muito mais simples de implementar.

    REGEX CRASH COURSE:
      ^           = início da linha
      (?:...|...) = grupo não-capturante com alternativas
      \\s*        = zero ou mais espaços
      .+          = um ou mais caracteres quaisquer
    """
    lines = content.split("\n")

    # Padrões que indicam início de um novo bloco lógico
    # Diferentes para cada linguagem
    if language in ("typescript", "typescript (react)", "javascript", "javascript (react)"):
        block_pattern = re.compile(
            r"^(?:"
            r"(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+"  # function declarations
            r"|(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?\("  # arrow functions
            r"|(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=])\s*=>"  # arrow fn
            r"|(?:export\s+)?class\s+"  # class declarations
            r"|(?:export\s+)?interface\s+"  # interface declarations
            r"|(?:export\s+)?type\s+\w+\s*="  # type aliases
            r"|(?:export\s+)?enum\s+"  # enums
            r"|router\.\w+\("  # express routes
            r"|app\.\w+\("  # express app routes
            r")",
            re.MULTILINE,
        )
    elif language == "python":
        block_pattern = re.compile(
            r"^(?:"
            r"(?:async\s+)?def\s+"  # function definitions
            r"|class\s+"  # class definitions
            r"|@\w+"  # decorators (start of a block)
            r")",
            re.MULTILINE,
        )
    else:
        # Para outras linguagens, usa divisão por tamanho
        return []

    # Encontra as posições de início de cada bloco
    block_starts: list[int] = []
    for i, line in enumerate(lines):
        if block_pattern.match(line.strip()):
            block_starts.append(i)

    if not block_starts:
        return []

    blocks: list[dict] = []

    # Se existe código antes do primeiro bloco (imports, configs)
    if block_starts[0] > 0:
        header_content = "\n".join(lines[: block_starts[0]])
        if header_content.strip():
            blocks.append({
                "content": header_content,
                "start_line": 1,
                "end_line": block_starts[0],
                "type": "file_header",
            })

    # Cria um bloco para cada função/classe encontrada
    for i, start in enumerate(block_starts):
        # O bloco vai do início até o próximo bloco (ou fim do arquivo)
        end = block_starts[i + 1] if i + 1 < len(block_starts) else len(lines)
        block_content = "\n".join(lines[start:end])

        if block_content.strip():
            blocks.append({
                "content": block_content,
                "start_line": start + 1,  # +1 porque lines é 0-indexed
                "end_line": end,
                "type": _detect_block_type(lines[start].strip()),
            })

    return blocks


def _detect_block_type(first_line: str) -> str:
    """Detecta o tipo de bloco pela primeira linha."""
    if "class " in first_line:
        return "class"
    if "interface " in first_line:
        return "interface"
    if "function " in first_line or "=>" in first_line or "def " in first_line:
        return "function"
    if "type " in first_line:
        return "type"
    if "enum " in first_line:
        return "enum"
    if "router." in first_line or "app." in first_line:
        return "route"
    return "block"


def _split_by_size(content: str, start_line: int) -> list[dict]:
    """
    Divide texto por tamanho com overlap.

    CONCEITO - OVERLAP:
      Imagine que temos o texto: "A função login() chama validatePassword()"
      Se cortarmos no meio: ["A função login() ch", "ama validatePassword()"]
      O chunk 1 não sabe que login chama validatePassword.
      O chunk 2 não sabe que estamos falando de login.

      Com overlap de 50 chars, repetimos um pedaço:
      ["A função login() chama vali", "chama validatePassword()"]
      Agora ambos os chunks têm a informação da conexão!
    """
    lines = content.split("\n")
    chunks: list[dict] = []
    current_chunk_lines: list[str] = []
    current_size = 0

    for i, line in enumerate(lines):
        current_chunk_lines.append(line)
        current_size += len(line) + 1  # +1 pelo \n

        if current_size >= MAX_CHUNK_SIZE:
            chunk_content = "\n".join(current_chunk_lines)
            chunks.append({
                "content": chunk_content,
                "start_line": start_line + i - len(current_chunk_lines) + 1,
                "end_line": start_line + i,
            })

            # Overlap: mantém as últimas linhas para o próximo chunk
            overlap_chars = 0
            overlap_lines: list[str] = []
            for prev_line in reversed(current_chunk_lines):
                overlap_chars += len(prev_line) + 1
                if overlap_chars >= CHUNK_OVERLAP:
                    break
                overlap_lines.insert(0, prev_line)

            current_chunk_lines = overlap_lines
            current_size = sum(len(l) + 1 for l in current_chunk_lines)

    # Último chunk (se sobrou conteúdo)
    if current_chunk_lines:
        chunk_content = "\n".join(current_chunk_lines)
        if chunk_content.strip():
            chunks.append({
                "content": chunk_content,
                "start_line": start_line + len(lines) - len(current_chunk_lines),
                "end_line": start_line + len(lines) - 1,
            })

    return chunks


def _enrich_chunk_content(content: str, filepath: str, language: str) -> str:
    """
    Adiciona contexto ao chunk para ajudar o LLM.

    Sem contexto, o LLM recebe um trecho de código solto e não sabe
    de qual arquivo veio. Adicionando o filepath e a linguagem, ele
    pode dar respostas muito mais precisas.
    """
    header = f"// File: {filepath} | Language: {language}\n"
    return header + content


def _create_chunk(
    document: Document,
    content: str,
    start_line: int,
    end_line: int,
    chunk_type: str,
    chunk_index: int,
) -> Chunk:
    """Cria um Chunk com ID único e metadados."""
    # ID único: combina filepath + índice
    # Exemplo: "src/auth/login.ts::chunk_0"
    chunk_id = f"{document.filepath}::chunk_{chunk_index}"

    return Chunk(
        chunk_id=chunk_id,
        content=content,
        filepath=document.filepath,
        language=document.language,
        start_line=start_line,
        end_line=end_line,
        chunk_type=chunk_type,
        metadata={
            "filepath": document.filepath,
            "language": document.language,
            "extension": document.extension,
            "start_line": start_line,
            "end_line": end_line,
            "chunk_type": chunk_type,
            "num_chars": len(content),
        },
    )
