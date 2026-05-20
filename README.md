# Code Intelligence RAG

Um sistema corporativo de Retrieval-Augmented Generation (RAG) desenvolvido para analisar repositorios de codigo complexos e responder a perguntas sobre arquitetura, regras de negocio e fluxos de dados de forma automatizada.

Este projeto foi construido nativamente para atuar como o "Cerebro Inteligente" do **MatchGame** (https://alexcristofari.com), um sistema de compatibilidade baseado em dados comportamentais. O objetivo e demonstrar capacidade avancada em Engenharia de IA, criando uma ferramenta que possa auxiliar desenvolvedores a navegar em codebases extensas.

## O que o sistema faz?

O Code Intelligence ingere o repositorio do MatchGame (ou qualquer outro repositorio), convertendo todo o codigo-fonte em embeddings e salvando em um banco de dados vetorial. Em seguida, ele utiliza a API da OpenAI para gerar respostas baseadas unicamente no contexto matematico (vetorial) recuperado do proprio codigo.

Exemplo de uso em producao:
- **Pergunta:** "Como funciona a autenticacao neste projeto?"
- **Resposta:** O modelo le os vetores mais similares a "autenticacao", encontra os arquivos `auth.middleware.ts` e `auth.routes.ts`, interpreta o codigo TypeScript, e responde detalhando o uso de JWT e Bcrypt, alem de citar as linhas exatas onde as funcoes ocorrem no repositorio.

## Metricas e Desempenho no MatchGame

Durante a execucao primaria no repositorio base (MatchGame), o sistema demonstrou os seguintes dados de ingestao e processamento em um computador padrao (sem GPU dedicada para processamento local intenso):

- **Arquivos processados:** 146 (TypeScript, JSON, Markdown, Prisma, etc)
- **Total de linhas de codigo ingeridas:** 46.815 linhas
- **Chunks (trechos estruturados) indexados:** 2.951 chunks semanticos
- **Tamanho processado (Raw):** 2.7 MB de codigo puro
- **Tempo de ingestao:** ~55 segundos para carregar, fatiar e gerar 2.951 embeddings.
- **Tamanho do espaco dimensional (Embeddings):** 1536 dimensoes via OpenAI text-embedding-3-small.

## Arquitetura do Sistema

O pipeline RAG foi dividido em 3 modulos independentes e escalaveis:

1. **Ingestion Pipeline (ETL de Codigo):** 
   - `Loader`: Percorre o sistema de arquivos ignorando diretorios irrelevantes (`node_modules`, `.git`, lockfiles).
   - `Chunker`: Quebra o codigo em pedacos logicos (ex: por funcoes ou classes via Regex) e aplica _overlap_ para nao perder o contexto.
   - `Embedder`: Converte o texto fatiado em vetores numericos densos.

2. **Storage and Retrieval (Busca Vetorial):** 
   - Utiliza `Numpy` para calculos de Cosine Similarity, garantindo altissima velocidade de busca matricial sem problemas de dependencia C++ (DLL) comuns no Windows. Em ambientes de producao Linux, pode facilmente ser substituido por ChromaDB ou Pinecone.
   - Retorna os `Top K` chunks mais relevantes para a pergunta do usuario.

3. **Generation Pipeline (LLM):** 
   - Um modelo `gpt-4o-mini` recebe um "System Prompt" robusto junto com o contexto vetorial recuperado para fundamentar a resposta e evitar alucinacoes, sempre referenciando o arquivo de origem.

## Tecnologias e Stack

- **Linguagem:** Python 3.14 (Backend / Data Pipeline)
- **LLM / Embeddings:** OpenAI API (`gpt-4o-mini` e `text-embedding-3-small`)
- **Algebra Linear / Vector DB:** Numpy (Arrays multidimensionais e busca cosine similarity)
- **API e Rotas:** FastAPI, Pydantic (Validacao de Esquemas)
- **Interface e Deploy:** Streamlit (UI interativa para conversacao RAG)
- **Controle de Dependencias:** Pyproject.toml / uv

## Quick Start (Execucao Local)

1. Instale as dependencias do projeto:
```bash
pip install -e .
```

2. Configure a variavel de ambiente em um arquivo `.env` na raiz:
```env
OPENAI_API_KEY=sua-chave-aqui
```

3. Inicie o processo de Ingestao em um repositorio alvo:
```bash
python ingest.py C:\caminho\do\seu\repo
```

4. Suba a aplicacao Streamlit para realizar perguntas no chat:
```bash
streamlit run app.py
```

## Melhorias Futuras (Roadmap de Escala)

- Implementacao de Busca Hibrida (BM25 + Semantic Search) para encontrar chaves de variaveis especificas.
- Adicao de modelos Cross-Encoders para Re-ranking dos documentos antes da injecao no prompt do LLM.
- Conversao de chunking estatico para Graph RAG, utilizando bancos baseados em grafos para mapear dependencias de import/export entre os arquivos Typescript do MatchGame.

---
Desenvolvido por Alexsander Cristofari como modulo avancado de Inteligencia Artificial associado ao projeto MatchGame.
