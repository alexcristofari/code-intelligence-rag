"""
🤖 Generator — Gera respostas inteligentes usando LLM + contexto recuperado

Este é o último passo do pipeline RAG:
  Pergunta + Contexto recuperado → [GENERATOR] → Resposta fundamentada

PROMPT ENGINEERING NA PRÁTICA:
  O prompt é composto por 3 partes:
    1. SYSTEM PROMPT: Instruções de comportamento para o LLM
       "Você é um especialista em código que responde baseado no contexto fornecido"
    2. CONTEXTO: Os trechos de código recuperados pelo retriever
    3. USER PROMPT: A pergunta do usuário

  A qualidade da resposta depende MUITO do system prompt.
  Um prompt mal escrito gera respostas genéricas.
  Um prompt bem escrito gera respostas específicas e referenciadas.
"""

from dataclasses import dataclass
from openai import OpenAI

from src.config import OPENAI_API_KEY, LLM_MODEL
from src.retrieval.retriever import RetrievalResult, retrieve, format_context


# System prompt — define o comportamento do LLM
# Este prompt foi cuidadosamente escrito para:
#   1. Forçar o LLM a basear respostas APENAS no código fornecido
#   2. Citar os arquivos de onde tirou a informação
#   3. Admitir quando não sabe (evita "alucinações")
#   4. Responder em português
SYSTEM_PROMPT = """Você é o Code Intelligence, um assistente especialista em análise de código-fonte.

REGRAS OBRIGATÓRIAS:
1. Responda EXCLUSIVAMENTE com base nos trechos de código fornecidos como contexto.
2. SEMPRE cite os arquivos relevantes no formato: `arquivo.ext` (linhas X-Y).
3. Se o contexto não contiver informação suficiente para responder, diga claramente: "Não encontrei informação suficiente nos trechos recuperados para responder com certeza."
4. Responda em português do Brasil.
5. Use formatação Markdown para melhor legibilidade.
6. Quando explicar código, inclua trechos curtos como exemplos.
7. Explique conceitos técnicos de forma clara e acessível.

FORMATO DA RESPOSTA:
- Comece com uma resposta direta e concisa
- Depois aprofunde com detalhes do código
- Termine listando os arquivos consultados como "📁 Fontes:"
"""


@dataclass
class GenerationResult:
    """
    Resultado completo de uma query RAG.

    Contém a resposta do LLM, os trechos de código usados como
    contexto, e metadata sobre o processo.
    """
    answer: str
    sources: list[RetrievalResult]
    model: str
    tokens_used: int


def generate_answer(
    query: str,
    top_k: int = 5,
) -> GenerationResult:
    """
    Pipeline RAG completo: pergunta → retrieval → geração.

    Este é o ponto de entrada principal do sistema.
    Orquestra todo o fluxo:
      1. Busca trechos relevantes (retrieve)
      2. Formata como contexto (format_context)
      3. Constrói o prompt (system + context + query)
      4. Chama o LLM (OpenAI API)
      5. Retorna resposta estruturada

    Args:
        query: Pergunta do usuário em linguagem natural
        top_k: Quantos trechos de código usar como contexto

    Returns:
        GenerationResult com resposta, fontes e metadata
    """
    # 1. Retrieval — busca os trechos mais relevantes
    retrieval_results = retrieve(query, top_k=top_k)

    # 2. Formata o contexto para o prompt
    context = format_context(retrieval_results)

    # 3. Constrói a mensagem para o LLM
    # CONCEITO - Messages:
    #   system: instrui o LLM sobre como se comportar
    #   user: a pergunta real do usuário, com o contexto incluído
    user_message = (
        f"## Contexto (trechos de código do repositório):\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"## Pergunta:\n{query}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # 4. Chama o LLM
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.1,  # Baixa temperatura = respostas mais precisas e determinísticas
        max_tokens=2000,
    )

    # 5. Extrai a resposta
    answer = response.choices[0].message.content or "Sem resposta."
    tokens = response.usage.total_tokens if response.usage else 0

    return GenerationResult(
        answer=answer,
        sources=retrieval_results,
        model=LLM_MODEL,
        tokens_used=tokens,
    )
