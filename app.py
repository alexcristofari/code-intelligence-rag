"""
🖥️ Code Intelligence RAG — Interface Streamlit

Streamlit transforma scripts Python em apps web interativas.
Não precisa de HTML, CSS ou JavaScript — tudo é Python.

Para rodar:
  streamlit run app.py
"""

import streamlit as st

from src.config import validate_config
from src.generation.generator import generate_answer
from src.retrieval.vector_store import get_collection_stats


# ========================================
# Configuração da Página
# ========================================
st.set_page_config(
    page_title="Code Intelligence RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado para uma interface mais bonita
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #888;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .source-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-left: 3px solid #667eea;
    }
    .score-badge {
        background: #667eea;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ========================================
# Sidebar — Estatísticas e Configuração
# ========================================
with st.sidebar:
    st.markdown("## ⚙️ Configuração")

    # Verifica se API key está configurada
    if validate_config():
        st.success("✅ OpenAI API Key configurada")
    else:
        st.error("❌ Configure OPENAI_API_KEY no arquivo .env")
        st.stop()

    # Estatísticas do vector store
    try:
        stats = get_collection_stats()
        st.markdown("---")
        st.markdown("## 📊 Vector Store")
        col1, col2 = st.columns(2)
        col1.metric("Chunks", stats["total_chunks"])
        col2.metric("Collection", stats["collection_name"])

        if stats["total_chunks"] == 0:
            st.warning("⚠️ Nenhum código ingerido!")
            st.code("python ingest.py <repo-path>", language="bash")
    except Exception as e:
        st.error(f"Erro ao conectar ao ChromaDB: {e}")

    # Configurações de busca
    st.markdown("---")
    st.markdown("## 🔧 Parâmetros")
    top_k = st.slider(
        "Chunks por busca (top_k)",
        min_value=1,
        max_value=15,
        value=5,
        help="Quantos trechos de código usar como contexto. Mais = mais contexto, mas mais caro.",
    )

    st.markdown("---")
    st.markdown(
        "Feito com ❤️ por Alex\n\n"
        "[GitHub](https://github.com) · "
        "[FastAPI Docs](http://localhost:8000/docs)"
    )


# ========================================
# Main — Chat Interface
# ========================================
st.markdown('<p class="main-header">🧠 Code Intelligence</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Faça perguntas sobre seu código — respostas com referências</p>',
    unsafe_allow_html=True,
)

# Inicializa histórico de chat no session_state
# session_state persiste dados entre re-renders do Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input do usuário
if prompt := st.chat_input("Faça uma pergunta sobre o código..."):
    # Adiciona mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Gera resposta
    with st.chat_message("assistant"):
        with st.spinner("🔍 Buscando no código e gerando resposta..."):
            try:
                result = generate_answer(query=prompt, top_k=top_k)

                # Mostra a resposta
                st.markdown(result.answer)

                # Mostra as fontes
                if result.sources:
                    st.markdown("---")
                    st.markdown("#### 📁 Fontes consultadas:")

                    for source in result.sources:
                        start = source.metadata.get("start_line", "?")
                        end = source.metadata.get("end_line", "?")
                        lang = source.metadata.get("language", "")
                        score_pct = round((1 - source.score) * 100, 1)

                        with st.expander(
                            f"📄 {source.filepath} (linhas {start}-{end}) — {score_pct}% relevante"
                        ):
                            st.code(source.content, language=lang if lang != "unknown" else None)

                # Mostra metadata
                st.caption(f"🤖 {result.model} · {result.tokens_used} tokens")

                # Salva no histórico
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.answer,
                })

            except Exception as e:
                st.error(f"❌ Erro: {e}")
                st.info("Verifique se o código foi ingerido: `python ingest.py <repo-path>`")
