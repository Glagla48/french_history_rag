import sys
from pathlib import Path
import asyncio

asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.rag.rag_pipeline import get_chat

def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

@st.cache_resource
def get_cached_chat():
    return get_chat()

def refresh_model():
    get_cached_chat.clear()
    st.session_state.model, st.session_state.retriever = get_cached_chat()

async def get_response(model, prompt):
    try:
        result = await model.achat(prompt)
        return result.response
    except Exception as e:
        if "TCPTransport" in str(e) or "closed" in str(e):
            refresh_model()
            result = await st.session_state.model.achat(prompt)
            return result.response
        raise e

async def debug_retrieval(prompt):
    nodes = await st.session_state.retriever.aretrieve(prompt)
    st.write(f"**Nodes récupérés : {len(nodes)}**")
    for n in nodes:
        st.write(f"Score: {n.score} | {n.text[:200]}")

def main():
    if "model" not in st.session_state:
        st.session_state.model, st.session_state.retriever = get_cached_chat()

    st.title("French History RAG")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Posez moi vos questions sur l'histoire de France !"):
        if not prompt.strip():
            pass
        else:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            run_async(debug_retrieval(prompt))

            try:
                response = run_async(get_response(st.session_state.model, prompt))
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Erreur : {e}")

if __name__ == "__main__":
    main()