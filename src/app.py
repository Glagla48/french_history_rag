import streamlit as st
from rag import get_chat_engine

if "model" not in st.session_state:
    st.session_state.model = get_chat_engine()

st.title("French History RAG")

#chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# React to user input
if prompt := st.chat_input("What is your question about French History?"):
    if not prompt.strip():
        pass
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            response = st.session_state.model.chat(prompt)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Erreur : {e}")