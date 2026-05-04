import streamlit as st

from type_helper import EmbeddingsScoredType


def write_chat_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})
    with st.chat_message(role):
        st.write(content)


def write_sidebar_message(content: EmbeddingsScoredType, title: str) -> None:
    with st.sidebar:
        st.write(title)
        st.json(content)