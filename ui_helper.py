import streamlit as st

from type_helper import ChatTurnScored, DocChunkScored


def write_chat_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})
    with st.chat_message(role):
        st.write(content)


def write_chat_memory_sidebar(content: list[ChatTurnScored], title: str) -> None:
    with st.sidebar:
        st.subheader(title)
        if not content:
            st.caption("_No matches_")
            return
        for item in content:
            with st.expander(f"chat - score {item['score']:.3f}"):
                st.markdown(f"**Q:** {item['raw_question']}")
                st.markdown(f"**A:** {item['raw_response']}")


def write_doc_sidebar(content: list[DocChunkScored], title: str) -> None:
    with st.sidebar:
        st.subheader(title)
        if not content:
            st.caption("_No matches_")
            return
        for item in content:
            with st.expander(f"{item['file_name']} - p.{item['page']} - score {item['score']:.3f}"):
                st.markdown(item["text_content"])