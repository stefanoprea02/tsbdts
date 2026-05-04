from llama_cpp import CreateChatCompletionResponse
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from file_helper import process_files
from database_helper import init_db_connection, insert_records, fetch_records
from llm_helper import load_llm, generate_prompt
from ui_helper import write_chat_message, write_sidebar_message
from type_helper import EmbeddingEntry
from typing import cast


@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


embeddings_model = load_embedding_model()
llm = load_llm()
db_connection = init_db_connection()


def initialize_messages():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def processs_message():
    if prompt := st.chat_input("Say something", accept_file='multiple'):
        file_number = len(prompt.files)
        user_text = prompt.text if prompt.text else "No user message."
        file_text = f" {file_number} file uploaded." if file_number > 0 else ""
        content = f"{user_text} {file_text}".strip()

        # Write user message to chat and session state
        write_chat_message("user", content)

        with st.spinner("Model is thinking..."):
            # Process files if any and update knowledge base
            if file_number > 0:
                parsed_files = process_files(prompt.files, embeddings_model)
                insert_records(db_connection, parsed_files)

                if not prompt.text:
                    write_chat_message("assistant", "Files processed.")
                    return
                
            # Fetch relevant records from DB based on prompt
            chat_memory, document_info = fetch_records(db_connection, embeddings_model, prompt.text)
            write_sidebar_message(chat_memory, "Chat Memories")
            write_sidebar_message(document_info, "Document Info")

            prompt_augmented = generate_prompt(prompt.text, chat_memory, document_info)
            
            # Generate response from LLM
            output = cast(CreateChatCompletionResponse, llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_augmented}
                ],
                max_tokens=400,
                temperature=0.3,
                repeat_penalty=1.15,
                stream=False
            ))
            full_response = output["choices"][0]["message"].get("content") or "Failed to generate a response."

            # Embed prompt and response
            combined_text = f"User: {prompt.text}\nAssistant: {full_response}"
            combined_embedding = embeddings_model.embed_query(combined_text)

            message_record: EmbeddingEntry = {
                "text_content": combined_text,
                "embedding": combined_embedding,
                "file_name": "user_chat_message",
                "raw_question": prompt.text,
                "raw_response": full_response
            }
            insert_records(db_connection, [message_record])

            # Respond
            write_chat_message("assistant", full_response)

def main():
    initialize_messages()
    processs_message()

if __name__ == "__main__":
    main()