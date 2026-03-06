import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import streamlit as st
from llama_cpp import Llama
from type_helper import EmbeddingsScoredType


load_dotenv()

@st.cache_resource
def load_llm():
    repo_id = os.getenv("MODEL_REPO")
    filename = os.getenv("MODEL_FILE")
    local_dir = os.getenv("MODEL_PATH", "./models")
    mode = os.getenv("HARDWARE_MODE", "CPU")
    context_size = int(os.getenv("CONTEXT_SIZE", 4096))

    os.makedirs(local_dir, exist_ok=True)
    model_full_path = os.path.abspath(os.path.join(local_dir, filename))

    if not os.path.exists(model_full_path):
        with st.spinner(f"Downloading {filename} from Hugging Face... Please wait."):
            model_full_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir
            )

    layers = -1 if mode == "GPU" else 0

    return Llama(
        model_path=model_full_path,
        n_gpu_layers=layers,
        n_ctx=context_size,
        verbose=False
    )


# Augment user prompt with context
def generate_prompt(
        user_input: str, 
        chat_memory: EmbeddingsScoredType, 
        document_info: EmbeddingsScoredType
    ) -> str:
    chat_text = "\n".join([f"{mem['text_content']}" for mem in chat_memory])
    document_text = "\n".join([f"Source: {mem['file_name']} - {mem['page']}: {mem['text_content']}" for mem in document_info])
            
    augmented_prompt = f"""Use the following context to answer the question. 
    If the answer isn't in the context, use your general knowledge.

    CHAT MEMORY:
    {chat_text}

    DOCUMENT INFORMATION:
    {document_text}

    QUESTION:
    {user_input}"""

    return augmented_prompt