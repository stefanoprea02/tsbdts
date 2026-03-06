from pypdf import PdfReader
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from type_helper import EmbeddingsType


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)


# Splits the pdf files in multiple chunks per page
def extract_pages(pdf_file: UploadedFile, embeddings_model: HuggingFaceEmbeddings) -> EmbeddingsType:
    reader = PdfReader(pdf_file)
    parsed_chunks = []
    pages = []
    pages_metadata = []

    for page_num, page in enumerate(reader.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text:
            pages.append(extracted_text)
            pages_metadata.append({
                "file_name": pdf_file.name,
                "page": page_num
            })

    docs = splitter.create_documents(pages, metadatas=pages_metadata)

    for doc in docs:
        if len(doc.page_content) > 50:
            parsed_chunks.append({
                "text_content": doc.page_content,
                "page": doc.metadata["page"],
                "file_name": doc.metadata["file_name"],
                "embedding": embeddings_model.embed_query(doc.page_content),
            })
    
    return parsed_chunks


# Creates embeddings for the chunks and adds them to the extracted data
def process_files(files: list[UploadedFile], embeddings_model: HuggingFaceEmbeddings) -> EmbeddingsType:
    extracted_data = []

    for f in files:
        if f.type == "application/pdf":
            file = extract_pages(f, embeddings_model)
            extracted_data.extend(file)

    return extracted_data