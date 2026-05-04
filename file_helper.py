from pypdf import PdfReader
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from type_helper import DocChunk


PAGE_SEPARATOR = "\n"
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""],
    add_start_index=True,
)


# Splits the pdf files in multiple chunks
def extract_pages(pdf_file: UploadedFile, embeddings_model: HuggingFaceEmbeddings) -> list[DocChunk]:
    reader = PdfReader(pdf_file)
    
    parts: list[str] = []
    page_offsets: list[tuple[int, int]] = []
    offset = 0
    for page_num, page in enumerate(reader.pages, start=1):
        extracted_text = page.extract_text()
        if not extracted_text:
            continue
        extracted_text = extracted_text.strip()
        if not extracted_text:
            continue
        page_offsets.append((page_num, offset))
        parts.append(extracted_text)
        offset += len(extracted_text) + len(PAGE_SEPARATOR)
    full_text = PAGE_SEPARATOR.join(parts)

    docs = splitter.create_documents([full_text])

    parsed_chunks = []
    for doc in docs:
        if len(doc.page_content) < 50:
            continue
        start = doc.metadata["start_index"]
        page_num = page_offsets[0][0]
        for p, off in page_offsets:
            if off <= start:
                page_num = p
            else:
                break
        parsed_chunks.append({
            "text_content": doc.page_content,
            "page": page_num,
            "file_name": pdf_file.name,
            "embedding": embeddings_model.embed_query(doc.page_content),
        })
    
    return parsed_chunks


# Creates embeddings for the chunks and adds them to the extracted data
def process_files(files: list[UploadedFile], embeddings_model: HuggingFaceEmbeddings) -> list[DocChunk]:
    extracted_data = []

    for f in files:
        if f.type == "application/pdf":
            file = extract_pages(f, embeddings_model)
            extracted_data.extend(file)

    return extracted_data