import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import oracledb
import array
from type_helper import EmbeddingsScoredType, EmbeddingsType

load_dotenv()


@st.cache_resource
def init_db_connection() -> oracledb.Connection:
    return oracledb.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dsn=os.getenv("DB_DSN")
    )


# Inserts embeddings and related metadata into the database
def insert_records(
        db_connection: oracledb.Connection, 
        records: EmbeddingsType
    ) -> None:
    cursor = db_connection.cursor()
    for record in records:
        text_val = record.get("text_content", "")
        emb_val = array.array("f", record.get("embedding", []))
        
        cursor.execute("""
            INSERT INTO document_chunks (file_name, page, text_content, raw_question, raw_response, embedding)
            VALUES (:1, :2, :3, :4, :5, :6)
        """, [
            record.get("file_name", ""),
            record.get("page", 0),
            text_val,
            record.get("raw_question", ""),
            record.get("raw_response", ""),
            emb_val
        ])
    db_connection.commit()
    cursor.close()


# Fetches relevant records from the database based on embedding similarity
def fetch_records(
        db_connection: oracledb.Connection, 
        embeddings_model: HuggingFaceEmbeddings, 
        text: str
    ) -> tuple[EmbeddingsScoredType, EmbeddingsScoredType]:
    query_embedding = embeddings_model.embed_query(text)
    vec_query = array.array("f", query_embedding)
    
    cursor = db_connection.cursor()
    cursor.execute("""
        SELECT text_content, file_name, page, VECTOR_DISTANCE(embedding, :1) as distance
        FROM document_chunks
        WHERE file_name = 'user_chat_message'
        ORDER BY distance
        FETCH FIRST 2 ROWS ONLY
    """, [vec_query])
    results = cursor.fetchall()
    cursor.execute("""
        SELECT text_content, file_name, page, VECTOR_DISTANCE(embedding, :1) as distance
        FROM document_chunks
        WHERE file_name != 'user_chat_message'
        ORDER BY distance
        FETCH FIRST 2 ROWS ONLY
    """, [vec_query])
    results.extend(cursor.fetchall())
    cursor.close()
   
    results_processed: EmbeddingsScoredType = []

    for row in results:
        raw_text = row[0].read() if hasattr(row[0], 'read') else str(row[0])
        
        results_processed.append({
            "text_content": raw_text, 
            "file_name": row[1], 
            "page": row[2],
            "score": round(float(row[3]), 4)
        })

    chat_memory = list(filter(lambda x: x["file_name"] == "user_chat_message", results_processed))
    document_info = list(filter(lambda x: x["file_name"] != "user_chat_message", results_processed))

    return chat_memory, document_info