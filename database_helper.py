import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import oracledb
import array
from type_helper import ChatTurn, ChatTurnScored, DocChunk, DocChunkScored

load_dotenv()

SIMILARITY_THRESHOLD = 0.7


@st.cache_resource
def init_db_connection() -> oracledb.Connection:
    return oracledb.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dsn=os.getenv("DB_DSN")
    )


# Inserts document chunks into the database
def insert_doc_chunks(
        db_connection: oracledb.Connection,
        records: list[DocChunk]
    ) -> None:
    cursor = db_connection.cursor()
    for record in records:
        emb_val = array.array("f", record["embedding"])
        cursor.execute("""
            INSERT INTO document_chunks (file_name, page, text_content, embedding)
            VALUES (:1, :2, :3, :4)
        """, [
            record["file_name"],
            record["page"],
            record["text_content"],
            emb_val
        ])
    db_connection.commit()
    cursor.close()


# Insert chat turn into the database
def insert_chat_turn(
        db_connection: oracledb.Connection,
        turn: ChatTurn
    ) -> None:
    cursor = db_connection.cursor()
    emb_val = array.array("f", turn["embedding"])
    cursor.execute("""
        INSERT INTO chat_turns (text_content, raw_question, raw_response, embedding)
        VALUES (:1, :2, :3, :4)
    """, [
        turn["text_content"],
        turn["raw_question"],
        turn["raw_response"],
        emb_val
    ])
    db_connection.commit()
    cursor.close()


def truncate_all(db_connection: oracledb.Connection) -> None:
    cursor = db_connection.cursor()
    cursor.execute("TRUNCATE TABLE document_chunks")
    cursor.execute("TRUNCATE TABLE chat_turns")
    cursor.close()


# Fetches relevant records from the database based on embedding similarity
def fetch_records(
        db_connection: oracledb.Connection, 
        embeddings_model: HuggingFaceEmbeddings, 
        text: str
    ) -> tuple[list[ChatTurnScored], list[DocChunkScored]]:
    query_embedding = embeddings_model.embed_query(text)
    vec_query = array.array("f", query_embedding)
    
    cursor = db_connection.cursor()
    cursor.execute("""
        SELECT text_content, raw_question, raw_response,
               VECTOR_DISTANCE(embedding, :vec) as distance
        FROM chat_turns
        WHERE VECTOR_DISTANCE(embedding, :vec) < :thresh
        ORDER BY distance
        FETCH FIRST 2 ROWS ONLY
    """, {"vec": vec_query, "thresh": SIMILARITY_THRESHOLD})
    chat_rows = cursor.fetchall()
    cursor.execute("""
        SELECT text_content, file_name, page,
               VECTOR_DISTANCE(embedding, :vec) as distance
        FROM document_chunks
        WHERE VECTOR_DISTANCE(embedding, :vec) < :thresh
        ORDER BY distance
        FETCH FIRST 2 ROWS ONLY
    """, {"vec": vec_query, "thresh": SIMILARITY_THRESHOLD})
    doc_rows = cursor.fetchall()
    cursor.close()

    chat_memory: list[ChatTurnScored] = []
    for row in chat_rows:
        text_content = row[0].read() if hasattr(row[0], 'read') else str(row[0])
        raw_question = row[1].read() if hasattr(row[1], 'read') else str(row[1])
        raw_response = row[2].read() if hasattr(row[2], 'read') else str(row[2])
        chat_memory.append({
            "text_content": text_content,
            "raw_question": raw_question,
            "raw_response": raw_response,
            "score": round(float(row[3]), 4),
        })

    document_info: list[DocChunkScored] = []
    for row in doc_rows:
        text_content = row[0].read() if hasattr(row[0], 'read') else str(row[0])
        document_info.append({
            "text_content": text_content,
            "file_name": row[1],
            "page": row[2],
            "score": round(float(row[3]), 4),
        })

    return chat_memory, document_info