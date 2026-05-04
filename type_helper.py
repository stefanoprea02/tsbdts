from typing import TypedDict


class DocChunk(TypedDict):
    text_content: str
    file_name: str
    page: int
    embedding: list[float]


class ChatTurn(TypedDict):
    text_content: str
    raw_question: str
    raw_response: str
    embedding: list[float]


class DocChunkScored(TypedDict):
    text_content: str
    file_name: str
    page: int
    score: float


class ChatTurnScored(TypedDict):
    text_content: str
    raw_question: str
    raw_response: str
    score: float
