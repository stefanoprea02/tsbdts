from typing import TypedDict

class EmbeddingEntry(TypedDict):
    text_content: str
    page: int | None
    file_name: str | None
    vector: list[float]
    raw_question: str | None
    raw_response: str | None

class EmbeddingEntryScored(EmbeddingEntry):
    score: float

EmbeddingsType = list[EmbeddingEntry]
EmbeddingsScoredType = list[EmbeddingEntryScored]