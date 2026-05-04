from typing import NotRequired, TypedDict

class EmbeddingEntry(TypedDict):
    text_content: str
    file_name: str
    page: NotRequired[int]
    embedding: NotRequired[list[float]]
    raw_question: NotRequired[str]
    raw_response: NotRequired[str]

class EmbeddingEntryScored(EmbeddingEntry):
    score: float

EmbeddingsType = list[EmbeddingEntry]
EmbeddingsScoredType = list[EmbeddingEntryScored]