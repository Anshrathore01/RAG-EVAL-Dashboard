import json
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class QueryLogDTO:
    query: str
    response: str
    contexts: List[str]
    model_name: str
    embedding_name: str
    chunk_size: int
    chunk_overlap: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    latency_ms: float
    session_id: str = "default"
    id: Optional[int] = None
    created_at: Optional[str] = None

    def contexts_to_json(self) -> str:
        """Serializes contexts list into a JSON string for SQLite storage."""
        return json.dumps(self.contexts)

    @staticmethod
    def json_to_contexts(json_str: str) -> List[str]:
        """Deserializes JSON string from database back into a list of contexts."""
        try:
            return json.loads(json_str)
        except (TypeError, json.JSONDecodeError):
            return []

@dataclass
class EvaluationScoreDTO:
    query_id: int
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    id: Optional[int] = None
    created_at: Optional[str] = None
