from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from src.db.models import QueryLogDTO, EvaluationScoreDTO

class BaseDatabase(ABC):
    
    @abstractmethod
    def initialize(self) -> None:
        """Initializes database tables, indices, and connections."""
        pass
        
    @abstractmethod
    def save_query_log(self, query_log: QueryLogDTO) -> int:
        """Saves a query execution log. Returns the generated log ID."""
        pass
        
    @abstractmethod
    def save_evaluation(self, eval_score: EvaluationScoreDTO) -> None:
        """Saves RAG evaluation scores for a specific query log."""
        pass
        
    @abstractmethod
    def get_query_logs(self, limit: int = 100) -> List[QueryLogDTO]:
        """Retrieves a list of query logs, ordered by newest first."""
        pass
        
    @abstractmethod
    def get_query_log_by_id(self, log_id: int) -> Optional[QueryLogDTO]:
        """Retrieves a single query log by its ID."""
        pass
        
    @abstractmethod
    def get_evaluation_by_query_id(self, query_id: int) -> Optional[EvaluationScoreDTO]:
        """Retrieves evaluation scores associated with a query log ID."""
        pass

    @abstractmethod
    def get_logs_with_evaluations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves query logs denormalized/joined with their evaluations."""
        pass

    @abstractmethod
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Computes summary statistics: total queries, total cost, average latency, etc."""
        pass

    @abstractmethod
    def get_model_comparison_data(self) -> List[Dict[str, Any]]:
        """Retrieves aggregated performance and evaluation metrics by LLM Model."""
        pass

    @abstractmethod
    def get_embedding_comparison_data(self) -> List[Dict[str, Any]]:
        """Retrieves aggregated retrieval performance metrics by Embedding Model."""
        pass
        
    @abstractmethod
    def clear_database(self) -> None:
        """Clears all logged data from the database (useful for resets/testing)."""
        pass
