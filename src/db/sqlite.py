import sqlite3
import json
from typing import List, Optional, Dict, Any
from src.config import DB_PATH
from src.db.interface import BaseDatabase
from src.db.models import QueryLogDTO, EvaluationScoreDTO

class SQLiteDatabase(BaseDatabase):
    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.initialize()

    def _get_connection(self) -> sqlite3.Connection:
        return self.conn

    def initialize(self) -> None:
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create query_logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    contexts TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    embedding_name TEXT NOT NULL,
                    chunk_size INTEGER NOT NULL,
                    chunk_overlap INTEGER NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    cost REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create evaluation_scores table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id INTEGER UNIQUE NOT NULL,
                    faithfulness REAL,
                    answer_relevancy REAL,
                    context_precision REAL,
                    context_recall REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (query_id) REFERENCES query_logs (id) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def save_query_log(self, query_log: QueryLogDTO) -> int:
        """Saves a query execution log. Returns the generated log ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO query_logs (
                    session_id, query, response, contexts, model_name, embedding_name, 
                    chunk_size, chunk_overlap, prompt_tokens, completion_tokens, 
                    total_tokens, cost, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                query_log.session_id,
                query_log.query,
                query_log.response,
                query_log.contexts_to_json(),
                query_log.model_name,
                query_log.embedding_name,
                query_log.chunk_size,
                query_log.chunk_overlap,
                query_log.prompt_tokens,
                query_log.completion_tokens,
                query_log.total_tokens,
                query_log.cost,
                query_log.latency_ms
            ))
            conn.commit()
            return cursor.lastrowid

    def save_evaluation(self, eval_score: EvaluationScoreDTO) -> None:
        """Saves or updates RAG evaluation scores for a specific query log."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO evaluation_scores (
                    query_id, faithfulness, answer_relevancy, context_precision, context_recall
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(query_id) DO UPDATE SET
                    faithfulness = excluded.faithfulness,
                    answer_relevancy = excluded.answer_relevancy,
                    context_precision = excluded.context_precision,
                    context_recall = excluded.context_recall
            """, (
                eval_score.query_id,
                eval_score.faithfulness,
                eval_score.answer_relevancy,
                eval_score.context_precision,
                eval_score.context_recall
            ))
            conn.commit()

    def _row_to_query_log(self, row: sqlite3.Row) -> QueryLogDTO:
        return QueryLogDTO(
            id=row["id"],
            session_id=row["session_id"],
            query=row["query"],
            response=row["response"],
            contexts=QueryLogDTO.json_to_contexts(row["contexts"]),
            model_name=row["model_name"],
            embedding_name=row["embedding_name"],
            chunk_size=row["chunk_size"],
            chunk_overlap=row["chunk_overlap"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            total_tokens=row["total_tokens"],
            cost=row["cost"],
            latency_ms=row["latency_ms"],
            created_at=row["created_at"]
        )

    def get_query_logs(self, limit: int = 100) -> List[QueryLogDTO]:
        """Retrieves a list of query logs, ordered by newest first."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM query_logs 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [self._row_to_query_log(row) for row in rows]

    def get_query_log_by_id(self, log_id: int) -> Optional[QueryLogDTO]:
        """Retrieves a single query log by its ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM query_logs WHERE id = ?", (log_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_query_log(row)
            return None

    def get_evaluation_by_query_id(self, query_id: int) -> Optional[EvaluationScoreDTO]:
        """Retrieves evaluation scores associated with a query log ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM evaluation_scores WHERE query_id = ?", (query_id,))
            row = cursor.fetchone()
            if row:
                return EvaluationScoreDTO(
                    id=row["id"],
                    query_id=row["query_id"],
                    faithfulness=row["faithfulness"],
                    answer_relevancy=row["answer_relevancy"],
                    context_precision=row["context_precision"],
                    context_recall=row["context_recall"],
                    created_at=row["created_at"]
                )
            return None

    def get_logs_with_evaluations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves query logs denormalized/joined with their evaluations."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    ql.id, ql.session_id, ql.query, ql.response, ql.contexts, ql.model_name, 
                    ql.embedding_name, ql.chunk_size, ql.chunk_overlap, ql.prompt_tokens, 
                    ql.completion_tokens, ql.total_tokens, ql.cost, ql.latency_ms, ql.created_at,
                    es.faithfulness, es.answer_relevancy, es.context_precision, es.context_recall
                FROM query_logs ql
                LEFT JOIN evaluation_scores es ON ql.id = es.query_id
                ORDER BY ql.created_at DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["contexts"] = QueryLogDTO.json_to_contexts(d["contexts"])
                result.append(d)
            return result

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Computes summary statistics: total queries, total cost, average latency, etc."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Query count, cost sum, avg latency
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_queries,
                    SUM(cost) as total_cost,
                    AVG(latency_ms) as avg_latency,
                    SUM(total_tokens) as total_tokens
                FROM query_logs
            """)
            summary_row = cursor.fetchone()
            
            # Averages of evaluation scores
            cursor.execute("""
                SELECT 
                    AVG(faithfulness) as avg_faithfulness,
                    AVG(answer_relevancy) as avg_answer_relevancy,
                    AVG(context_precision) as avg_context_precision,
                    AVG(context_recall) as avg_context_recall
                FROM evaluation_scores
            """)
            eval_row = cursor.fetchone()
            
            # Cost trends (cost grouped by day)
            cursor.execute("""
                SELECT 
                    DATE(created_at) as date_day,
                    SUM(cost) as cost_sum,
                    COUNT(*) as query_count
                FROM query_logs
                GROUP BY date_day
                ORDER BY date_day ASC
                LIMIT 30
            """)
            trends = [dict(row) for row in cursor.fetchall()]

            return {
                "total_queries": summary_row["total_queries"] or 0,
                "total_cost": summary_row["total_cost"] or 0.0,
                "avg_latency": summary_row["avg_latency"] or 0.0,
                "total_tokens": summary_row["total_tokens"] or 0,
                "avg_faithfulness": eval_row["avg_faithfulness"] if eval_row and eval_row["avg_faithfulness"] is not None else None,
                "avg_answer_relevancy": eval_row["avg_answer_relevancy"] if eval_row and eval_row["avg_answer_relevancy"] is not None else None,
                "avg_context_precision": eval_row["avg_context_precision"] if eval_row and eval_row["avg_context_precision"] is not None else None,
                "avg_context_recall": eval_row["avg_context_recall"] if eval_row and eval_row["avg_context_recall"] is not None else None,
                "cost_trends": trends
            }

    def get_model_comparison_data(self) -> List[Dict[str, Any]]:
        """Retrieves aggregated performance and evaluation metrics by LLM Model."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    ql.model_name,
                    COUNT(ql.id) as query_count,
                    AVG(ql.latency_ms) as avg_latency_ms,
                    AVG(ql.cost) as avg_cost,
                    AVG(es.faithfulness) as avg_faithfulness,
                    AVG(es.answer_relevancy) as avg_answer_relevancy,
                    AVG(es.context_precision) as avg_context_precision,
                    AVG(es.context_recall) as avg_context_recall
                FROM query_logs ql
                LEFT JOIN evaluation_scores es ON ql.id = es.query_id
                GROUP BY ql.model_name
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_embedding_comparison_data(self) -> List[Dict[str, Any]]:
        """Retrieves aggregated retrieval performance metrics by Embedding Model."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    ql.embedding_name,
                    COUNT(ql.id) as query_count,
                    AVG(ql.latency_ms) as avg_latency_ms,
                    AVG(ql.cost) as avg_cost,
                    AVG(es.context_precision) as avg_context_precision,
                    AVG(es.context_recall) as avg_context_recall
                FROM query_logs ql
                LEFT JOIN evaluation_scores es ON ql.id = es.query_id
                GROUP BY ql.embedding_name
            """)
            return [dict(row) for row in cursor.fetchall()]

    def clear_database(self) -> None:
        """Clears all logged data from the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM evaluation_scores")
            cursor.execute("DELETE FROM query_logs")
            conn.commit()
