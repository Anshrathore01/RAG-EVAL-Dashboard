import sys
from pathlib import Path
from datetime import datetime

# Add root directory to python path
sys.path.append(str(Path(__file__).resolve().parent))

def test_imports():
    print("=== Testing Imports ===")
    try:
        from src.config import get_api_key, MISTRAL_API_KEY, PRICING_CONFIG
        print("✓ Config imported successfully.")
        print(f"  Mistral Key Loaded: {'Yes' if MISTRAL_API_KEY else 'No'}")
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        return False

    try:
        from src.db.models import QueryLogDTO, EvaluationScoreDTO
        print("✓ Models DTOs imported successfully.")
    except Exception as e:
        print(f"✗ Models DTOs import failed: {e}")
        return False

    try:
        from src.db.sqlite import SQLiteDatabase
        print("✓ SQLiteDatabase client imported successfully.")
    except Exception as e:
        print(f"✗ SQLiteDatabase import failed: {e}")
        return False

    try:
        from src.rag.embeddings import get_embedding_model
        print("✓ Embeddings factory imported successfully.")
    except Exception as e:
        print(f"✗ Embeddings factory import failed: {e}")
        return False

    try:
        from src.rag.llm import get_llm_model
        print("✓ LLM factory imported successfully.")
    except Exception as e:
        print(f"✗ LLM factory import failed: {e}")
        return False

    try:
        from src.rag.vectorstore import VectorStoreManager
        print("✓ VectorStoreManager imported successfully.")
    except Exception as e:
        print(f"✗ VectorStoreManager import failed: {e}")
        return False

    try:
        from src.rag.pipeline import RAGPipeline
        print("✓ RAGPipeline imported successfully.")
    except Exception as e:
        print(f"✗ RAGPipeline import failed: {e}")
        return False

    try:
        from src.eval.evaluator import RagasEvaluator
        print("✓ RagasEvaluator imported successfully.")
    except Exception as e:
        print(f"✗ RagasEvaluator import failed: {e}")
        return False

    return True

def test_database():
    print("\n=== Testing Database Subsystem ===")
    from src.db.sqlite import SQLiteDatabase
    from src.db.models import QueryLogDTO, EvaluationScoreDTO

    db = SQLiteDatabase(":memory:")  # Use in-memory SQLite for testing
    print("✓ Initialized in-memory SQLite database.")

    # 1. Create a dummy Query Log
    query_log = QueryLogDTO(
        session_id="test-session",
        query="What is RAG evaluation?",
        response="RAG evaluation measures retrieval quality and answer quality.",
        contexts=["Context chunk 1 about retrieval.", "Context chunk 2 about Ragas metrics."],
        model_name="mistral-large-latest",
        embedding_name="mistral-embed",
        chunk_size=500,
        chunk_overlap=50,
        prompt_tokens=150,
        completion_tokens=50,
        total_tokens=200,
        cost=0.0006,
        latency_ms=850.5
    )

    log_id = db.save_query_log(query_log)
    print(f"✓ Saved query log. Log ID: {log_id}")
    assert log_id == 1, f"Expected log_id to be 1, got {log_id}"

    # 2. Retrieve Query Log
    retrieved_logs = db.get_query_logs()
    assert len(retrieved_logs) == 1, "Expected 1 query log in DB"
    retrieved_log = retrieved_logs[0]
    print("✓ Successfully retrieved query log from DB:")
    print(f"  Query: '{retrieved_log.query}'")
    print(f"  Contexts: {retrieved_log.contexts}")
    print(f"  Cost: ${retrieved_log.cost:.6f} | Latency: {retrieved_log.latency_ms:.1f}ms")

    # 3. Create and Save Evaluation Score
    eval_score = EvaluationScoreDTO(
        query_id=log_id,
        faithfulness=0.9,
        answer_relevancy=0.95,
        context_precision=0.85,
        context_recall=0.80
    )
    db.save_evaluation(eval_score)
    print("✓ Saved evaluation scores.")

    # 4. Retrieve Evaluation Score
    retrieved_eval = db.get_evaluation_by_query_id(log_id)
    assert retrieved_eval is not None, "Evaluation score should exist"
    print("✓ Successfully retrieved evaluation score from DB:")
    print(f"  Faithfulness: {retrieved_eval.faithfulness}")
    print(f"  Answer Relevancy: {retrieved_eval.answer_relevancy}")
    print(f"  Context Recall: {retrieved_eval.context_recall}")

    # 5. Joined Log & Eval Test
    joined = db.get_logs_with_evaluations()
    assert len(joined) == 1
    assert joined[0]["faithfulness"] == 0.9
    print("✓ Joined Query Logs + Evaluations verified.")

    # 6. Analytics verification
    analytics = db.get_analytics_summary()
    print("✓ Analytics summary computed:")
    print(f"  Total Queries: {analytics['total_queries']}")
    print(f"  Total Cost: ${analytics['total_cost']:.6f}")
    print(f"  Avg Latency: {analytics['avg_latency']:.2f}ms")
    print(f"  Avg Faithfulness: {analytics['avg_faithfulness']:.2f}")

    # 7. Comparison aggregation verification
    model_comparison = db.get_model_comparison_data()
    print("✓ Model comparison aggregations verified.")
    
    embedding_comparison = db.get_embedding_comparison_data()
    print("✓ Embedding comparison aggregations verified.")

    print("=== Database Tests Passed! ===")
    return True

if __name__ == "__main__":
    imports_ok = test_imports()
    if imports_ok:
        test_database()
    else:
        print("Imports failed. Cannot run database tests.")
