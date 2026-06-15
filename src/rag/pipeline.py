import time
import tiktoken
from typing import List, Dict, Any, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import PRICING_CONFIG
from src.rag.embeddings import get_embedding_model
from src.rag.llm import get_llm_model
from src.rag.vectorstore import VectorStoreManager

def count_tokens(text: str) -> int:
    """Helper to count tokens in a text block using tiktoken (cl100k_base)."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback approximation
        return max(1, len(text) // 4)

class RAGPipeline:
    def __init__(self, vectorstore_manager: VectorStoreManager = None):
        self.vectorstore_manager = vectorstore_manager or VectorStoreManager()

    def run(
        self, 
        query: str, 
        model_name: str, 
        embedding_name: str, 
        top_k: int = 4,
        temperature: float = 0.0,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Runs the full RAG pipeline: retrieval, formatting, generation, and metrics collection.
        """
        start_time = time.time()
        
        # 1. RETRIEVAL PHASE
        retrieval_start = time.time()
        retrieved_docs = self.vectorstore_manager.retrieve_contexts(
            query=query, 
            embedding_name=embedding_name, 
            top_k=top_k
        )
        retrieval_latency = (time.time() - retrieval_start) * 1000  # ms
        
        contexts = [doc.page_content for doc in retrieved_docs]
        context_str = "\n\n".join(contexts) if contexts else "No relevant context found."

        # 2. GENERATION PHASE
        # Load LLM
        llm = get_llm_model(model_name, temperature=temperature)
        
        # Build prompt
        system_prompt = (
            "You are a helpful assistant. Use the following context to answer the user's question. "
            "If you do not know the answer based on the context, state that clearly.\n\n"
            f"Context:\n{context_str}"
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        # Execute LLM and measure latency
        llm_start = time.time()
        response = llm.invoke(messages)
        llm_latency = (time.time() - llm_start) * 1000  # ms
        
        answer = response.content
        total_latency = (time.time() - start_time) * 1000  # ms

        # 3. METRICS AND COST COMPUTATION
        # Get token usage from response metadata or count it manually
        token_usage = response.response_metadata.get("token_usage", {})
        prompt_tokens = token_usage.get("prompt_tokens")
        completion_tokens = token_usage.get("completion_tokens")
        
        # If metadata is missing token info, calculate it manually
        if not prompt_tokens or not completion_tokens:
            prompt_tokens = count_tokens(system_prompt) + count_tokens(query)
            completion_tokens = count_tokens(answer)
            
        total_tokens = prompt_tokens + completion_tokens

        # Cost calculations
        # 3a. Embedding Cost
        embedding_tokens = count_tokens(query)
        embedding_rate = PRICING_CONFIG["embeddings"].get(embedding_name, 0.0)
        embedding_cost = embedding_tokens * embedding_rate
        
        # 3b. LLM Cost
        llm_rates = PRICING_CONFIG["llm"].get(model_name, {"input": 0.0, "output": 0.0})
        llm_input_cost = prompt_tokens * llm_rates["input"]
        llm_output_cost = completion_tokens * llm_rates["output"]
        
        total_cost = embedding_cost + llm_input_cost + llm_output_cost

        return {
            "query": query,
            "response": answer,
            "contexts": contexts,
            "model_name": model_name,
            "embedding_name": embedding_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": total_cost,
            "latency_ms": total_latency,
            "retrieval_latency_ms": retrieval_latency,
            "llm_latency_ms": llm_latency,
            "session_id": session_id
        }
