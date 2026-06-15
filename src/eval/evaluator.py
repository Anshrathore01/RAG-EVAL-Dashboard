import logging
import nest_asyncio
from typing import List, Dict, Any, Optional, Tuple
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from src.rag.llm import get_llm_model
from src.rag.embeddings import get_embedding_model
from src.db.models import EvaluationScoreDTO

import asyncio
try:
    # Switch off uvloop if it is set as active, since nest_asyncio cannot patch uvloop
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
except Exception:
    pass

# Apply nest_asyncio to allow Ragas' asynchronous loop to run inside Streamlit
nest_asyncio.apply()

logger = logging.getLogger(__name__)

# Handle Ragas version differences in Wrapper imports
try:
    from ragas.llms import LangchainLLMWrapper as RagasLLMWrapper
except ImportError:
    try:
        from ragas.llms import LangchainLLM as RagasLLMWrapper
    except ImportError:
        RagasLLMWrapper = None
        logger.warning("Could not import LangchainLLMWrapper or LangchainLLM from ragas.llms")

try:
    from ragas.embeddings import LangchainEmbeddingsWrapper as RagasEmbeddingsWrapper
except ImportError:
    try:
        from ragas.embeddings import LangchainEmbeddings as RagasEmbeddingsWrapper
    except ImportError:
        RagasEmbeddingsWrapper = None
        logger.warning("Could not import LangchainEmbeddingsWrapper or LangchainEmbeddings from ragas.embeddings")


class RagasEvaluator:
    def __init__(
        self, 
        evaluator_llm_name: str = "mistral-large-latest",
        evaluator_embeddings_name: str = "mistral-embed"
    ):
        self.llm_name = evaluator_llm_name
        self.embeddings_name = evaluator_embeddings_name

    def _get_evaluator_models(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Instantiates and wraps the LangChain models for Ragas."""
        # 1. Initialize LangChain models
        try:
            lc_llm = get_llm_model(self.llm_name, temperature=0.0)
            lc_embeddings = get_embedding_model(self.embeddings_name)
        except Exception as e:
            logger.error(f"Error loading evaluator models: {e}")
            return None, None

        # 2. Wrap them for Ragas
        ragas_llm = None
        ragas_embeddings = None
        
        if RagasLLMWrapper:
            try:
                ragas_llm = RagasLLMWrapper(lc_llm)
            except Exception as e:
                logger.error(f"Error wrapping LLM for Ragas: {e}")
                
        if RagasEmbeddingsWrapper:
            try:
                ragas_embeddings = RagasEmbeddingsWrapper(lc_embeddings)
            except Exception as e:
                logger.error(f"Error wrapping embeddings for Ragas: {e}")
                
        return ragas_llm, ragas_embeddings

    def evaluate_query(
        self, 
        query: str, 
        response: str, 
        contexts: List[str], 
        ground_truth: Optional[str] = None
    ) -> Dict[str, Optional[float]]:
        """
        Evaluates a single query/response run.
        If ground_truth is provided, computes: faithfulness, answer_relevancy, context_precision, context_recall.
        Otherwise computes: faithfulness, answer_relevancy, context_precision.
        """
        # Ensure contexts is not empty (Ragas requires at least one context chunk)
        if not contexts:
            contexts = ["No context retrieved."]

        # 1. Construct evaluation dataset
        data = {
            "question": [query],
            "answer": [response],
            "contexts": [contexts]
        }
        
        metrics = [faithfulness, answer_relevancy, context_precision]
        
        if ground_truth:
            data["ground_truth"] = [ground_truth]
            metrics.append(context_recall)

        dataset = Dataset.from_dict(data)
        
        # 2. Load evaluator models
        ragas_llm, ragas_embeddings = self._get_evaluator_models()
        
        if not ragas_llm or not ragas_embeddings:
            logger.error("Could not run evaluation: Ragas wrappers are unavailable or misconfigured.")
            return {
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
                "context_recall": None
            }

        try:
            # 3. Run evaluation
            # For newer Ragas, we can assign LLM and Embeddings to each metric
            # Or pass them to evaluate() depending on the version. We'll pass them to evaluate().
            eval_result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                raise_exceptions=False
            )
            
            # Extract results safely
            return {
                "faithfulness": eval_result.get("faithfulness"),
                "answer_relevancy": eval_result.get("answer_relevancy"),
                "context_precision": eval_result.get("context_precision"),
                "context_recall": eval_result.get("context_recall") if ground_truth else None
            }
        except Exception as e:
            logger.error(f"Ragas evaluation failed: {e}")
            return {
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
                "context_recall": None
            }
