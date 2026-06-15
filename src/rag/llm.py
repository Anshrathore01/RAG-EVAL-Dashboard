from typing import Optional
from langchain_mistralai import ChatMistralAI
from src.config import get_api_key

def get_llm_model(model_name: str, temperature: float = 0.0) -> ChatMistralAI:
    """
    Factory to instantiate Mistral Chat models.
    
    Supported models:
    - mistral-large-latest
    - mistral-medium-latest
    - mistral-small-latest
    """
    api_key = get_api_key()
    
    valid_models = [
        "mistral-large-latest",
        "mistral-medium-latest",
        "mistral-small-latest"
    ]
    
    if model_name not in valid_models:
        raise ValueError(
            f"Unsupported LLM model: {model_name}. "
            f"Valid models are: {', '.join(valid_models)}"
        )
        
    return ChatMistralAI(
        model=model_name,
        temperature=temperature,
        mistral_api_key=api_key
    )
