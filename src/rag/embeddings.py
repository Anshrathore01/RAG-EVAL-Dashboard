import os
from typing import Optional
from langchain_core.embeddings import Embeddings
from src.config import get_api_key

def get_embedding_model(embedding_name: str) -> Embeddings:
    """
    Factory to load embedding models based on the configuration name.
    
    Supported models:
    - "mistral-embed" (requires Mistral API Key)
    - "bge-small-en-v1.5" (local, sentence-transformers under the hood)
    - "sentence-transformers-all-MiniLM-L6-v2" (local, sentence-transformers)
    """
    if embedding_name == "mistral-embed":
        from langchain_mistralai import MistralAIEmbeddings
        api_key = get_api_key()
        return MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=api_key
        )
        
    elif embedding_name == "bge-small-en-v1.5":
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        
        # BGE embeddings standard config
        model_name = "BAAI/bge-small-en-v1.5"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        
        return HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
    elif embedding_name == "sentence-transformers-all-MiniLM-L6-v2":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {"device": "cpu"}
        
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs
        )
        
    else:
        raise ValueError(
            f"Unsupported embedding model: {embedding_name}. "
            f"Please choose from: mistral-embed, bge-small-en-v1.5, sentence-transformers-all-MiniLM-L6-v2."
        )
