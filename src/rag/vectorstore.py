import os
from typing import List, Dict, Any
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.config import CHROMA_DIR
from src.rag.embeddings import get_embedding_model

class VectorStoreManager:
    def __init__(self, persist_dir: str = str(CHROMA_DIR)):
        self.persist_dir = persist_dir

    def _get_collection_name(self, embedding_name: str) -> str:
        """Returns a clean collection name specific to the embedding model."""
        clean_name = embedding_name.replace("-", "_").replace(".", "_")
        return f"collection_{clean_name}"

    def get_vectorstore(self, embedding_name: str) -> Chroma:
        """Loads or creates a Chroma vectorstore instance for a specific embedding model."""
        embedding_model = get_embedding_model(embedding_name)
        collection_name = self._get_collection_name(embedding_name)
        
        return Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=self.persist_dir
        )

    def ingest_file(
        self, 
        file_path: Path, 
        embedding_name: str, 
        chunk_size: int = 500, 
        chunk_overlap: int = 50
    ) -> int:
        """
        Loads, splits, and embeds a file into the specific Chroma collection.
        Returns the number of created chunks.
        """
        # 1. Load document based on file extension
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
        elif suffix in [".txt", ".md", ".json"]:
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents = loader.load()
        else:
            # Fallback text reading
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            documents = [Document(page_content=text, metadata={"source": file_path.name})]

        # 2. Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(documents)
        
        if not chunks:
            return 0

        # 3. Add to Chroma vector database
        db = self.get_vectorstore(embedding_name)
        db.add_documents(chunks)
        db.persist()
        
        return len(chunks)

    def retrieve_contexts(
        self, 
        query: str, 
        embedding_name: str, 
        top_k: int = 4
    ) -> List[Document]:
        """Retrieves top K documents matching the query using the specified embedding collection."""
        db = self.get_vectorstore(embedding_name)
        # Check if the collection has any elements to prevent empty query errors
        try:
            results = db.similarity_search(query, k=top_k)
            return results
        except Exception:
            return []

    def clear_collection(self, embedding_name: str) -> None:
        """Deletes all items in the specified embedding collection."""
        db = self.get_vectorstore(embedding_name)
        # Chroma doesn't have a direct clear() function in all versions, we can delete the collection or get all ids and delete.
        try:
            coll = db._client.get_collection(self._get_collection_name(embedding_name))
            if coll.count() > 0:
                # Get all documents ids and delete
                results = coll.get()
                if results and "ids" in results and results["ids"]:
                    coll.delete(ids=results["ids"])
                db.persist()
        except Exception:
            pass
