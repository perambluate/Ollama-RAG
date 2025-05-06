import chromadb
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from typing import List, Any, Dict
import hashlib
from dataclasses import dataclass

@dataclass
class DocumentData:
    file: str
    timestamp: float
    docs: List[Any]

class VectorStore:
    def __init__(
        self, 
        persist_directory: str,
        embedding_function: Embeddings,
        search_k: int = 3,
        path_to_ids: Dict[str, List[str]] = None
    ):
        self.persist_directory = persist_directory
        self.search_k = search_k
        
        # Initialize ChromaDB
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            client_settings=chromadb.config.Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        self.id_counter = 0
        self.path_to_ids = path_to_ids if path_to_ids else {}
    
    def _generate_unique_id(self, path: str, timestamp: float, num: int = 1) -> List[str]:
        """Generate unique IDs for document chunks"""
        ids = []
        for i in range(num):
            unique_string = f"{path}_{timestamp}_{self.id_counter}"
            ids.append(hashlib.sha256(unique_string.encode()).hexdigest()[:32])
            self.id_counter += 1
        return ids
    
    def add_documents(self, documents_to_update: List[DocumentData]) -> Dict[str, List[str]]:
        """Add documents to vector store
            Returns:
                Dict[str, List[str]]: Mapping of file paths to their unique IDs
        """

        # Update path to IDs mapping
        updated_path_ids = {}
        for doc_info in documents_to_update:
            docs = doc_info['docs']
            path = doc_info['file']
            ts = doc_info['timestamp']
            ids = self._generate_unique_id(path, ts, len(docs))
            updated_path_ids[path] = ids
            self.db.add_documents(docs, ids=ids)
        
        self.path_to_ids.update(updated_path_ids)
        return updated_path_ids

    def remove_documents(self, files_to_remove: List[str]) -> None:
        """Remove documents from vector store"""
        ids_to_remove = []

        for path in files_to_remove:
            if path in self.path_to_ids:
                ids_to_remove.extend(self.path_to_ids[path])
                del self.path_to_ids[path]
        
        if ids_to_remove:
            print(f"Removing {len(ids_to_remove)} document chunks from vector store")
            self.db._collection.delete(ids=ids_to_remove)

    def get_retriever(self):
        """Get retriever with configured search parameters"""
        return self.db.as_retriever(
            search_kwargs={"k": self.search_k}
        )

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self.db, '_client'):
            self.db._client = None