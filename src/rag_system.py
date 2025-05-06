import os
import subprocess
from typing import List
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM as Ollama
from .config import ConfigManager
from .document_processor import DocumentProcessor
from .file_tracker import FileTracker
from .vector_store import VectorStore

DEFAULT_CONFIG_PATH = "./config/default.yaml" # Default path for configuration file
DEFAULT_DB_ROOT = "./data/vectordb/"  # Default path for vector database root
DEFAULT_TRACK_FILE = ".file_tracker.json"  # Default track file name
DEFAULT_LLM_MODEL = "gemma3:1b"  # Default model name if not specified
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"  # Default embedding model name if not specified
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama base URL if not specified

class RAGSystem:
    def _get_available_models(self) -> List[str]:
        """Get list of locally available models"""
        try:
            result = subprocess.run(
                ['docker', 'exec', '-it', 'ollama', 'ollama', 'list'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                # Parse output to get model names
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                return [line.split()[0] for line in lines]
            return []
        except subprocess.CalledProcessError:
            print("Failed to get model list from Ollama")
            return []

    def _ensure_model_available(self, model_name: str) -> bool:
        """Ensure the requested model is available, pull if needed"""
        available_models = self._get_available_models()
        
        if model_name not in available_models:
            print(f"Model {model_name} not found locally. Pulling...")
            try:
                result = subprocess.run(
                    ['docker', 'exec', '-it', 'ollama', 'ollama', 'pull', model_name]
                )
                if result.returncode == 0:
                    print(f"Successfully pulled model {model_name}")
                    return True
                else:
                    print(f"Failed to pull model {model_name}")
                    return False
            except subprocess.CalledProcessError as e:
                print(f"Error pulling model: {e}")
                return False
        return True

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH, model_name: str = ""):
        self.config = ConfigManager(config_path)
        ollama_config = self.config.get_ollama_config()
        vectordb_config = self.config.get_vectordb_config()
        filetracker_config = self.config.get_file_tracker_config()

        os.environ["OLLAMA_BASE_URL"] = ollama_config.get("base_url", DEFAULT_OLLAMA_BASE_URL)
        # Model precedence: CLI arg > config file > default
        self.model_name = (
            model_name or
            ollama_config.get("model") or
            DEFAULT_LLM_MODEL
        )
        print(f"Using LLM model: {self.model_name}")

        # Check model availability
        if not self._ensure_model_available(self.model_name):
            raise RuntimeError(f"Required model {self.model_name} is not available")
        
        # Get embedding model from config or use default embedding model
        self.embedding_model = (
            ollama_config.get("embedding_model") or
            DEFAULT_EMBEDDING_MODEL
        )
        print(f"Using embedding model: {self.embedding_model}")

        # Check embedding model availability
        if not self._ensure_model_available(self.embedding_model):
            raise RuntimeError(f"Required embedding model {self.embedding_model} is not available")

        # Initialize embeddings with dedicated embedding model
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)

        db_root = vectordb_config.get("db_root", DEFAULT_DB_ROOT)
        self.persist_directory = os.path.join(db_root, self.embedding_model.replace(":", "_"))
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)

        self.search_k = vectordb_config.get("search_k", 3)
        self.chunk_size = vectordb_config.get("chunk_size", 1000)
        self.chunk_overlap = vectordb_config.get("chunk_overlap", 200)

        track_file = filetracker_config.get("track_file", DEFAULT_TRACK_FILE)
        self.file_tracker = FileTracker(os.path.join(self.persist_directory, track_file))

        self.processor = DocumentProcessor(self.chunk_size, self.chunk_overlap)
        
        path_to_ids = {}
        if self.file_tracker.tracking_dict:
            # Remove the 'ids' key from the tracking dictionary
            for f, info in self.file_tracker.tracking_dict.items():
                path_to_ids[f] = info['ids']

        # Create/load vector store
        self.vector_store = VectorStore(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            search_k=self.search_k,
            path_to_ids=path_to_ids
        )
        
        # Initialize LLM
        self.llm = Ollama(model=self.model_name)
        
        # Create retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.get_retriever(),
            return_source_documents=True,
        )

    def load_documents(self, path: str):
        """Load single document or directory of documents"""
        if os.path.isdir(path):
            new_files, modified_files, deleted_files = self.file_tracker.get_modified_files(path)

            if not (new_files or modified_files or deleted_files):
                print("No new or modified files to process")
                return 0
            
            # Handle deleted files
            files_to_remove = [*deleted_files, *modified_files.keys()]
            if files_to_remove:
                print(f"Removing {len(files_to_remove)} deleted/modified files from vector store")
                self.vector_store.remove_documents(files_to_remove)
            
            # Process new and modified files
            files_to_process = {**new_files, **modified_files}
            if files_to_process:
                print(f"Processing {len(files_to_process)} new/modified files")
                documents_to_update = []
                num_chunks = 0
                for f, ts in files_to_process.items():
                    file_docs = self.processor.load_document(f)
                    documents_to_update.append(dict(file=f, timestamp=ts, docs=file_docs))
                    num_chunks += len(file_docs)
                print(f"Adding {num_chunks} document chunks to vector store")
                updated_path_ids = self.vector_store.add_documents(documents_to_update)
            
            files_to_update = [
                dict(file=f, timestamp=ts, ids=updated_path_ids[f]) for f, ts in files_to_process.items()
            ]
            
            self.file_tracker.update_tracking(files_to_update, deleted_files)
            return len(docs) if 'docs' in locals() else 0
        else:
            # Single file processing
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
                
            mod_time = os.path.getmtime(path)
            if path in self.file_tracker.tracking_data:
                if mod_time == self.file_tracker.tracking_data[path]:
                    print("File unchanged, skipping processing")
                    return 0
                    
            docs = self.processor.load_document(path)
            file_data = {path: mod_time}
            self.file_tracker.update_tracking(file_data)
            self.vector_store.add_documents(docs, [(path, mod_time)] * len(docs))        
            return len(docs)

    def query(self, question: str) -> str:
        """Query the RAG system"""
        response = self.qa_chain.invoke({"query": question})
        return response["result"]
    
    def __del__(self):
        """Cleanup when the RAG system is destroyed"""
        if hasattr(self, 'vector_store'):
            self.vector_store.cleanup()