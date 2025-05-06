import os
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredHTMLLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Use custom TextLoader with UTF-8 encoding
class UTF8TextLoader(TextLoader):
    def __init__(self, file_path: str):
        super().__init__(file_path, encoding='utf-8')

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.supported_extensions = {
            '.pdf': PyPDFLoader,
            '.html': UnstructuredHTMLLoader,
            '.htm': UnstructuredHTMLLoader,
            '.txt': UTF8TextLoader,
            '.md': UTF8TextLoader,
        }
        
    def load_document(self, file_path: str) -> List:
        """Load and split a document into chunks
        
        Args:
            file_path: Path to the document
            
        Raises:
            ValueError: If file type is not supported
            UnicodeDecodeError: If text file has encoding issues
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.supported_extensions:
            raise ValueError(
                f"Unsupported file type: {file_ext}\n"
                f"Supported types: {', '.join(self.supported_extensions.keys())}"
            )
            
        try:
            loader = self.supported_extensions[file_ext](file_path)
            doc = loader.load()
            return self.text_splitter.split_documents(doc)
        except UnicodeDecodeError:
            # Try with different encodings if UTF-8 fails
            try:
                if file_ext in ['.txt', '.md']:
                    loader = TextLoader(file_path, encoding='latin-1')
                    doc = loader.load()
                    return self.text_splitter.split_documents(doc)
            except Exception as e:
                raise ValueError(f"Unable to decode {file_path}. Error: {str(e)}")
    
    def load_all_documents(self, directory_path: str) -> List:
        all_docs = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    doc = self.load_document(file_path)
                    all_docs.extend(doc)
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
        return all_docs