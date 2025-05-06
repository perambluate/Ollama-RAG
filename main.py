from src.rag_system import RAGSystem

# Usage example
def main():
    try:
        # Initialize RAG system
        rag = RAGSystem()
        
        # Load documents
        docs_count = rag.load_documents("./materials")
        if docs_count > 0:
            print(f"Loaded {docs_count} new/modified document chunks")
        
        # Query system
        while True:
            question = input("\nEnter your question (or type 'quit/q/exit' to exit): ")
            if question.lower() in ('quit', 'q', 'exit'):
                break
                
            answer = rag.query(question)
            print("\nAnswer:", answer)
    finally:
        if 'rag' in locals():
            del rag

if __name__ == "__main__":
    main()