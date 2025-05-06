import os
import argparse
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM as Ollama

os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

template = """Question: {question}

Answer: Let's think step by step."""

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test Ollama LLM with different models')
    parser.add_argument('--model', type=str, default="gemma3:1b",
                       help='Name of the Ollama model to use (default: gemma3:1b)')
    parser.add_argument('--question', type=str, default="What is LangChain?",
                       help='Question to ask the model (default: What is LangChain?)')
    args = parser.parse_args()

    model_name = args.model

    # Initialize the prompt and model
    prompt = ChatPromptTemplate.from_template(template)
    model = Ollama(model=model_name)
    chain = prompt | model
    
    print(f"Using model: {args.model}\n")
    print(f"Question: {args.question}\n")
    response = chain.invoke({"question": args.question})
    print("Response:", response)

    print(response)

if __name__ == "__main__":
    main()