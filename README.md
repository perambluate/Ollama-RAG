# Ollama RAG

A project to build a simple RAG system via Ollama.

## Prerequisite
### Hardware Requirements
- Minimum 8GB RAM (16GB recommended)
- NVIDIA GPU with at least 4GB VRAM

### Recommended Models
For GPU with 4GB VRAM, use one of these models:

#### DeepSeek
- `deepseek-r1:1.5b` (1.5b)

#### Alibaba Cloud
- `qwen` (0.5b, 1.8b, 4b)
- `qwen2` (0.5b, 1.8b)
- `qwen2.5` (0.5b, 1.5b, 3b)
- `qwen3` (0.6b, 1.7b, 4b)

#### Google's models
- `gemma` (2b)
- `gemma2` (2b)
- `gemma3` (1b, 4b)

#### Meta's models
- `tinyllama` (1.1b)
- `llama3.2` (1b, 3b)

#### Microsoft's models
- `phi` (2.7b)
- `phi3` (3.8b)
- `llava-phi3` (3.8b)
- `phi3.5` (3.8b)
- `phi4-mini` (3.8b)
- `phi4-mini-reasoning` (3.8b)

#### IBM's models
- `granite3-moe` (1b, 3b)
- `granite3.1-moe` (1b, 3b)
- `granite3.2` (2b)
- `granite3.3` (2b)

#### Others
- `tinydolphin` (1.1b)
- `mistral-small` (2.2b)
- `dolphin-phi` (2.7b)
- `cogito` (3b)

### Docker GPU
Install docker with GPU support. See https://docs.docker.com/desktop/features/gpu/.

### Python
#### Tested Python version
Python=3.11

## Configuration (with conda env)
### Conda
Set up conda environment:
```bash
conda env create -n ${env_name} python=3.11
conda activate ${env_name}
pip install -r requirements.txt
```

Change `${env_name}` to the name you want.

### Configuring RAG system
```shell
mv ./config/default.yaml.template ./config/default.yaml
```

Configure RAG system in `./config/default.yaml`:
```yaml
ollama:
  base_url: "http://localhost:11434" # url running ollama service
  model: "gemma3:1b" # LLM model available in ollama lib
  embedding_model: "nomic-embed-text" # Embedding model available in ollama lib
vectordb:
  db_root: "./data/vectordb" # Root directory of vector db
  search_k: 3 # Increase to get more context chunks
  chunk_size: 1000 # Size of each chunk in tokens
  chunk_overlap: 200 # Overlap between chunks in tokens
file_tracker:
  track_file: ".file_tracking.json" # The name of the file recording the materials info
```

### Setting materials for querying
Create a folder to save materials
```shell
mkdir materials/
```

Create a folder for vector DB
```shell
mkdir data/vactordb/
```

Move materials (documents with `.pdf`, `.html`, `.htm`, `.txt`, `.md` extension, or directories including above-metioned document types) into `materials/`.
 
Sample materials are provided in `materials/samples/`.

### Vector Store Organization
Embeddings are stored in model-specific directories:
```
data/
└── vectordb/
    ├── qwen3_1.7b/
    ├── gemma3_1b/
    ├── deepseek-r1_1.5b/
    └── nomic-embed-text/
```

Changing models will create a new embeddings directory, preserving existing embeddings.

## Quick Use
### Start Ollama container

Start a docker container running Ollama
```shell
docker run -d --gpus=all --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --env CUDA_MEMORY_FRACTION=0.8 -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

Pull a LLM model (choose `gemma3:1b` as example)
```bash
docker exec -it ollama ollama pull gemma3:1b
```

Run the LLM model
```shell
docker exec -it ollama ollama run gemma3:1b
```

Test the connection to Ollama
```shell
python tests/testOllama.py --model gemma3:1b
```

### Start conversation
```shell
python main.py
```

You will see the following prompt
```shell
Enter your question (or type 'quit/q/exit' to exit):
```

## Use Ollama with Open WebUI interface (through Docker network)
1. Create a Docker network
```shell
docker network create ollama-network
```

2. Start Ollama container
```shell
docker run -d --gpus=all --network ollama-network --name ollama --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --env CUDA_MEMORY_FRACTION=0.8 -v ollama:/root/.ollama -p 11434:11434 ollama/ollama
```

3. Start Open WebUI container
```shell
docker run -d --network ollama-network --name open-webui -p 3000:8080 -e OLLAMA_BASE_URL=http://ollama:11434 -v open-webui:/app/backend/data --restart always ghcr.io/open-webui/open-webui:main
```

4. Access the web interface at `http://localhost:3000`

### Docker network debugging

If you need to verify container connectivity:

- Check network status:
```shell
docker network inspect ollama-network
```

- Test container communication:
```shell
# Install wget in the open-webui container before testing
docker exec -it open-webui sh -c "apt update && apt install -y wget"
docker exec -it open-webui wget -qO- http://ollama:11434/api/version
```

- View container logs:
```powershell
docker logs open-webui
docker logs ollama
```

These commands help diagnose connection issues between containers.