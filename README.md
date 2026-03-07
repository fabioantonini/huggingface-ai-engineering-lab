# Hugging Face AI Engineering Lab

Hands-on Jupyter notebooks for building AI systems with open-source models from Hugging Face.

## Topics

- Transformers and NLP pipelines
- Embeddings and semantic search
- Retrieval-Augmented Generation (RAG)
- Chatbots and LLM agents
- Multimodal AI (images, vision-language models)
- Voice AI (speech recognition and synthesis)

## Project Structure

```
huggingface-ai-engineering-lab/
├── notebooks/                    # Core notebooks
│   ├── 01_huggingface_ecosystem.ipynb
│   └── 04_embeddings.ipynb
├── advanced_notebooks/           # Advanced topics
│   ├── 13_llm_agents.ipynb
│   ├── 14_multimodal_models.ipynb
│   └── 15_voice_ai.ipynb
├── src/                          # Reusable Python modules
│   ├── embeddings.py
│   ├── rag.py
│   └── vector_search.py
├── datasets/
│   └── example_documents.txt
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Getting Started

### Option 1 — Docker (recommended)

```bash
docker compose up --build
```

Open `http://localhost:8888` in your browser.
If prompted for a token, copy it from the container logs (look for a line containing `?token=`).

**GPU support:** the compose file passes all NVIDIA GPUs to the container. This requires:
- NVIDIA drivers installed on the host
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed (`nvidia-ctk`)

To verify the GPU is visible inside the container:
```bash
docker exec hf_ai_lab python -c "import torch; print(torch.cuda.is_available())"
```

### Option 2 — Local installation

```bash
pip install -r requirements.txt
jupyter lab
```

## Environment Variables

Create a `.env` file in the project root for optional configuration:

```env
HUGGING_FACE_HUB_TOKEN=hf_...   # Required for gated models
```

The Docker container sets `HF_HOME=/workspace/cache` to persist downloaded models inside the project directory.

## Building for Multiple Architectures (amd64 + arm64)

Docker's `buildx` allows building a single multi-platform image that works on both Intel/AMD machines and Apple Silicon (M1/M2/M3) or ARM servers.

### One-time setup — create a multi-platform builder

```bash
docker buildx create --name multiarch --driver docker-container --use
docker buildx inspect --bootstrap
```

### Build and push to Docker Hub

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t your-dockerhub-username/hf-ai-lab:latest \
  --push \
  .
```

Replace `your-dockerhub-username/hf-ai-lab` with your actual Docker Hub repository name.
`--push` is required for multi-platform builds — Docker cannot load both architectures into the local daemon simultaneously.

### Build a single platform locally (no push)

```bash
# For the current machine architecture:
docker buildx build --platform linux/amd64 -t hf-ai-lab:latest --load .
docker buildx build --platform linux/arm64 -t hf-ai-lab:latest --load .
```

### Notes

- `buildx` with `docker-container` driver uses QEMU emulation for cross-compilation — the arm64 build on an amd64 machine (and vice versa) will be slower than a native build.
- The base image `python:3.11` is available for both platforms on Docker Hub, so no changes to the Dockerfile are needed.
- After pushing, Docker Desktop automatically pulls the correct variant for the host architecture.

## Requirements

Key dependencies are listed in [requirements.txt](requirements.txt):

| Group | Packages |
|---|---|
| Hugging Face | `transformers`, `datasets`, `huggingface_hub`, `sentence-transformers`, `accelerate`, `diffusers` |
| Vector search | `faiss-cpu`, `sentence-transformers` |
| LLM agents | `langchain`, `langchain-community` |
| Voice AI | `openai-whisper`, `librosa`, `soundfile` |
| Vision | `opencv-python`, `Pillow` |
| UI / demos | `gradio`, `streamlit` |
| Backend | `torch`, `torchvision`, `torchaudio` |
