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
