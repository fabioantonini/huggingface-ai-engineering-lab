# CLAUDE.md — Hugging Face AI Engineering Lab

## Project Overview

Hands-on Jupyter notebooks for building AI systems with open-source Hugging Face models.
Topics: Transformers, NLP pipelines, embeddings, semantic search, RAG, chatbots, multimodal AI, voice AI, LLM agents.

## Repository Structure

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
│   ├── embeddings.py             # SentenceTransformer wrapper
│   ├── rag.py                    # RAG pipeline
│   └── vector_search.py         # FAISS-based vector search
├── datasets/
│   └── example_documents.txt    # Sample dataset for RAG demos
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── CLAUDE.md
```

## Running the Lab

### Docker (recommended)

```bash
docker compose up --build
```

Then open `http://localhost:8888` — check the container logs for the Jupyter access token if prompted.

### Local

```bash
pip install -r requirements.txt
jupyter lab
```

## Key Dependencies

| Package | Purpose |
|---|---|
| `transformers` | HF Transformers models and pipelines |
| `sentence-transformers` | Sentence embeddings |
| `datasets` | HF Datasets loading/processing |
| `faiss-cpu` | Vector similarity search |
| `langchain`, `langchain-community` | LLM agent orchestration |
| `diffusers` | Multimodal / image generation models |
| `openai-whisper` | Speech recognition (voice AI) |
| `gradio`, `streamlit` | Demo UIs |
| `torch`, `torchvision`, `torchaudio` | PyTorch backend |

## Environment Variables

| Variable | Description |
|---|---|
| `HF_HOME` | Hugging Face cache directory (default: `/workspace/cache` in Docker) |
| `HUGGING_FACE_HUB_TOKEN` | Optional: HF token for gated models |

Set variables in a `.env` file at the project root (loaded via `python-dotenv`).

## Development Notes

- `src/` modules are meant to be imported inside notebooks; add the project root to `sys.path` if needed.
- The Docker volume mount (`.:/workspace`) syncs local changes into the container in real time — no rebuild needed for notebook edits.
- FAISS index is built in memory; persistence is not implemented in the current `vector_search.py`.
- Advanced notebooks (13–15) are scaffolded but may be partially filled; add cells as needed.
