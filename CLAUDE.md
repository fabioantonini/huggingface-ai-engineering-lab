# CLAUDE.md — Hugging Face AI Engineering Lab

## Project Overview

Hands-on Jupyter notebooks for building AI systems with open-source Hugging Face models.
Topics: Transformers, NLP pipelines, embeddings, semantic search, RAG, chatbots, image classification, multimodal AI (CLIP, image captioning).

## Repository Structure

```
huggingface-ai-engineering-lab/
├── notebooks/                        # Core notebooks
│   ├── 00_course_index.ipynb         # Course overview and environment test
│   ├── 01_huggingface_ecosystem.ipynb
│   ├── 02_transformers_pipeline.ipynb
│   ├── 03_text_generation.ipynb
│   ├── 04_embeddings_similarity.ipynb
│   ├── 04_embeddings.ipynb           # Legacy embeddings notebook
│   ├── 05_huggingface_datasets.ipynb
│   ├── 06_fine_tuning_classifier.ipynb
│   └── 07_semantic_search.ipynb
├── advanced_notebooks/               # Advanced topics
│   ├── 10_rag_pipeline.ipynb
│   ├── 11_chatbot_transformers.ipynb
│   ├── 12_image_classification.ipynb
│   ├── 13_clip_multimodal.ipynb
│   └── 14_image_captioning.ipynb
├── src/                              # Reusable Python modules
│   ├── embeddings.py                 # SentenceTransformer wrapper
│   ├── rag.py                        # RAG pipeline
│   └── vector_search.py             # FAISS-based vector search
├── datasets/
│   └── example_documents.txt        # Sample dataset for RAG demos
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
| `transformers` | HF Transformers models and pipelines (NLP, vision, multimodal) |
| `sentence-transformers` | Sentence embeddings |
| `datasets` | HF Datasets loading/processing |
| `faiss-cpu` | Vector similarity search |
| `accelerate` | Distributed training / fine-tuning acceleration |
| `evaluate` | Model evaluation metrics |
| `scikit-learn` | ML utilities and metrics |
| `torch`, `torchvision` | PyTorch backend |
| `Pillow` | Image loading and processing |
| `numpy`, `pandas` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |

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
- Advanced notebooks (10–14) cover RAG, chatbots, image classification, CLIP, and image captioning; add cells as needed.
