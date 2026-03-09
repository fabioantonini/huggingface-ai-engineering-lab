# Hugging Face AI Engineering Lab

Hands-on Jupyter notebooks for building AI systems with open-source models from Hugging Face.

## Topics

- Hugging Face ecosystem (Hub, Transformers, Datasets)
- NLP pipelines: sentiment, NER, QA, summarization, translation
- Text generation and decoding strategies
- Embeddings and semantic similarity
- Semantic search with FAISS
- Fine-tuning transformer classifiers
- Retrieval-Augmented Generation (RAG)
- Chatbots with DialoGPT
- Vision Transformers (ViT) for image classification
- Multimodal AI: CLIP and image captioning

## Project Structure

```
huggingface-ai-engineering-lab/
├── notebooks/                        # Core notebooks
│   ├── 00_course_index.ipynb         # Course roadmap and environment check
│   ├── 01_huggingface_ecosystem.ipynb
│   ├── 02_transformers_pipeline.ipynb
│   ├── 03_text_generation.ipynb
│   ├── 04_embeddings_similarity.ipynb
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
│   └── example_documents.txt        # Sample corpus for RAG and semantic search
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
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
| Hugging Face | `transformers`, `datasets`, `huggingface_hub`, `sentence-transformers`, `accelerate`, `evaluate` |
| Vector search | `faiss-cpu` |
| ML utilities | `scikit-learn`, `numpy`, `pandas` |
| Vision | `Pillow` |
| Visualization | `matplotlib`, `seaborn` |
| Backend | `torch`, `torchvision` |
| Notebook | `jupyterlab`, `ipywidgets` |
| Utilities | `python-dotenv`, `tqdm`, `requests` |
