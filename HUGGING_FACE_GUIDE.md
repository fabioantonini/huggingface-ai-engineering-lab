# Hugging Face — Student Reference Guide

This document is your go-to reference for everything related to the Hugging Face platform. Keep it open while working through the lab notebooks, or consult it whenever you need a quick reminder.

---

## Table of Contents

1. [What is Hugging Face](#1-what-is-hugging-face)
2. [The Model Hub](#2-the-model-hub)
3. [The Dataset Hub](#3-the-dataset-hub)
4. [Creating and Using an HF Token](#4-creating-and-using-an-hf-token)
5. [Serverless Inference API](#5-serverless-inference-api)
6. [Dedicated Endpoints](#6-dedicated-endpoints)
7. [Spaces](#7-spaces)
8. [The `huggingface_hub` Library](#8-the-huggingface_hub-library)
9. [Quick Reference Table](#9-quick-reference-table)
10. [Practical Tips](#10-practical-tips)

---

## 1. What is Hugging Face

Hugging Face is the leading open-source platform for AI. Think of it as the "GitHub of AI": a place where researchers, companies, and developers publish models, datasets, and applications that anyone can use for free.

### Main Components

| Component | What it does | URL |
|---|---|---|
| **Hub** | Repository of models, datasets, and Spaces | huggingface.co |
| **Transformers** | Python library to load and run models | `pip install transformers` |
| **Datasets** | Library to load and process datasets | `pip install datasets` |
| **Tokenizers** | Fast tokenization library | included in Transformers |
| **Inference API** | Free cloud endpoints to run models | Via REST API |
| **Spaces** | Free hosting for interactive demos | huggingface.co/spaces |
| **Inference Endpoints** | Dedicated production endpoints (paid) | huggingface.co/docs/inference-endpoints |

### Why it became the standard

- Over **900,000 public models** (as of 2025)
- Over **200,000 public datasets**
- Supports PyTorch, TensorFlow, and JAX
- Unified API: the same code works across very different models
- Active community, papers with code, public benchmarks

---

## 2. The Model Hub

### Finding a model

Go to **huggingface.co/models** and use the left-side filters:

- **Task** — select the problem type (text-classification, text-generation, image-classification, etc.)
- **Library** — filter by Transformers, sentence-transformers, etc.
- **Language** — filter by language (it, en, multilingual…)
- **License** — important for commercial use (MIT, Apache-2.0, cc-by-4.0, llama-…)
- **Sort by** — Trending, Most Downloads, Recently Updated

### Reading a Model Card

Every model has a page containing:

| Section | What you'll find |
|---|---|
| **Model description** | Architecture, training data, authors |
| **Intended uses** | What it was trained for and what NOT to use it for |
| **Limitations** | Bias, supported languages, known limitations |
| **Training data** | Which data was used, data license |
| **Usage** | Ready-to-run code snippets |
| **Files and versions** | Model weights, tokenizer, config |

> **Tip:** The "Limitations and bias" section is critical — always read it before using a model in production.

### Using a model: 3 methods

#### Method 1 — `pipeline()` (simplest)

Ideal for rapid prototyping. Downloads the model and tokenizer and handles everything automatically.

```python
from transformers import pipeline

# The model is downloaded automatically on first run
classifier = pipeline("sentiment-analysis")
result = classifier("Hugging Face is amazing!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Specify a particular model
generator = pipeline("text-generation", model="gpt2")
out = generator("The future of AI is", max_new_tokens=30)
print(out[0]["generated_text"])
```

#### Method 2 — `AutoModel` + `AutoTokenizer` (more control)

Ideal when you need direct access to logits, embeddings, or custom forward passes.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "This library is incredibly useful!"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
probs = torch.softmax(logits, dim=-1)
label_id = probs.argmax().item()
print(model.config.id2label[label_id])  # POSITIVE
```

#### Method 3 — Manual download with CLI

Useful for offline work, copying weights, or inspecting files.

```bash
# Install the CLI
pip install huggingface_hub

# Download a model to the current directory
huggingface-cli download gpt2

# Download specific files only
huggingface-cli download distilbert-base-uncased config.json tokenizer.json

# Download to a specific directory
huggingface-cli download gpt2 --local-dir ./my_models/gpt2
```

### Most common `Auto*` classes

| Class | When to use it |
|---|---|
| `AutoTokenizer` | Any tokenizer |
| `AutoModel` | Raw output (embeddings, logits) |
| `AutoModelForSequenceClassification` | Text classification |
| `AutoModelForTokenClassification` | NER, POS tagging |
| `AutoModelForQuestionAnswering` | Extractive QA |
| `AutoModelForCausalLM` | Text generation (GPT-style) |
| `AutoModelForSeq2SeqLM` | Translation, summarization |
| `AutoModelForImageClassification` | Image classification |

---

## 3. The Dataset Hub

### Finding a dataset

Go to **huggingface.co/datasets** and use the filters:

- **Task** — problem type
- **Language** — text language
- **License** — important for commercial use
- **Size** — from < 1K to > 1M examples

### Loading a dataset

```python
from datasets import load_dataset

# Basic public dataset
dataset = load_dataset("imdb")

# Dataset with a specific configuration (subset)
dataset = load_dataset("glue", "sst2")
dataset = load_dataset("common_voice", "it")  # Italian Common Voice

# Single split only
train = load_dataset("imdb", split="train")

# Streaming — does not load everything into memory (ideal for huge datasets)
dataset = load_dataset("cc100", "it", streaming=True)
for example in dataset["train"]:
    print(example)
    break

# From a local file
dataset = load_dataset("csv", data_files="my_data.csv")
dataset = load_dataset("json", data_files="my_data.jsonl")
dataset = load_dataset("text", data_files="my_data.txt")
```

### Dataset structure

```python
dataset = load_dataset("imdb")

print(dataset)
# DatasetDict({
#     train: Dataset({features: ['text', 'label'], num_rows: 25000}),
#     test:  Dataset({features: ['text', 'label'], num_rows: 25000})
# })

print(dataset["train"].features)
# {'text': Value('string'), 'label': ClassLabel(names=['neg', 'pos'])}

print(dataset["train"][0])
# {'text': 'I love this movie...', 'label': 1}
```

### Core operations

```python
train = dataset["train"]

# Filter
positives = train.filter(lambda x: x["label"] == 1)

# Map (add or transform columns)
def add_length(example):
    example["length"] = len(example["text"].split())
    return example

train = train.map(add_length)

# Batched map (much faster for tokenization)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train = train.map(tokenize, batched=True, batch_size=64)

# Shuffle and select (sampling)
small = train.shuffle(seed=42).select(range(1000))

# Rename columns
train = train.rename_column("label", "labels")

# Set PyTorch format
train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Save to disk (to avoid re-downloading)
train.save_to_disk("./my_cached_dataset")
train = load_from_disk("./my_cached_dataset")
```

---

## 4. Creating and Using an HF Token

### When you need a token

| Scenario | Token required? |
|---|---|
| Using public models (e.g. GPT-2, DistilBERT) | No |
| Using public datasets | No |
| **Using gated models** (Llama 3, Gemma, Mistral) | **Yes** |
| **Using the Inference API** | **Yes** |
| Uploading models/datasets to your profile | Yes (write) |
| Accessing private repositories | Yes |

"Gated" models require you to accept their terms of use on the model page before downloading — the token identifies your approval.

### How to create a token (step by step)

1. Go to **huggingface.co** and sign in (or create a free account)
2. Click your profile picture → **Settings**
3. In the left-side menu → **Access Tokens**
4. Click **"New token"**
5. Choose:
   - **Name**: a descriptive name (e.g. `lab-token`, `personal-read`)
   - **Type**:
     - `Read` — to download models and datasets (sufficient for 95% of use cases)
     - `Write` — to upload models, datasets, Spaces
     - `Fine-grained` — for granular permissions on specific repos
6. Click **"Generate a token"**
7. **Copy the token immediately** — it will not be visible again after closing the page

> ⚠️ **Security:** Never commit your token to a code file. Always use environment variables or a `.env` file excluded from git.

### How to use the token

#### Option 1 — `.env` file (recommended for this lab)

Create a `.env` file in the project root:

```env
HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

Then in your code:

```python
from dotenv import load_dotenv
import os

load_dotenv()
token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
```

The notebooks in this lab automatically load environment variables if `python-dotenv` is installed.

#### Option 2 — Programmatic login

```python
from huggingface_hub import login

login(token="hf_xxxxxxxxxxxxxxxxxxxx")
# or interactive (prompts for the token via input):
login()
```

After logging in, the token is saved to `~/.cache/huggingface/token` and used automatically.

#### Option 3 — CLI

```bash
huggingface-cli login
# Enter your token when prompted
```

#### Option 4 — Shell environment variable

```bash
export HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
python my_script.py
```

#### Option 5 — Docker (this lab)

Place the token in the `.env` file at the project root — `docker-compose.yml` automatically passes it into the container:

```env
HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

### Requesting access to gated models

1. Go to the model page on huggingface.co (e.g. `meta-llama/Meta-Llama-3-8B`)
2. Click **"Request access"** or **"Agree and access repository"**
3. Fill in the form (name, organisation, intended use)
4. Wait for approval (some models are approved automatically, others manually)
5. Once approved, download the model using your token

---

## 5. Serverless Inference API

### What it is

The Inference API lets you run models on Hugging Face's cloud without a local GPU. It is **free** but has limits (rate limiting, variable response times, not all models supported).

You can find the interactive widget directly on the page of every supported model.

### When to use it

- You want to test a model without downloading it (~GB of weights)
- You don't have a local GPU
- You are building a prototype or demo
- You want to integrate a model into an app via HTTP calls

### Method 1 — `InferenceClient` (recommended)

```python
from huggingface_hub import InferenceClient

client = InferenceClient(token="hf_xxxxxxxxxxxxxxxxxxxx")

# Text generation
result = client.text_generation(
    "The capital of Italy is",
    model="gpt2",
    max_new_tokens=20,
)
print(result)

# Sentiment analysis
result = client.text_classification(
    "I love this product!",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)
print(result)
# [ClassificationOutput(label='POSITIVE', score=0.9998)]

# Named Entity Recognition
result = client.token_classification(
    "Hugging Face was founded in New York.",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
)

# Question Answering
result = client.question_answering(
    question="Where is Hugging Face based?",
    context="Hugging Face is a company based in New York and Paris.",
    model="deepset/roberta-base-squad2",
)
print(result.answer)

# Image classification
from PIL import Image
result = client.image_classification("path/to/image.jpg")

# Summarization
result = client.summarization(
    "Long article text here...",
    model="facebook/bart-large-cnn",
)
print(result.summary_text)
```

### Method 2 — `requests` (direct HTTP)

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": "Bearer hf_xxxxxxxxxxxxxxxxxxxx"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

result = query({"inputs": "The future of AI is"})
print(result)
```

### Handling common errors

```python
import time
import requests

def query_with_retry(api_url, headers, payload, max_retries=3):
    for attempt in range(max_retries):
        response = requests.post(api_url, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()

        elif response.status_code == 503:
            # Model is still loading — wait
            wait = response.json().get("estimated_time", 20)
            print(f"Model loading... waiting {wait:.0f}s (attempt {attempt+1})")
            time.sleep(wait)

        elif response.status_code == 429:
            # Rate limited — back off before retrying
            print("Rate limited. Waiting 60s...")
            time.sleep(60)

        else:
            print(f"Error {response.status_code}: {response.text}")
            break

    return None
```

| Code | Meaning | Solution |
|---|---|---|
| 200 | OK | — |
| 400 | Invalid input | Check the payload |
| 401 | Invalid token | Regenerate your token |
| 403 | Access denied | Request access to the gated model |
| 429 | Rate limit exceeded | Wait a few minutes |
| 503 | Model loading | Wait `estimated_time` seconds and retry |

---

## 6. Dedicated Endpoints

### Difference from Serverless

| | Serverless (free) | Dedicated Endpoint (paid) |
|---|---|---|
| **Cost** | Free (with limits) | Pay-per-use (~$0.06–$3/hr per GPU) |
| **Latency** | Variable, cold start | Low, guaranteed |
| **Availability** | Not guaranteed | 99.9% SLA |
| **Models** | Only HF-supported ones | Any HF model |
| **GPU** | Shared | Dedicated |
| **Recommended for** | Development, testing, demos | Production |

### How to create an Endpoint

1. Go to **huggingface.co/inference-endpoints**
2. Click **"New Endpoint"**
3. Select the model from the Hub
4. Choose:
   - **Cloud provider** (AWS, Azure, GCP)
   - **Region** (close to your users)
   - **Hardware** (CPU, GPU T4, A10G, A100…)
   - **Scaling** (min/max instances, autoscaling)
5. Click **"Create Endpoint"**
6. Wait for the status to become `Running`

### How to call an Endpoint

Once created, you receive a URL like:
`https://abc123xyz.us-east-1.aws.endpoints.huggingface.cloud`

```python
import requests

endpoint_url = "https://your-endpoint-url.huggingface.cloud"
headers = {"Authorization": "Bearer hf_xxxxxxxxxxxxxxxxxxxx"}

response = requests.post(
    endpoint_url,
    headers=headers,
    json={"inputs": "Hello, world!"}
)
print(response.json())
```

With `InferenceClient`:

```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="https://your-endpoint-url.huggingface.cloud",
    token="hf_xxxxxxxxxxxxxxxxxxxx"
)
result = client.text_generation("Hello, world!", max_new_tokens=50)
```

> **Cost reminder:** remember to **pause or delete** your endpoint when not in use — you are billed for every hour it stays in `Running` state.

---

## 7. Spaces

### What a Space is

A Space is an interactive web application hosted for free by Hugging Face. Typically built with **Gradio** or **Streamlit**, it lets you demo a model without writing any frontend code.

You can find thousands of demos at **huggingface.co/spaces**.

### Using an existing Space

Simply open the page on huggingface.co/spaces and use the web interface — no installation needed.

Some Spaces useful for this course:
- **Sentiment analysis demo** — to test classification models
- **Stable Diffusion** — text-to-image
- **Whisper** — speech-to-text
- **Text generation** — to compare different LLMs

### Cloning a Space locally

```bash
git clone https://huggingface.co/spaces/username/space-name
cd space-name
pip install -r requirements.txt
python app.py  # or: gradio app.py
```

### Creating your own Space (out of scope for this lab)

If you want to build a Space in the future:
1. Go to huggingface.co/new-space
2. Choose an SDK: Gradio, Streamlit, or Docker
3. Upload your files (`app.py`, `requirements.txt`)
4. The Space starts automatically

---

## 8. The `huggingface_hub` Library

This library provides full programmatic access to the Hub — useful when you want to automate searches, downloads, or uploads without using the browser.

```python
from huggingface_hub import (
    model_info, dataset_info,
    list_models, list_datasets,
    hf_hub_download, snapshot_download,
    HfApi, login
)
```

### Model and dataset metadata

```python
from huggingface_hub import model_info, dataset_info

# Details for a model
info = model_info("distilbert-base-uncased-finetuned-sst-2-english")
print(info.modelId)       # model name
print(info.pipeline_tag) # sentiment-analysis
print(info.downloads)    # download count (last 30 days)
print(info.likes)        # number of likes
print(info.tags)         # list of tags
print(info.library_name) # transformers

# Details for a dataset
info = dataset_info("imdb")
print(info.downloads)
print(info.tags)
```

### Programmatic search

```python
from huggingface_hub import list_models, list_datasets

# Top text-classification models sorted by downloads
models = list(list_models(
    filter="text-classification",
    sort="downloads",
    direction=-1,
    limit=10
))
for m in models:
    print(f"{m.downloads:>10,}  {m.modelId}")

# Models in Italian
it_models = list(list_models(language="it", limit=5))

# QA datasets
qa_datasets = list(list_datasets(filter="question-answering", limit=5))
```

### Downloading files and repositories

```python
from huggingface_hub import hf_hub_download, snapshot_download

# Download a single file
path = hf_hub_download(
    repo_id="gpt2",
    filename="config.json",
    local_dir="./my_cache"
)
print(path)  # local path of the downloaded file

# Download an entire repository
local_dir = snapshot_download(
    repo_id="distilbert-base-uncased",
    local_dir="./models/distilbert"
)

# With token (for gated models)
local_dir = snapshot_download(
    repo_id="meta-llama/Meta-Llama-3-8B",
    token="hf_xxxxxxxxxxxxxxxxxxxx",
    local_dir="./models/llama3"
)
```

### Advanced operations with `HfApi`

```python
from huggingface_hub import HfApi

api = HfApi(token="hf_xxxxxxxxxxxxxxxxxxxx")

# Upload a file to your repository
api.upload_file(
    path_or_fileobj="./my_model/config.json",
    path_in_repo="config.json",
    repo_id="your-username/my-model",
)

# Upload an entire folder
api.upload_folder(
    folder_path="./my_model",
    repo_id="your-username/my-model",
)

# Create a new repository
api.create_repo(repo_id="my-new-model", private=False)

# List your repositories
repos = api.list_models(author="your-username")
```

---

## 9. Quick Reference Table

### Task → Pipeline tag → Recommended model

| Task | Pipeline tag | Recommended model | Notes |
|---|---|---|---|
| Sentiment analysis | `sentiment-analysis` | `distilbert-base-uncased-finetuned-sst-2-english` | Fast, accurate |
| NER | `ner` | `dbmdz/bert-large-cased-finetuned-conll03-english` | English |
| Extractive QA | `question-answering` | `deepset/roberta-base-squad2` | Extracts answer from text |
| Summarization | `summarization` | `facebook/bart-large-cnn` | News, articles |
| Translation EN→IT | `translation` | `Helsinki-NLP/opus-mt-en-it` | Fast |
| Translation EN→FR | `translation` | `Helsinki-NLP/opus-mt-en-fr` | Fast |
| Text generation | `text-generation` | `gpt2` (small), `mistralai/Mistral-7B-v0.1` (large) | GPT-2 requires no token |
| Zero-shot classification | `zero-shot-classification` | `facebook/bart-large-mnli` | No training required |
| Sentence embeddings | — | `sentence-transformers/all-MiniLM-L6-v2` | Fast, 384 dims |
| Sentence embeddings (better) | — | `sentence-transformers/all-mpnet-base-v2` | 768 dims, more accurate |
| Image classification | `image-classification` | `google/vit-base-patch16-224` | ImageNet 1k classes |
| Object detection | `object-detection` | `facebook/detr-resnet-50` | — |
| Image captioning | `image-to-text` | `nlpconnect/vit-gpt2-image-captioning` | — |
| CLIP | `zero-shot-image-classification` | `openai/clip-vit-base-patch32` | Text + image |
| Chatbot | `conversational` | `microsoft/DialoGPT-medium` | Conversational |
| Speech-to-text | `automatic-speech-recognition` | `openai/whisper-base` | Multilingual |

### Library → When to use it

| Library | When to use it |
|---|---|
| `transformers.pipeline()` | Rapid prototyping, standard tasks |
| `transformers.AutoModel` | Direct access to logits/embeddings |
| `sentence-transformers` | Sentence embeddings for similarity/search |
| `datasets` | Loading and preprocessing datasets |
| `huggingface_hub` | Programmatic search, download, upload |
| `evaluate` | Standard metrics (accuracy, F1, BLEU…) |
| `accelerate` | Distributed training, mixed precision |

### Useful Links

| Resource | URL |
|---|---|
| Hub (models) | huggingface.co/models |
| Hub (datasets) | huggingface.co/datasets |
| Hub (Spaces) | huggingface.co/spaces |
| Transformers docs | huggingface.co/docs/transformers |
| Datasets docs | huggingface.co/docs/datasets |
| huggingface_hub docs | huggingface.co/docs/huggingface_hub |
| Inference API docs | huggingface.co/docs/api-inference |
| Inference Endpoints docs | huggingface.co/docs/inference-endpoints |
| Papers With Code | paperswithcode.com |
| Community forum | discuss.huggingface.co |

---

## 10. Practical Tips

### Managing the model cache

Downloaded models are stored in:
- **Local default:** `~/.cache/huggingface/hub/`
- **In this lab (Docker):** `/workspace/cache/` (mounted as a volume)

```python
import os

# Check where the cache is
cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
print(f"Cache: {cache}")

# Clear the cache from the terminal
# huggingface-cli delete-cache

# Use a custom cache directory
os.environ["HF_HOME"] = "/path/to/my/cache"
```

Models are only downloaded on the first run — subsequent runs read from the cache and are much faster.

### Checking whether a model is gated

From the model page on huggingface.co:
- If you see a **"Request access"** button or a form before downloading → the model is gated
- If you see the files directly in the "Files" tab → you can download freely

From the library:

```python
from huggingface_hub import model_info

info = model_info("meta-llama/Meta-Llama-3-8B")
print(info.gated)  # 'auto', 'manual', or False
```

### Reading licenses

Before using a model in a real project, check its license:

| License | Commercial use | Redistribution | Modification |
|---|---|---|---|
| **MIT** | ✅ | ✅ | ✅ |
| **Apache-2.0** | ✅ | ✅ | ✅ |
| **cc-by-4.0** | ✅ | ✅ | ✅ (with attribution) |
| **cc-by-nc-4.0** | ❌ | ✅ | ✅ |
| **llama-2** | ✅ (with restrictions) | Limited | Limited |
| **llama-3** | ✅ (with restrictions) | Limited | Limited |
| **gemma** | ✅ (with restrictions) | Limited | — |

> **Practical rule:** for academic and learning purposes, any license is fine. For commercial production, stick to MIT or Apache-2.0 models.

### Model versioning

Every model has a `main` branch by default, but you can pin an exact revision to ensure reproducibility:

```python
from transformers import AutoModel

# Specific version (commit hash or tag)
model = AutoModel.from_pretrained(
    "bert-base-uncased",
    revision="a265f773a47193eed794233aa2a0f0bb6d3aaaf5"
)
```

### Speeding up inference

```python
import torch

# Use GPU if available
device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("sentiment-analysis", device=device)

# Mixed precision (FP16) on GPU
model = AutoModel.from_pretrained("bert-base-uncased", torch_dtype=torch.float16)

# Batch processing — much faster than single examples
texts = ["text 1", "text 2", "text 3", ...]
results = pipe(texts, batch_size=32)
```

### Common errors and solutions

| Error | Likely cause | Solution |
|---|---|---|
| `OSError: Can't load tokenizer` | Wrong model name | Verify the ID on huggingface.co |
| `RepositoryNotFoundError` | Private or non-existent model | Check the name, add a token |
| `GatedRepoError` | Gated model without access | Request access on the HF model page |
| `OutOfMemoryError` | Model too large for the GPU | Use CPU, FP16, or a smaller model |
| Slow download | First run — large weights | Normal; subsequent runs use the cache |
| `RuntimeError: CUDA error` | GPU issue | Restart the kernel, check CUDA drivers |
| Token not recognised | Env variable not loaded | Run `load_dotenv()` or check `.env` |
