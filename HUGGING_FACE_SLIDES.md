---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
    font-size: 22px;
  }
  section.title {
    text-align: center;
    justify-content: center;
    background: #FFD21E;
    color: #1a1a1a;
  }
  section.title h1 { font-size: 2.4em; margin-bottom: 0.2em; }
  section.title p  { font-size: 1.1em; color: #333; }
  section.section-header {
    background: #1a1a1a;
    color: #FFD21E;
    justify-content: center;
  }
  section.section-header h2 { font-size: 2em; }
  h1, h2 { color: #1a1a1a; }
  h3 { color: #555; font-size: 0.95em; text-transform: uppercase; letter-spacing: 0.05em; }
  code { background: #f4f4f4; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; }
  pre  { background: #1e1e1e; color: #d4d4d4; padding: 16px; border-radius: 8px; font-size: 0.72em; }
  table { font-size: 0.82em; width: 100%; }
  th { background: #FFD21E; color: #1a1a1a; }
  blockquote { border-left: 4px solid #FFD21E; background: #fffde7; padding: 8px 16px; font-size: 0.9em; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 2em; }
---

<!-- _class: title -->

# 🤗 Hugging Face
## Platform Overview

A practical guide for AI Engineering students

---

## Agenda

1. What is Hugging Face?
2. The Model Hub
3. The Dataset Hub
4. HF Token — why and how
5. Serverless Inference API
6. Dedicated Endpoints
7. Spaces
8. `huggingface_hub` library
9. Quick Reference
10. Practical Tips

---

<!-- _class: section-header -->

## 1 · What is Hugging Face?

---

## The "GitHub of AI"

Hugging Face is the **leading open-source platform for AI** — a place where researchers, companies, and developers share models, datasets, and apps that anyone can use for free.

<br>

| Component | What it does |
|---|---|
| **Hub** | Repository of 900k+ models, 200k+ datasets, Spaces |
| **Transformers** | Python library to load and run any model |
| **Datasets** | Fast dataset loading and processing |
| **Inference API** | Free cloud endpoints — no GPU needed |
| **Spaces** | Free hosting for interactive demos |
| **Inference Endpoints** | Dedicated production endpoints (paid) |

---

## Why it became the standard

- **900,000+** public models (2025)
- **200,000+** public datasets
- Unified API — the same code works across very different models
- Supports **PyTorch**, **TensorFlow**, and **JAX**
- Active community, papers with code, public benchmarks

<br>

```python
from transformers import pipeline

# Three lines to run a state-of-the-art model
classifier = pipeline("sentiment-analysis")
result = classifier("Hugging Face is amazing!")
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

---

<!-- _class: section-header -->

## 2 · The Model Hub

---

## Finding a Model

Go to **huggingface.co/models** and filter by:

| Filter | Examples |
|---|---|
| **Task** | text-classification, text-generation, image-classification… |
| **Library** | Transformers, sentence-transformers, diffusers… |
| **Language** | en, it, multilingual… |
| **License** | MIT, Apache-2.0, cc-by-4.0… |
| **Sort by** | Trending, Most Downloads, Recently Updated |

<br>

> **Tip:** Start with "Most Downloads" to find the most battle-tested models for your task.

---

## Reading a Model Card

Every model page contains:

| Section | What you'll find |
|---|---|
| **Model description** | Architecture, training data, authors |
| **Intended uses** | What it was designed for — and what NOT to use it for |
| **Limitations** | Bias, supported languages, known issues |
| **Training data** | Which data, data license |
| **Usage** | Ready-to-run code snippets |
| **Files and versions** | Weights, tokenizer, config |

<br>

> ⚠️ Always read **"Limitations and bias"** before using a model in production.

---

## Using a Model — Method 1: `pipeline()`

**Best for:** rapid prototyping, standard tasks

```python
from transformers import pipeline

# Downloads model + tokenizer automatically on first run
classifier = pipeline("sentiment-analysis")
print(classifier("I love this course!"))
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Specify a model explicitly
generator = pipeline("text-generation", model="gpt2")
out = generator("The future of AI is", max_new_tokens=30)
print(out[0]["generated_text"])
```

---

## Using a Model — Method 2: `AutoModel`

**Best for:** direct access to logits, embeddings, custom forward passes

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model     = AutoModelForSequenceClassification.from_pretrained(model_name)

inputs = tokenizer("This is great!", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

probs    = torch.softmax(outputs.logits, dim=-1)
label_id = probs.argmax().item()
print(model.config.id2label[label_id])  # POSITIVE
```

---

## Using a Model — Method 3: CLI Download

**Best for:** offline work, inspecting files, scripting

```bash
# Download an entire model
huggingface-cli download gpt2

# Download specific files only
huggingface-cli download distilbert-base-uncased \
    config.json tokenizer.json

# Download to a specific directory
huggingface-cli download gpt2 --local-dir ./models/gpt2
```

---

## Most Common `Auto*` Classes

| Class | Use case |
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

<!-- _class: section-header -->

## 3 · The Dataset Hub

---

## Loading Datasets

```python
from datasets import load_dataset

# Public dataset
dataset = load_dataset("imdb")

# With a specific configuration (subset)
dataset = load_dataset("glue", "sst2")

# Single split
train = load_dataset("imdb", split="train")

# Streaming — no full download (great for huge datasets)
dataset = load_dataset("cc100", "it", streaming=True)

# From a local file
dataset = load_dataset("csv",  data_files="my_data.csv")
dataset = load_dataset("json", data_files="my_data.jsonl")
```

---

## Dataset Structure

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

---

## Core Dataset Operations

```python
train = dataset["train"]

# Filter rows
positives = train.filter(lambda x: x["label"] == 1)

# Add or transform columns
def add_length(ex):
    ex["length"] = len(ex["text"].split())
    return ex
train = train.map(add_length)

# Batched tokenization (much faster)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
train = train.map(
    lambda b: tokenizer(b["text"], truncation=True, max_length=128),
    batched=True
)

# Shuffle and sample
small = train.shuffle(seed=42).select(range(1000))
```

---

<!-- _class: section-header -->

## 4 · HF Token

---

## When Do You Need a Token?

| Scenario | Token required? |
|---|---|
| Public models (GPT-2, DistilBERT…) | No |
| Public datasets | No |
| **Gated models** (Llama 3, Gemma, Mistral…) | **Yes** |
| **Inference API** | **Yes** |
| Uploading to your profile | Yes (write) |
| Private repositories | Yes |

<br>

> "Gated" models require you to accept their terms of use on the model page — the token identifies your approval.

---

## Creating a Token (Step by Step)

1. Go to **huggingface.co** → sign in (or create a free account)
2. Click your profile picture → **Settings**
3. Left menu → **Access Tokens**
4. Click **"New token"**
5. Set a name (e.g. `lab-token`) and choose type:
   - `Read` — download models & datasets *(sufficient for 95% of cases)*
   - `Write` — upload models, datasets, Spaces
   - `Fine-grained` — granular permissions per repo
6. Click **"Generate a token"**
7. **Copy it immediately** — it won't be shown again

<br>

> ⚠️ Never commit your token to a code file. Use `.env` or environment variables.

---

## Using the Token — 5 Options

```python
# Option 1 — .env file (recommended for this lab)
# In .env:  HUGGING_FACE_HUB_TOKEN=hf_xxx...
from dotenv import load_dotenv; load_dotenv()

# Option 2 — programmatic login
from huggingface_hub import login
login(token="hf_xxx...")   # or login() for interactive prompt

# Option 3 — CLI
# $ huggingface-cli login

# Option 4 — shell environment variable
# $ export HUGGING_FACE_HUB_TOKEN=hf_xxx...

# Option 5 — Docker (this lab)
# Put it in .env — docker-compose.yml passes it to the container
```

---

## Requesting Access to Gated Models

1. Open the model page on huggingface.co
   *(e.g. `meta-llama/Meta-Llama-3-8B`)*
2. Click **"Request access"** or **"Agree and access repository"**
3. Fill in the form (name, organisation, intended use)
4. Wait for approval *(automatic for some models, manual for others)*
5. Once approved → download with your token

---

<!-- _class: section-header -->

## 5 · Serverless Inference API

---

## What Is It?

Run models **in Hugging Face's cloud** — no local GPU needed.

- **Free** tier with rate limits
- Interactive widget on every supported model page
- Variable response time (cold start on first call)

**When to use it:**
- Test a model without downloading its weights (~GB)
- No local GPU available
- Building a prototype or demo
- Integrating a model via HTTP

---

## InferenceClient (Recommended)

```python
from huggingface_hub import InferenceClient

client = InferenceClient(token="hf_xxx...")

# Text generation
result = client.text_generation("The capital of Italy is",
                                model="gpt2", max_new_tokens=20)

# Sentiment analysis
result = client.text_classification(
    "I love this!",
    model="distilbert-base-uncased-finetuned-sst-2-english")

# Question Answering
result = client.question_answering(
    question="Where is HF based?",
    context="Hugging Face is based in New York and Paris.",
    model="deepset/roberta-base-squad2")
print(result.answer)

# Summarization
result = client.summarization("Long text...", model="facebook/bart-large-cnn")
```

---

## Direct HTTP with `requests`

```python
import requests, time

API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": "Bearer hf_xxx..."}

def query(payload):
    r = requests.post(API_URL, headers=headers, json=payload)
    return r.json()

result = query({"inputs": "The future of AI is"})
```

**Common HTTP status codes:**

| Code | Meaning | Fix |
|---|---|---|
| 200 | OK | — |
| 401 | Invalid token | Regenerate token |
| 403 | Access denied | Request model access |
| 429 | Rate limit | Wait a few minutes |
| 503 | Model loading | Wait `estimated_time` seconds |

---

<!-- _class: section-header -->

## 6 · Dedicated Endpoints

---

## Serverless vs Dedicated

| | Serverless (free) | Dedicated (paid) |
|---|---|---|
| **Cost** | Free (with limits) | ~$0.06–$3 / hr per GPU |
| **Latency** | Variable, cold start | Low, guaranteed |
| **Availability** | Not guaranteed | 99.9% SLA |
| **Models** | HF-supported only | Any HF model |
| **GPU** | Shared | Dedicated |
| **Use** | Dev, testing, demos | Production |

<br>

> ⚠️ **Remember to pause or delete** your endpoint when not in use — you are billed per hour in `Running` state.

---

## Creating and Calling an Endpoint

**Create:** go to **huggingface.co/inference-endpoints** → New Endpoint → choose model, cloud, region, hardware → Create

<br>

**Call — same as Inference API:**

```python
import requests
from huggingface_hub import InferenceClient

endpoint = "https://abc123.us-east-1.aws.endpoints.huggingface.cloud"
headers  = {"Authorization": "Bearer hf_xxx..."}

# Option A — requests
r = requests.post(endpoint, headers=headers, json={"inputs": "Hello!"})

# Option B — InferenceClient
client = InferenceClient(model=endpoint, token="hf_xxx...")
result = client.text_generation("Hello!", max_new_tokens=50)
```

---

<!-- _class: section-header -->

## 7 · Spaces

---

## What Are Spaces?

Free hosting for **interactive AI demos** — typically built with Gradio or Streamlit.

- Browse thousands of demos at **huggingface.co/spaces**
- No installation needed — runs in the browser
- Clone locally to customise or study the code

**Spaces useful for this course:**
- Sentiment analysis demo
- Stable Diffusion (text-to-image)
- Whisper (speech-to-text)
- Text generation comparisons

```bash
# Clone a Space and run it locally
git clone https://huggingface.co/spaces/username/space-name
cd space-name && pip install -r requirements.txt
python app.py
```

---

<!-- _class: section-header -->

## 8 · `huggingface_hub` Library

---

## Model & Dataset Metadata

```python
from huggingface_hub import model_info, dataset_info

info = model_info("distilbert-base-uncased-finetuned-sst-2-english")
print(info.modelId)        # model name
print(info.pipeline_tag)  # sentiment-analysis
print(info.downloads)     # download count (last 30 days)
print(info.likes)         # number of likes
print(info.tags)          # list of tags
print(info.gated)         # False / 'auto' / 'manual'
```

---

## Search, Download, Upload

```python
from huggingface_hub import list_models, hf_hub_download, HfApi

# Search — top text-classification models by downloads
models = list(list_models(filter="text-classification",
                          sort="downloads", direction=-1, limit=5))

# Download a single file
path = hf_hub_download(repo_id="gpt2", filename="config.json")

# Download an entire repo
from huggingface_hub import snapshot_download
snapshot_download("distilbert-base-uncased", local_dir="./models/distilbert")

# Upload (write token required)
api = HfApi(token="hf_xxx...")
api.upload_folder(folder_path="./my_model",
                  repo_id="your-username/my-model")
```

---

<!-- _class: section-header -->

## 9 · Quick Reference

---

## Task → Model Cheatsheet

| Task | Pipeline tag | Recommended model |
|---|---|---|
| Sentiment analysis | `sentiment-analysis` | `distilbert-base-uncased-finetuned-sst-2-english` |
| NER | `ner` | `dbmdz/bert-large-cased-finetuned-conll03-english` |
| Extractive QA | `question-answering` | `deepset/roberta-base-squad2` |
| Summarization | `summarization` | `facebook/bart-large-cnn` |
| Translation EN→IT | `translation` | `Helsinki-NLP/opus-mt-en-it` |
| Text generation | `text-generation` | `gpt2` / `mistralai/Mistral-7B-v0.1` |
| Zero-shot classification | `zero-shot-classification` | `facebook/bart-large-mnli` |
| Sentence embeddings | — | `sentence-transformers/all-MiniLM-L6-v2` |
| Image classification | `image-classification` | `google/vit-base-patch16-224` |
| CLIP | `zero-shot-image-classification` | `openai/clip-vit-base-patch32` |
| Speech-to-text | `automatic-speech-recognition` | `openai/whisper-base` |

---

## Library → When to Use It

| Library | When to use it |
|---|---|
| `transformers.pipeline()` | Rapid prototyping, standard tasks |
| `transformers.AutoModel` | Direct access to logits / embeddings |
| `sentence-transformers` | Sentence embeddings for similarity & search |
| `datasets` | Loading and preprocessing datasets |
| `huggingface_hub` | Programmatic search, download, upload |
| `evaluate` | Standard metrics (accuracy, F1, BLEU…) |
| `accelerate` | Distributed training, mixed precision |

---

<!-- _class: section-header -->

## 10 · Practical Tips

---

## Model Cache

Downloaded models are cached locally — only downloaded **once**.

| Environment | Cache location |
|---|---|
| Local | `~/.cache/huggingface/hub/` |
| This lab (Docker) | `/workspace/cache/` (mounted volume) |

```python
import os
cache = os.environ.get("HF_HOME",
                       os.path.expanduser("~/.cache/huggingface/hub"))
print(f"Cache: {cache}")

# Clear cache interactively
# $ huggingface-cli delete-cache
```

---

## License Guide

| License | Commercial | Redistribution | Modification |
|---|---|---|---|
| **MIT** | ✅ | ✅ | ✅ |
| **Apache-2.0** | ✅ | ✅ | ✅ |
| **cc-by-4.0** | ✅ | ✅ | ✅ (attribution) |
| **cc-by-nc-4.0** | ❌ | ✅ | ✅ |
| **llama-3** | ✅ (restricted) | Limited | Limited |
| **gemma** | ✅ (restricted) | Limited | — |

<br>

> **Rule of thumb:** for academic / learning use, any license is fine.
> For commercial production → choose **MIT** or **Apache-2.0**.

---

## Common Errors & Fixes

| Error | Likely cause | Fix |
|---|---|---|
| `OSError: Can't load tokenizer` | Wrong model name | Check the ID on huggingface.co |
| `RepositoryNotFoundError` | Private or non-existent model | Check name, add token |
| `GatedRepoError` | No access to gated model | Request access on HF page |
| `OutOfMemoryError` | Model too large for GPU | Use CPU, FP16, or smaller model |
| Very slow download | First run, large weights | Normal — subsequent runs use cache |
| `RuntimeError: CUDA error` | GPU issue | Restart kernel, check CUDA drivers |
| Token not recognised | Env variable not loaded | Run `load_dotenv()`, check `.env` |

---

## Speeding Up Inference

```python
import torch

# Use GPU automatically
device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("sentiment-analysis", device=device)

# FP16 on GPU (halves memory usage, minimal accuracy drop)
model = AutoModel.from_pretrained(
    "bert-base-uncased",
    torch_dtype=torch.float16
)

# Batch processing — much faster than one example at a time
texts   = ["text 1", "text 2", "text 3", ...]
results = pipe(texts, batch_size=32)
```

---

<!-- _class: title -->

# Summary

| Topic | Key takeaway |
|---|---|
| **Hub** | 900k+ models & 200k+ datasets — always search here first |
| **pipeline()** | Fastest way to run any model |
| **Token** | Required for gated models & Inference API — keep it secret |
| **Inference API** | Free cloud inference — great for prototyping |
| **Endpoints** | Production-grade, paid, low-latency |
| **Spaces** | Browse & run demos instantly in the browser |
| **Cache** | Models download once, then run offline |

<br>

**Full reference:** `HUGGING_FACE_GUIDE.md`

---

<!-- _class: title -->

# 🤗 Happy Building!

**Lab notebooks:** `notebooks/` and `advanced_notebooks/`

**Reference guide:** `HUGGING_FACE_GUIDE.md`

**Docs:** huggingface.co/docs
