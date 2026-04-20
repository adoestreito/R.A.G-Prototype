# R.A.G Prototype

Small **retrieval-augmented** chat app: **FastAPI** UI, **Chroma** + sentence-transformers embeddings, **LangChain** + **Ollama** for answers. Ingest either the [m-ric/huggingface_doc](https://huggingface.co/datasets/m-ric/huggingface_doc) dataset or a local clone of [mdn/content](https://github.com/mdn/content).

## Requirements

- Python 3.10+ (3.9 may work; project targets 3.10+)
- [Ollama](https://ollama.com/) running locally, with your chat model pulled (e.g. `ollama pull gemma4:e2b`)

## Setup

```bash
cd RAG-2
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: OLLAMA_MODEL, and MDN or HF ingest vars if you change defaults
```

Build the vector index and topic tiles (needed before `app` will serve chat):

```bash
python ingest.py
```

Start the server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Configuration notes

- Copy **`.env.example` → `.env`**; never commit `.env` (see `.gitignore`).
- **`INGEST_SOURCE`**: `hf` (default in example) or `mdn` with `MDN_CONTENT_ROOT` pointing at the **repo root** of your mdn/content clone and optional `MDN_PATH_PREFIX` to limit folders.
- After changing corpus or ingest settings, delete `.chroma` if you need a clean re-index, then run `ingest.py` again.
- User notes from the UI are stored in a separate Chroma collection (`user_notes`) alongside the main doc index.


