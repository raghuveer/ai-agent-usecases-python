# UC1 — RAG (raw-api)

Retrieval-augmented QA over a tiny Northwind Robotics knowledge base, built the
**raw-api** way: the `openai` SDK pointed at the gateway, with retrieval, prompt
building, and the chat call all hand-written so you can see exactly what is sent.

## What it demonstrates

- **Approach trade-off:** raw-api gives you full, explicit control. There is no
  hidden chain — `app/rag.py` chunks the corpus, queries Chroma, assembles the
  grounded prompt string, and calls `chat.completions.create`. The cost is that
  *you* write every step (and would write more for streaming, re-ranking, etc.).
- Embeddings stay local via **chromadb's default embedding function** (ONNX
  MiniLM — no torch, no sentence-transformers).
- The LLM client and the retriever are **injectable**, so unit tests run fully
  offline with stubs.

## How it works

1. **Ingest** (`rag.load_chunks` + `rag.ChromaRetriever`): split each `data/*.md`
   on blank lines into chunks, add them to a fresh persistent Chroma collection.
2. **Retrieve** (`/run`): query the collection for the top-k chunks.
3. **Prompt** (`rag.build_prompt`): inline the retrieved context and instruct the
   model to answer ONLY from it and cite source filenames.
4. **Generate** (`llm.chat`): one `chat.completions` call. For qwen3 models we
   prepend `/no_think` to the system prompt to disable thinking mode.

## API

- `GET /health` → `{"status":"ok","approach":"raw-api","usecase":"01-rag"}`
- `POST /run` body `{"question": str, "top_k": int|null}` →
  `{"answer": str, "sources": [{"source": str, "snippet": str}]}`

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer` key |
| `LLM_MODEL` | `qwen3:1.7b` | model id (qwen3 → `/no_think` auto-applied) |
| `LLM_TEMPERATURE` | `0.0` | sampling temperature (0 = deterministic) |
| `LLM_MAX_TOKENS` | per use case | max tokens for the primary generation |
| `RAG_TOP_K` | `3` | default chunks retrieved per question |
| `CHROMA_DIR` | `.chroma` | Chroma persistence directory |

**Swapping models/providers:** set `LLM_BASE_URL` / `LLM_GATEWAY_KEY` / `LLM_MODEL` (and optional `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`) in `.env` — no code changes. See the root README's "Use a different model or provider" table.


## Run

```bash
python -m uv sync --extra dev   # creates .venv, installs from uv.lock

# Offline unit tests (must pass with no network):
python -m pytest tests/test_unit.py -q

# Live model (needs the local gateway + Qwen running):
RUN_INTEGRATION=1 python -m pytest -q -m integration

# Serve:
python -m uvicorn app.main:app --reload
# then: curl -X POST localhost:8000/run -H 'content-type: application/json' \
#            -d '{"question":"How long is the return window?"}'
```

## Feasibility note

RAG is a natural fit for raw-api: it's a linear retrieve → prompt → generate
pipeline with no branching or cyclic state, so hand-writing it stays readable and
there is little framework value to add. No Anthropic model is needed; the local
Qwen handles grounded QA fine.
