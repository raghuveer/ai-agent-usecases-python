# UC1 — RAG (langgraph approach)

Retrieval-augmented Q&A over a small bundled corpus (Northwind Robotics product,
returns, warranty, and support docs), implemented as a **langgraph
`StateGraph`**. Ask a question; the app retrieves the most relevant chunks from a
local Chroma index, builds a grounded prompt, calls the model, and returns the
answer plus the sources it used.

## Approach trade-offs this demonstrates

- One uniform LLM path: `langchain-openai`'s `ChatOpenAI` pointed at the
  OpenAI-compatible gateway, for both Qwen and Claude.
- A **typed state** (`RAGState`) flowing through explicit nodes:
  `retrieve` → `generate` → `END`.
- Dependency injection at the graph boundary: the retriever and the LLM are
  passed in, so tests run fully offline.

### Feasibility note — RAG is overkill on a graph

This is **intentionally a tiny graph**. RAG is a linear pipeline (retrieve, then
generate) with no branching, looping, or shared mutable state to coordinate —
exactly the situation where a graph buys you nothing over a plain function or an
LCEL chain. It is modelled as a `StateGraph` here only so the langgraph approach
can be compared apples-to-apples with the raw-api and langchain versions. The
graph idiom earns its keep on the later use cases (multi-agent, autonomous
ReAct, HITL), not here.

## Layout

```
app/
  settings.py   # pydantic-settings: env contract + RAG_TOP_K, CHROMA_DIR
  llm.py        # ChatOpenAI factory (injectable) + /no_think for qwen3
  rag.py        # ingest (Chroma, default embeddings) + retrieve→generate graph
  main.py       # FastAPI: /health, /run; ingests + compiles graph on startup
data/           # the 4 bundled markdown docs (identical across approaches)
tests/
  test_unit.py        # fake retriever + FakeListChatModel, no network
  test_integration.py # live local Qwen, gated by RUN_INTEGRATION=1
```

## Embeddings

Uses **chromadb's default embedding function** (ONNX MiniLM). No torch or
sentence-transformers are installed.

## API

- `GET /health` → `{"status":"ok","approach":"langgraph","usecase":"01-rag"}`
- `POST /run` `{"question": str, "top_k": int|null}` →
  `{"answer": str, "sources": [{"source": str, "snippet": str}]}`

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | `Authorization: Bearer <key>` |
| `LLM_MODEL` | `qwen3:1.7b` | model id (qwen3 → `/no_think` is prepended) |
| `LLM_TEMPERATURE` | `0.0` | sampling temperature |
| `LLM_MAX_TOKENS` | `512` | max generation tokens |
| `RAG_TOP_K` | `3` | chunks retrieved per query |
| `CHROMA_DIR` | `.chroma` | Chroma persistence directory |

**Swapping models/providers:** set `LLM_BASE_URL` / `LLM_GATEWAY_KEY` / `LLM_MODEL` (and optional `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`) in `.env` — no code changes. See the root README's "Use a different model or provider" table.

## Run

```bash
python -m uv venv
python -m uv pip install -e ".[dev]"

# Unit tests — offline, mocked, no network
python -m pytest tests/test_unit.py -q

# Integration — needs the live local gateway/model
RUN_INTEGRATION=1 python -m pytest -q -m integration

# Serve
uvicorn app.main:app --reload
# POST /run  {"question": "How long is the return window?"}
```
