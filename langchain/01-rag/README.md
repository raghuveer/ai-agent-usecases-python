# UC1 — RAG (langchain approach)

Retrieval-Augmented Generation over a small bundled corpus (Northwind Robotics
product/return/warranty/support docs), built the **langchain** way: a
`langchain-chroma` vector store + an **LCEL retrieval chain** wired to
`ChatOpenAI` pointed at the local OpenAI-compatible gateway.

## What it does

`POST /run` with a question:
1. Retrieves the top-k chunks from Chroma (default embeddings, ONNX MiniLM).
2. Builds a grounded prompt that tells the model to answer **only** from the
   retrieved context and cite the source file names.
3. Calls the LLM and returns the answer plus the `sources` used.

## Which trade-offs this approach demonstrates

- **Less plumbing than raw-api**: retrieval, prompt formatting, and the model
  call compose as one LCEL pipeline
  (`{"context": retriever | format_docs, "question": passthrough} | prompt | llm | parser`).
- **Uniform client path**: the same `ChatOpenAI` talks to Qwen or Claude — no
  per-provider branching.
- **Cost**: the retrieval/format/parse glue is hidden; you see less of exactly
  what bytes go over the wire than in the raw-api version (that's the comparison
  point against `raw-api/01-rag`).

## API

- `GET /health` → `{"status":"ok","approach":"langchain","usecase":"01-rag"}`
- `POST /run` body `{"question": str, "top_k": int|null}` →
  `{"answer": str, "sources": [{"source": str, "snippet": str}]}`

## Env vars (`.env.example`)

| var | default | meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible gateway |
| `LLM_GATEWAY_KEY` | placeholder | bearer key (placeholder fine for local/unit) |
| `LLM_MODEL` | `qwen3:1.7b` | model id; `qwen3:*` → `/no_think` system prompt |
| `RAG_TOP_K` | `3` | chunks retrieved per query |
| `CHROMA_DIR` | `.chroma` | persistent vector-store directory |

## Run

```bash
python -m uv venv
python -m uv pip install -e .

# offline, mocked — no network:
python -m pytest tests/test_unit.py -q

# live against the local gateway + real Chroma:
RUN_INTEGRATION=1 python -m pytest -q -m integration

# serve it:
uvicorn app.main:app --reload
```

## Testing

- **Unit** (`tests/test_unit.py`): a fake in-memory retriever + `FakeListChatModel`
  injected on `app.state`. Zero network; asserts the LCEL wiring, source
  reporting, and `/no_think` handling. Runs in CI.
- **Integration** (`tests/test_integration.py`): marked `@pytest.mark.integration`
  and skipped unless `RUN_INTEGRATION=1`. Ingests the corpus into a throwaway
  Chroma dir and round-trips a question against local Qwen.

## Feasibility note

RAG is a natural fit for langchain — the LCEL retrieval chain is exactly the
idiom the framework is built for, so this approach is the most concise of the
three with no awkwardness. Embeddings stay local (chromadb default ONNX MiniLM);
no torch / sentence-transformers, no Anthropic needed.
