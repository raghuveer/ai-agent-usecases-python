# UC01 — rag (claude-agent-sdk)

Grounded question answering over a local corpus — **with no vector store, no
embedding model, and no chunking.**

## A different retrieval architecture

The other three approaches embed `data/` into Chroma with
`sentence-transformers` and retrieve the top-k nearest chunks. Here the agent is
handed `Grep`, `Glob`, and `Read` over the corpus directory and devises its own
search strategy: grep a term, read what looks relevant, refine, repeat.

| | raw-api / langchain / langgraph | claude-agent-sdk |
|---|---|---|
| Index | Chroma + `sentence-transformers` | none |
| Retrieval | top-k vector similarity | agent-driven `Grep` / `Read` |
| Adding a document | re-embed and re-index | drop the file in `data/` |
| Matching | semantic | lexical / exact |
| Cost per question | one embedding + one completion | several agent turns |

**The honest trade-off:** there is no semantic matching. A question worded
differently from the documents may retrieve nothing, where embeddings would still
match. The agent partly compensates by trying different wording across turns —
but that costs turns, so this is slower and pricier per question than a vector
lookup. Use this when the corpus is small, textual, and on disk; use the vector
implementations when you need semantic recall at scale.

## Making retrieval inspectable

The response reports what the agent actually did, not what it was asked to do:

- `sources` — files it opened with `Read`, de-duplicated in first-read order
- `searches` — the `Grep` patterns it tried, i.e. its retrieval strategy

An answer with an empty `sources` list is a red flag (the model answered without
opening anything), and the API surfaces that rather than hiding it.

## Corpus

`data/` holds three short markdown notes from this repository's own build
findings. Everything is local, so this runs air-gapped.

## API

- `GET /health` → `{"status":"ok","approach":"claude-agent-sdk","usecase":"01-rag"}`
- `POST /run` body `{"question": str}` →
  `{"answer": str, "sources": [str], "searches": [str], "num_turns": int,
    "cost_usd": float}`

## Env vars (`.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8094` | Gateway **Anthropic** surface — no `/v1` suffix |
| `LLM_GATEWAY_KEY` | placeholder | virtual key, sent as `Authorization: Bearer` |
| `LLM_MODEL` | `claude-haiku` | ample for corpus Q&A |
| `AGENT_MAX_TURNS` | `12` | each search/read round costs a turn |
| `AGENT_MAX_BUDGET_USD` | `1.00` | hard spend cap per run |
| `AGENT_EFFORT` | `low` | |

## Prerequisites

- **Python 3.12+** — all app code is Python (`claude-agent-sdk`).
- **Node.js 18+ and the Claude Code CLI on PATH** for live runs; the Python SDK
  spawns it. Unit tests need neither.

## Run

```bash
python -m uv venv
python -m uv pip install --python .venv/Scripts/python.exe -e ".[dev]"

# Offline unit tests (no network, no key, no CLI):
.venv/Scripts/python.exe -m pytest tests/test_unit.py -q

# Live agent (gateway + key + Node/CLI; spends a little budget):
RUN_INTEGRATION=1 RUN_ANTHROPIC_TESTS=1 \
  .venv/Scripts/python.exe -m pytest tests/test_integration.py -q -m integration

# Serve:
.venv/Scripts/python.exe -m uvicorn app.main:app --reload
#   curl -X POST localhost:8000/run -H 'content-type: application/json' \
#     -d '{"question":"Could small local models drive a text ReAct loop?"}'
```
