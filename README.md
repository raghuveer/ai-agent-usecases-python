# AI Agent Use Cases — Raw API · LangChain · LangGraph

Ten common agent use cases, each implemented three ways — direct **Raw API**, **LangChain**, and **LangGraph** — in **Python + FastAPI**. The goal is to *compare* the approaches on the same problems. Each folder is a self-contained, runnable example with its own tests.

> ⚠️ **Status:** early scaffolding. Folder structure is in place; example code is being built (see `PLAN.md`). Nothing here is production-hardened.

## Layout

```
<approach>/<use-case>/   →   raw-api/01-rag, langchain/01-rag, langgraph/01-rag, …
```

| # | Use case | raw-api | langchain | langgraph |
|---|----------|---------|-----------|-----------|
| 1 | Q&A / RAG chatbot | [▸](raw-api/01-rag) | [▸](langchain/01-rag) | [▸](langgraph/01-rag) |
| 2 | Code generation | [▸](raw-api/02-code-generation) | [▸](langchain/02-code-generation) | [▸](langgraph/02-code-generation) |
| 3 | Data extraction | [▸](raw-api/03-data-extraction) | [▸](langchain/03-data-extraction) | [▸](langgraph/03-data-extraction) |
| 4 | Research agent | [▸](raw-api/04-research-agent) | [▸](langchain/04-research-agent) | [▸](langgraph/04-research-agent) |
| 5 | Customer support triage | [▸](raw-api/05-support-triage) | [▸](langchain/05-support-triage) | [▸](langgraph/05-support-triage) |
| 6 | SQL / DB agent | [▸](raw-api/06-sql-agent) | [▸](langchain/06-sql-agent) | [▸](langgraph/06-sql-agent) |
| 7 | Multi-agent orchestration | [▸](raw-api/07-multi-agent) | [▸](langchain/07-multi-agent) | [▸](langgraph/07-multi-agent) |
| 8 | Autonomous ReAct | [▸](raw-api/08-autonomous-react) | [▸](langchain/08-autonomous-react) | [▸](langgraph/08-autonomous-react) |
| 9 | Recommendations | [▸](raw-api/09-recommendations) | [▸](langchain/09-recommendations) | [▸](langgraph/09-recommendations) |
| 10 | Human-in-the-loop approval | [▸](raw-api/10-hitl-approval) | [▸](langchain/10-hitl-approval) | [▸](langgraph/10-hitl-approval) |

Approach suitability per use case (and combos deliberately kept minimal) is tracked in **`TRACKING.md`**.

## How models are accessed

Every example speaks **OpenAI-compatible HTTP** to a local gateway — no provider is hardcoded. Configure via env (`.env.example` in each folder):

- `LLM_BASE_URL` — gateway base URL (default `http://localhost:8080`)
- `LLM_GATEWAY_KEY` — your platform Bearer key
- `LLM_MODEL` — e.g. a self-hosted Qwen model (default) or `claude-haiku-4-5`

**Self-hosted Qwen is the default** (zero cost); a few agentic use cases use `claude-haiku-4-5` where small models fall short. Details and the cost rules are in `SPEC.md`.

## Docs
- `agent-development-guide.md` — the 10 use cases, three approaches, scalability, language support.
- `SPEC.md` — project contract, model strategy, testing & cost rules.
- `PLAN.md` — phased build order and current state.
- `TRACKING.md` — feasibility matrix & build status.

## License
TBD before public release.
