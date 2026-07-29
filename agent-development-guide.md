<!-- Extended 2026-07-29 with the fourth approach (claude-agent-sdk).
Per-use-case implementation deltas live in SPEC.md §7 and TRACKING.md. -->

# Agent Development: Use Cases, Frameworks & Scalability

> A reference guide for enterprise architects covering 10 agent use cases, four development approaches (Raw API, LangChain, LangGraph, Claude Agent SDK), scalability considerations, and language support.

---

## Table of Contents

1. [The 10 Use Cases](#1-the-10-use-cases)
2. [Approach Comparison Matrix](#2-approach-comparison-matrix)
3. [The Four Approaches](#3-the-four-approaches)
   - [Raw API + LiteLLM / AI Gateway](#31-raw-api--litellm--ai-gateway)
   - [LangChain](#32-langchain)
   - [LangGraph](#33-langgraph)
   - [Claude Agent SDK](#34-claude-agent-sdk)
4. [Workflows vs Agents](#4-workflows-vs-agents)
5. [Scalability](#5-scalability)
6. [Language Support](#6-language-support)
7. [Recommendations by Complexity](#7-recommendations-by-complexity)

---

## 1. The 10 Use Cases

### 1. Q&A / RAG Chatbot
Embed documents into a vector store, retrieve relevant chunks at query time, and pass them as context to the LLM for grounded, citation-backed answers.

**Key components:** vector store (Qdrant, Weaviate, pgvector), embedding model, retrieval chain  
**Complexity:** Low — single-call pattern, stateless

---

### 2. Code Generation Agent
Takes a specification, generates code, optionally executes tests, and iterates on failures. Single-shot for simple generation; stateful for iterative refinement loops.

**Key components:** code execution sandbox, test runner, feedback loop  
**Complexity:** Low to Medium

---

### 3. Data Extraction
Structured output extraction from unstructured documents — invoices, PDFs, logs, HTML pages. Works well with function calling / JSON mode on any LLM provider.

**Key components:** document loaders, output parsers, schema validation  
**Complexity:** Low — excellent fit for raw API with structured output

---

### 4. Research Agent
Plans a set of sub-queries, calls search tools (web search, internal knowledge bases) in parallel or sequentially, then synthesises and cites findings.

**Key components:** search tools, parallel tool execution, summarisation chain  
**Complexity:** Medium — inherently multi-step, needs looping logic

---

### 5. Customer Support Triage Agent
Classifies user intent, routes to specialist chains (billing, technical, general), maintains conversation memory, and escalates to a human when confidence is low.

**Key components:** intent classifier, routing logic, conversation memory, escalation handler  
**Complexity:** Medium — benefits from state machines and conditional routing

---

### 6. SQL / DB Agent
Translates natural language questions into SQL queries, validates syntax, executes against a database, and explains results in plain language.

**Key components:** schema-aware prompting, query validator, DB connector, result formatter  
**Complexity:** Low to Medium — schema injection is the critical step

---

### 7. Multi-Agent Orchestration
An orchestrator agent delegates tasks to specialised sub-agents (researcher, writer, reviewer, compliance auditor) and aggregates their outputs. Analogous to a team of LLM workers with defined roles.

**Key components:** orchestrator node, sub-agent registry, result aggregation, shared state  
**Complexity:** High — requires explicit coordination primitives

---

### 8. Autonomous Workflow (ReAct / Plan-Act-Reflect)
The agent decides which tool to invoke next based on its observation of the previous step, then retries on failure. Implements the Reasoning + Acting (ReAct) pattern with optional self-reflection.

**Key components:** tool registry, observation parser, loop-until-done logic, error handling  
**Complexity:** High — requires graph cycles; LangGraph's native territory

---

### 9. Personalised Recommendations
Pulls user profile data and contextual signals, optionally queries a product/content database, then generates ranked recommendations with natural language explanations.

**Key components:** profile store, retrieval, ranking logic, explanation generation  
**Complexity:** Low to Medium — works well across all four approaches

---

### 10. Human-in-the-Loop (HITL) Approval Workflow
The agent pauses at defined checkpoints (e.g., before executing a high-risk action, before sending an email, before deploying code) and waits for human approval before resuming.

**Key components:** interrupt/pause mechanism, async queue, approval UI, state persistence  
**Complexity:** High — LangGraph's `interrupt()` is the cleanest implementation; raw API requires manual queue/webhook plumbing

---

## 2. Approach Comparison Matrix

| # | Use Case | Raw API | LangChain | LangGraph | Claude Agent SDK |
|---|----------|---------|-----------|-----------|------------------|
| 1 | Q&A / RAG chatbot | ✓ Simple | ✓ Best fit | ~ Overkill | ~ Lexical only, no vector store |
| 2 | Code generation | ✓ Fine | ✓ Tools help | ~ Only if iterative | ✓✓ Built-in Write/Bash loop |
| 3 | Data extraction | ✓ Great | ✓ Document loaders | ~ Unless pipeline | ~ One-shot; harness idle |
| 4 | Research agent | ~ Manual loops | ✓ Agents + tools | ✓ Parallel nodes | ✓ Built-in WebSearch/WebFetch |
| 5 | Customer support | ~ No routing | ✓ Chain + memory | ✓ State machines | ✓ Agent decides routing |
| 6 | SQL / DB agent | ✓ Direct | ✓ SQL chain | ✓ With validators | ✓ Agent discovers schema |
| 7 | Multi-agent system | ✗ Complex DIY | ~ Possible | ✓ Native support | ✓✓ Subagents as data |
| 8 | Autonomous workflow | ✗ Hard to build | ~ Partial | ✓ Graph cycles | ✓✓ The SDK *is* the loop |
| 9 | Recommendations | ✓ Direct | ✓ With memory | ✓ Profile + state | ~ Modest win over one call |
| 10 | Human-in-the-loop | ✗ Ad-hoc only | ~ Callbacks | ✓ `interrupt()` built-in | ✓✓ `can_use_tool` — but in-process only |

**Legend:** ✓✓ Showcase &nbsp;|&nbsp; ✓ Suitable &nbsp;|&nbsp; ~ Partial / workaround &nbsp;|&nbsp; ✗ Not ideal

---

## 3. The Four Approaches

### 3.1 Raw API + LiteLLM / AI Gateway

Call the LLM directly via `/v1/chat/completions` with tool definitions in the payload. Any LLM proxy (LiteLLM, Highper AI Gateway, OpenRouter) normalises authentication and model routing behind a single endpoint.

**How it works:**
```
User input
  → Build messages[] + tools[]
  → POST /v1/chat/completions
  → If response has tool_calls → execute tool → append result → loop
  → Return final text response
```

**Strengths:**
- Full control over retry logic, token budgets, model routing
- No framework lock-in — works in any language that can make HTTP calls
- LiteLLM virtual key + budget management integrates natively
- Ideal for sovereign / air-gapped deployments where you control every component
- Easiest to audit and debug — you see exactly what is sent to the LLM

**Weaknesses:**
- You build your own agent loop, memory management, and state machine
- Multi-agent coordination requires significant DIY effort
- HITL pause/resume requires manual async queue implementation

**Best for:** Use cases 1, 2, 3, 6, 9 — and any context where framework overhead is unacceptable.

---

### 3.2 LangChain

A framework providing abstractions over LLM calls: chains (sequential steps), document loaders, retrievers, memory backends, and tool wrappers. Supports Python and JavaScript/TypeScript.

**How it works:**
```
Prompt Template → LLM → Output Parser → (optional) Tool → next Chain step
```

Memory backends (Redis, Postgres, in-memory) provide conversation history across turns.

**Strengths:**
- Fast to prototype RAG, SQL agents, and support bots
- Rich ecosystem of document loaders and tool integrations
- Supports multiple LLM providers (OpenAI, Anthropic, local models via Ollama)
- Good for teams already familiar with the ecosystem

**Weaknesses:**
- Abstraction layer can obscure what is actually sent to the LLM — harder to debug in production
- Callback chain overhead adds latency
- In sovereign/air-gapped deployments, verify no telemetry or external calls are triggered by default
- Complex multi-agent scenarios require workarounds

**Best for:** Use cases 1, 2, 3, 4, 5, 6, 9 — rapid prototyping and medium-complexity agents.

---

### 3.3 LangGraph

Built on top of LangChain, but replaces linear chains with a directed graph where nodes are functions and edges carry typed state. Supports cycles (essential for ReAct), parallel branches, conditional routing, and built-in HITL via `interrupt()`.

**How it works:**
```
Define State (TypedDict)
  → Add Nodes (Python functions: reason, act, validate, human_review…)
  → Add Edges (conditional or unconditional)
  → Compile graph
  → Invoke with initial state → graph runs until END node
```

**Strengths:**
- Native support for looping / ReAct patterns via graph cycles
- `interrupt()` pauses execution at any node for human review, then resumes
- State is fully typed and serialisable to PostgreSQL or Redis checkpointer
- Multi-agent: each sub-agent is a sub-graph
- Parallel branches execute concurrently

**Weaknesses:**
- Steeper learning curve — graph mental model is different from chain mental model
- State must be serialised at each node — adds latency and infra dependency
- HITL blocking patterns need async queue design at the application layer

**Best for:** Use cases 7, 8, 10 — and any scenario requiring "loop until done", conditional branching, or human approval gates.

---

### 3.4 Claude Agent SDK

Claude Code packaged as a library (`claude-agent-sdk`). Where the other three give you
pieces to assemble a loop, this supplies the loop, a set of built-in tools, subagents,
hooks, and a permission callback. You provide tools and a prompt.

**How it works:**
```
ClaudeAgentOptions(system_prompt, tools, agents, can_use_tool, max_turns, max_budget_usd)
  → query(prompt, options)
  → SDK spawns the Claude Code CLI, runs the agent loop, executes tools
  → async stream of messages back
```

**Strengths:**
- Built-in `Read`/`Write`/`Edit`/`Bash`/`Glob`/`Grep`/`WebSearch`/`WebFetch` — no tool plumbing
- Subagents are declarative: `{name: AgentDefinition}`, each with its own context **and tool allow-list** (least privilege per role)
- `can_use_tool` is an async permission callback, so human-in-the-loop is just awaiting a future
- Structured tool calls, so no text ReAct protocol to parse or have mangled by a redaction layer
- The only approach that caps **both** turns and spend per run

**Weaknesses:**
- Anthropic-only — it speaks the Messages API, so it is not provider-portable like the other three
- Cannot run on small local models; the harness needs a capable cloud model
- Spawns the Claude Code CLI (Node) as a subprocess — an extra runtime dependency
- Poor value on one-shot tasks where no loop runs (3, 9)
- Largest blast radius: shell execution and filesystem writes are on by default in the code-gen case, and `cwd` does **not** sandbox them

**Best for:** Use cases 2, 7, 8, 10 — agentic work with tools, delegation, or approval gates.

---

## 4. Workflows vs Agents

All four approaches support both patterns:

| Pattern | Definition | Routing | Example |
|---------|-----------|---------|---------|
| **Workflow** | Deterministic sequence of steps | Hardcoded | Invoice → extract → validate → store |
| **Agent** | LLM dynamically decides next action | LLM-decided | Research agent choosing which tool to call next |

The key distinction is whether the control flow is predetermined (workflow) or decided at runtime by the LLM (agent). LangGraph makes this explicit — a pure workflow has no cycles and only unconditional edges; an agent has cycles and conditional edges driven by LLM output.

In practice, most production systems are hybrid: deterministic outer workflow with agentic inner loops for specific steps.

---

## 5. Scalability

### Horizontal Scale Comparison

| Approach | Scale Ceiling | Stateless? | Primary Bottlenecks |
|----------|--------------|------------|---------------------|
| Raw API + LiteLLM | Very high | Yes | Token rate limits on LLM provider; mitigated by LiteLLM budget routing and load balancing across model endpoints |
| LangChain | Medium | Configurable | Memory store contention; callback chain overhead adds per-request latency |
| LangGraph | Medium-high | No (state graph) | State serialisation at each node; HITL blocking requires async queue; PostgreSQL checkpointer is production-safe but adds infra |

### Design Patterns for Scale

**Raw API agents** are the easiest to scale horizontally — each invocation is a stateless HTTP call sequence. Drop them behind any load balancer or serverless function.

**LangChain agents** scale well if you externalise memory to Redis or PostgreSQL and avoid in-memory stores. Each chain invocation can be stateless if the memory backend handles persistence.

**LangGraph agents** require a checkpointer (PostgreSQL recommended for production) to persist graph state between nodes and across HITL pauses. The graph compiler produces a deterministic execution plan, which aids reproducibility but requires the state store to be highly available.

### Sovereign / Air-Gapped Considerations

For BFSI and Aerospace deployments operating air-gapped:

- Raw API + LiteLLM is the most robust — no external calls, all routing is internal
- LangGraph with a local PostgreSQL checkpointer is viable and production-safe
- Audit LangChain callbacks carefully — some integrations may attempt external telemetry calls by default

---

## 6. Language Support

| Language | Raw API | LangChain | LangGraph |
|----------|---------|-----------|-----------|
| **Python** | `anthropic`, `openai` SDKs; or plain `httpx` / `requests` | Full support — primary language | Full support — primary language |
| **PHP** | `curl` / Guzzle to `/v1/chat/completions` (LiteLLM proxy normalises auth) | No official SDK | No |
| **Node.js / TypeScript** | `@anthropic-ai/sdk`, `openai` npm | `langchain` npm package | `@langchain/langgraph` npm |
| **Rust** | `reqwest` + `serde_json` — works via LiteLLM proxy | No | No |
| **Go** | `net/http` to proxy endpoint | Community ports only | No |
| **Java / C#** | HTTP client to proxy endpoint — framework-agnostic | Community ports only | No |

### PHP Agent Pattern (via LiteLLM proxy)

Since LiteLLM exposes an OpenAI-compatible endpoint, a PHP agent is a structured loop:

```php
class AgentLoop
{
    private string $baseUrl;
    private string $virtualKey;
    private array $tools;

    public function run(string $userMessage): string
    {
        $messages = [['role' => 'user', 'content' => $userMessage]];

        while (true) {
            $response = $this->callLLM($messages);
            $choice = $response['choices'][0];

            if ($choice['finish_reason'] === 'stop') {
                return $choice['message']['content'];
            }

            if ($choice['finish_reason'] === 'tool_calls') {
                $messages[] = $choice['message'];
                foreach ($choice['message']['tool_calls'] as $toolCall) {
                    $result = $this->executeTool(
                        $toolCall['function']['name'],
                        json_decode($toolCall['function']['arguments'], true)
                    );
                    $messages[] = [
                        'role'         => 'tool',
                        'tool_call_id' => $toolCall['id'],
                        'content'      => json_encode($result),
                    ];
                }
            }
        }
    }

    private function callLLM(array $messages): array
    {
        // POST to LiteLLM proxy — identical to OpenAI API
        $ch = curl_init("{$this->baseUrl}/v1/chat/completions");
        curl_setopt($ch, CURLOPT_HTTPHEADER, [
            'Content-Type: application/json',
            "Authorization: Bearer {$this->virtualKey}",
        ]);
        curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode([
            'model'    => 'claude-sonnet-4-6',
            'messages' => $messages,
            'tools'    => $this->tools,
        ]));
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        $result = curl_exec($ch);
        curl_close($ch);
        return json_decode($result, true);
    }
}
```

This `AgentLoop` class is a natural addition to any PHP framework ecosystem (Highper/Easeapp) and works with any LiteLLM-proxied model.

---

## 7. Recommendations by Complexity

| Scenario | Recommended Approach | Rationale |
|----------|---------------------|-----------|
| Single-shot / RAG tasks | Raw API + LiteLLM | Minimal overhead, works in any language, no framework lock-in |
| Tool-using agents (2–5 steps) | LangChain (Python/JS) | Good primitives, fast development, rich tool ecosystem |
| Stateful / multi-agent | LangGraph | Native state graph, parallel branches, sub-graph support |
| Loop-until-done (ReAct) | LangGraph | Graph cycles are first-class; clean retry and reflection patterns |
| Human-in-the-loop / approvals | LangGraph | `interrupt()` built-in; no ad-hoc queue plumbing needed |
| PHP agents | Raw API only | No LangChain/LangGraph PHP SDK; wrap in `AgentLoop` class |
| Sovereign / air-gapped | Raw API or LangGraph + local checkpointer | Avoid LangChain's external callback URLs; control every component |
| High-throughput production | Raw API + LiteLLM | Stateless, horizontally scalable, budget-aware via virtual keys |

---

## Appendix: Framework Quick Reference

| Feature | Raw API | LangChain | LangGraph |
|---------|---------|-----------|-----------|
| Tool / function calling | Manual parse | Built-in | Built-in |
| Conversation memory | DIY (Redis/DB) | Built-in backends | State schema |
| Vector store integration | DIY | 50+ integrations | Via LangChain |
| Streaming | Supported | Supported | Supported |
| Parallel tool execution | DIY | Partial | Native (parallel edges) |
| Graph / DAG execution | No | No | Yes |
| HITL interrupt/resume | Manual | Callback workaround | `interrupt()` native |
| State persistence | DIY | Memory backends | Checkpointer (PG/Redis) |
| Multi-agent sub-graphs | No | Workaround | Yes |
| Python support | Yes | Yes | Yes |
| JS/TS support | Yes | Yes | Yes |
| PHP / Rust support | Yes (HTTP) | No | No |
| Air-gap safe | Yes | Verify callbacks | Yes (local checkpointer) |

---

*Document version: June 2026*  
*Scope: Enterprise agent architecture reference*
