# Feasibility & Build Tracking

Status of each use case × approach. Legend:
- ✅ **Build** — natural fit, ship full example on local Qwen.
- ⚠️ **Build w/ caveat** — works, but awkward in this approach and/or needs Haiku / `qwen2.5:7b` for reliability (noted).
- 📝 **Impractical-but-stub** — ship a *minimal* demo + `README.md` "Why impractical here" pointing to the clean approach. Tracked, never dropped.

Build state column: ☐ not started · ◐ in progress · ☑ done.

| # | Use case | raw-api | langchain | langgraph | claude-agent-sdk | Model default | Build state |
|---|----------|:------:|:--------:|:--------:|:----------------:|---------------|:----------:|
| 1 | Q&A / RAG chatbot | ✅ | ✅ | ⚠️ overkill | ⚠️ no vector store — lexical only | local Qwen + local embeddings · **cloud for agent-sdk** | ☑ ☑ ☑ ☑ |
| 2 | Code generation | ✅ | ✅ | ⚠️ only if iterative | ⭐ native (Write/Bash loop) | `qwen-local-coder` · **cloud for agent-sdk** | ☑ ☑ ☑ ☑ |
| 3 | Data extraction | ✅ | ✅ | ⚠️ unless pipeline | ⚠️ one-shot; harness earns little | **cloud** (local JSON too flaky) | ☑ ☑ ☑ ☑ |
| 4 | Research agent | ⚠️ manual loop | ✅ | ✅ | ✅ built-in WebSearch/WebFetch | **cloud** (local fails ReAct) | ☑ ☑ ☑ ☑ |
| 5 | Customer support triage | ⚠️ no native routing | ✅ | ✅ | ✅ agentic routing | local Qwen · **cloud for agent-sdk** | ☑ ☑ ☑ ☑ |
| 6 | SQL / DB agent | ✅ | ✅ | ⚠️ with validators | ✅ agent discovers schema | `qwen-local-coder` · **cloud for agent-sdk** | ☑ ☑ ☑ ☑ |
| 7 | Multi-agent orchestration | 📝 complex DIY | ⚠️ workaround | ✅ native | ⭐ native subagents + per-role tools | **cloud** | ☑ ☑ ☑ ☑ |
| 8 | Autonomous ReAct | ⚠️ hand-written loop | ⚠️ AgentExecutor | ✅ graph cycles | ⭐ the SDK *is* the loop | **cloud** (local fails ReAct) | ☑ ☑ ☑ ☑ |
| 9 | Recommendations | ✅ | ✅ | ⚠️ profile+state | ⚠️ modest win over one call | local Qwen · **cloud for agent-sdk** | ☑ ☑ ☑ ☑ |
| 10 | Human-in-the-loop approval | 📝 manual queue | ⚠️ callback workaround | ✅ `interrupt()` (durable) | ⭐ `can_use_tool` (in-process only) | local Qwen · **cloud for agent-sdk** | ☑ ☑ ☑ ☑ |

Build-state cells map left→right to raw-api / langchain / langgraph / claude-agent-sdk.
⭐ = the approach's showcase for that use case.

## Combos flagged impractical (carry a README note) — ALL BUILT ✅
- **raw-api/07-multi-agent** — hand-rolled orchestrator; "Why impractical in raw API" note → `langgraph/07`. Built + green.
- **raw-api/10-hitl-approval** — hand-built checkpoint store + `/resume`; note → `langgraph/10` (`interrupt()`). Built + green.
- **claude-agent-sdk/03-data-extraction** — an agent harness doing a one-shot job; no loop runs, no built-in tools used. Note → `raw-api/03` (one HTTP call, cheaper per document).
- **claude-agent-sdk/09-recommendations** — ranking 7 items needs no loop; the win (grounded explanations, partial catalog reads) is modest at this scale. Note → `raw-api/09`.
- **claude-agent-sdk/01-rag** — no vector store: retrieval is lexical `Grep`/`Read`, so semantically-phrased questions can miss. Note → `langchain/01` for semantic recall.

## Empirical model findings (after building)
- **Local Qwen handles:** RAG (1), code-gen (2), triage (5), SQL (6), recommendations (9), and HITL drafting (10) — all on free `qwen-local-instruct`/`qwen-local-coder`.
- **Needs a cloud model:** data-extraction (3, strict JSON), research (4), ReAct (8), multi-agent (7) — local models can't reliably do strict-JSON or multi-step text ReAct. These integration tests are marked `anthropic` (small capped spend).
- **The whole `claude-agent-sdk` column needs a cloud model** — even the six use cases the other approaches run free. The SDK drives the Claude Code harness (large system prompt + built-in tool loop); `qwen3:0.6b` and its larger local siblings cannot sustain it. Every project there caps `AGENT_MAX_TURNS` and `AGENT_MAX_BUDGET_USD`; `claude-sonnet` is the documented escalation for code-gen and multi-agent.
- **Structured tool calls sidestep the PII-redaction bug.** The gateway's Presidio redaction masked the literal ReAct label `Action Input` → `<PERSON>`, breaking text-ReAct parsing (mitigated there by renaming the field to `Arguments:`). `claude-agent-sdk/08` is immune: its tool calls are structured protocol messages, so there is no prose control protocol to mangle.
- See memory `aiup-gateway-quirks` for the redaction behavior and ReAct stop-sequence mitigations.

## Platform drift (found 2026-07-29) — RESOLVED
- Gateway moved **`:8080` → `:8094`**; LiteLLM tier removed (ADR 0036) — `projects-ms` mints native `sk-aiup-…` keys, and model aliases dropped version suffixes (`claude-haiku`, not `claude-haiku-4-5`). Minting a key now also requires an **MFA step-up** (ADR 0042), which `seed-virtual-keys.mjs` predates.
- **Impact:** all 30 pre-existing projects pointed at the retired port with stale aliases and dead keys, so none of their integration tests could run.
- **Keys re-provisioned 2026-07-29:** all ten per-use-case phases recreated with fresh keys (4 model aliases each), distributed across all 40 projects and verified live.
- **Fixed 2026-07-29:** swept 66 files (33 `.env.example` + 33 `settings.py`) to `:8094` and the unsuffixed aliases, re-keyed all 30 `.env` files, and re-verified live — `raw-api/01-rag` and `langgraph/10-hitl-approval` on free local Qwen, `langchain/03-data-extraction` on cloud. **350 offline unit tests green across the 30.** The three `_template/` defaults also moved off the raw `qwen3:1.7b` tag, which the gateway never accepted (aliases only).

## Live-run findings (claude-agent-sdk, validated 2026-07-29)

Three use cases were run live against the gateway (UC02, UC08, UC10). **Every one of the following was invisible to the mocked unit tests and only surfaced under a real agent** — the clearest argument in this repo for keeping gated integration tests:

1. **`can_use_tool` requires streaming input.** Passing a string prompt while a permission callback is set raises `ValueError: can_use_tool callback requires streaming mode`. `default_runner` now wraps the prompt in an `AsyncIterable` when a gate is present.
2. **`allowed_tools` silently shadows the permission gate.** Listing the guarded tool there auto-approves it *before* `can_use_tool` is consulted — the agent sent the message with no approval. The SDK emits `CanUseToolShadowedWarning`. UC10 now passes `allowed_tools=[]`; the tool is still available via its MCP server (that list controls auto-approval, not availability).
3. **Turn/budget caps can raise instead of returning.** *(Refined 2026-07-31 — see finding 18: they sometimes return a `ResultMessage` carrying `stop_reason` and a real cost. Both paths occur; both are handled. As first written this said caps never return, which is too strong.)* Exhausting `max_turns` / `max_budget_usd` may raise from `query()` instead of yielding a `ResultMessage`, in which case `stop_reason` handling never ran. `default_runner` maps those two messages back to `stop_reason="max_turns"` / `"max_budget"`; genuine errors still propagate.
4. **A denial with `interrupt=True` surfaces as an error result.** For UC10 that is the expected terminal state, so `resolve_run` translates it into a rejection — but only on the denied path.
5. **`cwd` is not a sandbox.** The `Write` tool accepts absolute paths, and the model repeatedly wrote to `/tmp/solution.py` instead of the workdir, making UC02 flaky. Mitigated by an explicit relative-paths-only instruction plus reading artefacts back only from the workdir. Not a security boundary — a container/VM or the SDK `sandbox` setting is.
6. **Prompt caching does not pass through the gateway** (`cache_read_input_tokens: 0`), so every agent turn re-pays the full Claude Code harness prompt. Measured: ~847 input tokens for a *trivial* one-shot call, and **$0.35–0.48 for a 9–10 turn code-gen run** on `claude-haiku`. The original `$0.25` cap was exhausted in 15 seconds. Defaults raised to `AGENT_MAX_TURNS=12`, `AGENT_MAX_BUDGET_USD=1.00`.
7. **The gateway's `empty_input` guardrail blocks short prompts** — a `"Say OK"` probe returns `400 NO_SOURCE_PROVIDED / empty_input`. Affects hand-testing with curl, not the examples themselves.

**Second live pass (the remaining seven use cases) found three more:**

8. **`setting_sources=[]` does NOT isolate the run — the most serious finding.** It gates `settings.json` only; the CLI still loads the developer's `~/.claude` project **memory** and parent `CLAUDE.md`. A probe agent recited this repo's private memory index *verbatim*, and UC07 was answering from that leaked context instead of its own corpus (zero tool calls, wrong answer). Fixed by pointing **`CLAUDE_CONFIG_DIR` at a throwaway directory** in `sdk_env()`, which returns `NONE VISIBLE`. Both a reproducibility bug and a disclosure risk — see `docs/security-review.md` **F14**.
9. **The delegation tool is named `Agent`, not `Task`.** UC07's trace extraction matched `Task` and so always reported zero subagents even when delegation worked. The live call is `Agent` carrying `subagent_type`. Both names are now accepted.
10. **The SDK has no `tool_choice`, so "always call this tool" must be carried by the prompt.** UC05 skipped `emit_triage` entirely on a simple question and just answered conversationally — 1 turn, no decision. Fixed by making the prompt state that every ticket gets a decision and that the reply belongs in the tool's field, not in message text.

One further failure was a **test bug, not an app bug**: UC03 asserted `"Northwind" in vendor` case-sensitively while the agent correctly copied `NORTHWIND TRADERS` verbatim from the document, exactly as its prompt demands.

## Live-run findings (the older 30, full sweep 2026-07-30)

All 30 pre-`claude-agent-sdk` projects were live-run after the re-pointing — not just the
three spot-checks. **24 passed first time; 6 failed, all for one shared cause.**

11. **The gateway's Anthropic path returns `500 internal_error` on an OpenAI `stop` array.**
    It does not translate `stop` into Anthropic's `stop_sequences`. Isolated with curl: the
    identical request passes without `stop` and 500s with it, on any `claude-*` alias; the
    Ollama path honours `stop` normally. This broke exactly the six text-ReAct projects —
    UC04 and UC08 across all three approaches — because those are the only ones that send
    `stop`, and both default to `claude-haiku`. **Every one failed on its first LLM call**,
    so it was not a subtle degradation; these two use cases could not have run at all since
    the platform rebuild.

    Fixed in all six by making the stop-cut a client-side guarantee rather than a
    server-side favour: `model_profile()` gained a `supports_stop` capability (False for
    `claude-*`) so `stop` is sent only where it works, and `truncate_at_stop()` now cuts the
    reply at the first `Observation:` regardless. That is the more portable design anyway —
    `stop` is advisory, and a model that writes its own `Observation:` would otherwise feed
    itself fabricated tool output. Each project gained a regression test proving a
    model-supplied observation is discarded, plus one asserting `stop` is withheld from
    `claude-*`.

    Worth noting for the platform: this is a gateway defect, not a repo defect. The
    workaround belongs here regardless, but `/v1/chat/completions` → `/v1/messages`
    translation should map `stop` → `stop_sequences` rather than 500.

## Live-run findings (post-lockfile re-verification, 2026-07-30)

After committing lockfiles for all 44 projects, every venv was re-synced to its
lock and the whole live suite re-run. 39 of 40 passed unchanged; one failure
exposed a defect the earlier passes had hidden.

12. **A capped agent run discarded everything it had already produced.**
    `claude-agent-sdk/07-multi-agent` returned `report: ""` with HTTP 200. It was
    not caused by the new pins — it re-ran green — but chasing it found a real
    bug in the shared `agent.py` seam, present in **all 11 agent-SDK projects**:

    ```python
    return AgentResult(is_error=True, stop_reason=reason)   # fresh, empty
    ```

    On hitting a configured cap, `default_runner` returned a *new* empty result,
    throwing away the accumulated text, every tool call, and the turn count. A
    run that had delegated to all three subagents and written most of a report
    came back indistinguishable from one that did nothing. The intent was right —
    the comment says callers should be able to surface "incomplete" — but there
    was nothing left to surface.

    Fixed by threading a result object into `collect()` so partial state survives
    an exception mid-stream, keeping `.text` current per block rather than only
    at the end, and counting turns as they happen. `cost_usd` deliberately stays
    `0.0` on this path and is documented as *unknown*: no `ResultMessage` arrives,
    so any figure would be invented — which matters because a capped run is by
    definition the expensive one.

    Separately, UC07's `max_turns` floor went 12 → 20. That is what actually ends
    the flake: three delegations plus the lead's own turns did not fit in 12.
    Spend stays bounded by `AGENT_MAX_BUDGET_USD`, which is the cap that really
    protects the budget — turns are a poor proxy for cost.

    Three regression tests: partial work survives a cap, non-cap errors still
    propagate rather than being swallowed, and a capped team run still surfaces
    its report. Verified with three consecutive live UC07 runs, all green.

## Measured: what a trace shows (UC08, all four approaches, 2026-07-30)

`?trace=1` is implemented on UC08 across all four approaches (see
`docs/trace-format.md`). Running the same use case through each, live on
`claude-haiku`, produced the first hard numbers for the comparison this repo
exists to make:

- **raw-api, langchain and langgraph send byte-identical payloads** — 3,469 input
  / 193 output tokens each. LangChain and LangGraph add no prompt overhead here.
  Commonly assumed, rarely measured.
- **Visibility falls as the framework writes more of your loop.** The three
  loop-owning approaches record exact messages, tool results, and per-call
  latency. The Agent SDK records none of those — it does not expose them — so its
  trace marks them `null` and lists them in `not_captured`. Zeros would have read
  as measurements.
- **Only the Agent SDK reports real cost**, and that run cost **$0.42** versus a
  few tenths of a cent for the same model elsewhere: the harness prompt is
  re-paid every turn with no caching through the gateway. Task phrasing differs,
  so it is not a controlled benchmark, but it matches the code-gen figures.
- **Only langgraph can report a route.** `graph_path` shows the cycle
  (`reason→act→observe→reason…`), and an early exit shows up as `["reason"]` with
  zero tool calls — the structural claim of that approach made checkable.

Instrumentation differed per approach and that is itself the lesson: raw-api is
hand-instrumented, langchain attaches a callback handler, langgraph must put
callbacks in the *graph run config* (node identity only appears there), and the
Agent SDK can only be read off the finished result.

## Found by building the Docker quickstart (2026-07-31)

Running the examples the way a newcomer would — `docker compose up`, straight at
Ollama, no gateway — surfaced four things the usual configuration hides:

13. **qwen3 leaks empty `<think></think>` tags into the answer.** `/no_think`
    stops the model *reasoning* in the block but not emitting the tags, so the
    first answer a newcomer sees reads
    `"<think>\n\n</think>\n\nThe return window is 30 days."`. Invisible in normal
    use because the gateway alias `qwen-local-instruct` is qwen2.5, which has no
    thinking mode; the Docker default is a raw qwen3 tag, which does.
    Fixed with `strip_thinking()`, which also keeps chain-of-thought out of API
    responses on principle. **Swept across all 33 OpenAI-surface projects**
    (raw-api, langchain, langgraph, including the three `_template/`s), because
    `PROJECT=` makes every one of them reachable from the Docker quickstart.

    Three different seams, one per approach — the same split the trace work
    found: raw-api strips inside `chat()`; langchain appends `| strip_thinking`
    to its LCEL chains (a bare callable is coerced to a Runnable); langgraph
    strips at each `.content` read, since its nodes consume the message
    directly. Verified live against `qwen3:1.7b` on both the chain-pipe and
    `.content` paths.
14. **Port 11434 collides** with an Ollama the user already runs — and that user
    is exactly who tries this first. The compose file no longer publishes it;
    the app reaches Ollama over the compose network.
15. **Chroma re-downloads its ~80 MB ONNX embedding model on every container
    recreate** unless `/root/.cache/chroma` is a volume. Now it is.
16. **nerdctl ignores `depends_on: condition`** (Rancher Desktop in containerd
    mode), so the app starts before the model finishes pulling and its first
    request fails. Documented in the README rather than worked around — real
    `docker compose` honours it.

## Found while adding streaming to UC07 (2026-07-31)

17. **A plain function in an LCEL chain silently disables streaming.** The
    `strip_thinking` sweep (finding 13) appended `| strip_thinking` to every
    langchain chain. LangChain wraps a bare callable in a `RunnableLambda`,
    which materialises its whole input before running — so `chain.stream()`
    collapsed to **2 token frames instead of 149**, with no error and no
    warning. The endpoint still streamed; it just had nothing to stream.

    Fixed in `langchain/07` with `strip_thinking_stream`, a *generator*
    function: LangChain treats it as a transform, so it yields as it goes.
    `invoke()` is unaffected — the generator is drained.

    **Swept 2026-07-31.** Seven more chain sites carried the same step —
    `01-rag`, `02-code-generation`, `03-data-extraction`, `04-research-agent`,
    `05-support-triage`, `06-sql-agent`, `09-recommendations` — all now use
    `strip_thinking_stream`. (`08` drives the model directly and has no chain;
    `_template` reads `.content`, so it cannot reintroduce the bug into new
    projects.) That is seven, not the "nine" first estimated: two of the
    original matches were the docstring example inside `llm.py` itself.

    Verified by streaming a swept chain live rather than trusting the tests,
    which only exercise `invoke()`: `05-support-triage` now yields **141
    chunks, first at 1.30s of an 8.48s run**. Before the fix it would have been
    one chunk at the end.

## Closing finding 12 properly (2026-07-31)

18. **A capped run kept its partial work but never said it was partial.**
    Finding 12 made the accumulated text, tool calls and turn count survive a
    cap. It stopped one step short: `AgentResult.stop_reason` was populated in
    all 11 agent-SDK projects and read by **none** of them. Every response was
    built from `.text`, `.num_turns` and `.cost_usd` only, so a run cut off at
    `max_turns` still answered **200 with a plausible-looking partial answer** —
    the exact failure mode this repo keeps rediscovering: not an error, a
    believable value.

    It showed up differently per use case, all of them quiet:

    | use case | what a capped run looked like before |
    |---|---|
    | 01, 04, 06 | a short answer that reads as complete |
    | 03, 05, 09 | `valid: false`, "agent did not call emit_x" — blaming the agent for a cap |
    | 02 | missing files reported as an empty solution |
    | 07 | an empty report; the trace *guessed* `max_turns` for any early stop |
    | 08 | `hit_turn_limit` covered the turn cap only — a budget cap read as `ok` |
    | 10 | a rejected send indistinguishable from a failed one |

    Fixed by adding `outcome_of()` to the shared `agent.py`: the raw SDK field
    stays truthful (nullable, exactly what the SDK reported) and the normalised
    reason goes in the response, so callers get one field that is always present
    and always means something. `end_turn` fills the gap where the SDK reports
    nothing — a null reason arrives precisely when the run was fine, which is
    the least useful moment for a caller to have to guess.

    **The vocabulary split is the interesting part.** The HTTP response returns
    the SDK's own reason; the *trace* translates it (`to_trace_reason`) into the
    cross-approach vocabulary in `docs/trace-format.md`. Leaking `end_turn` into
    a trace whose other three rows say `final_answer` would have made the
    comparison tables compare spelling instead of behaviour — and comparison is
    what the trace format exists for.

    UC10 is the deliberate exception: `stop_reason` is nullable there, and the
    null carries meaning. A run parked at the approval gate has not stopped, so
    no reason is the honest answer. Every other use case can only report why a
    run ended; this one can report that it hasn't.

    23 regression tests: two per project, plus three for UC10's gate states. Each
    was mutation-checked — removing the plumbing fails the capped-run test.

    **Live-verified on `01-rag`, and it corrected finding 3.** A normal run
    returns `end_turn` (7 turns, $0.115). Re-run with `AGENT_MAX_TURNS=2` it
    returns `max_turns` — with `cost_usd: 0.0466`, which means a `ResultMessage`
    *did* arrive. Finding 3 recorded that caps "raise, they do not return"; that
    is true of some cap paths but not this one. The SDK does both, and the code
    already handled both, so nothing needed changing — but the note was too
    absolute and is corrected here.

    The capped answer began *"I'll search the documents for information about
    Qwen models, ReAct loops, and coder models…"* — indistinguishable, without
    this field, from a run that was about to finish properly. That is the whole
    argument for the change in one line of output.

## Found while sandboxing the shell (2026-07-31)

19. **A security control that reported its intent instead of its effect.**
    F9 (`02-code-generation` grants `Bash` to a loop driven by untrusted request
    text) had stood at 🔴 High-as-shipped, with the review advising deployers to
    "use the SDK's `sandbox` setting" — advice the repo was not itself taking.
    Taking it was straightforward; verifying it was where the finding appeared.

    The first version set `sandboxed` from `options.sandbox is not None`. Forcing
    the sandbox on under Windows, a live run printed:

    ```
    ⚠ Sandbox disabled: sandbox is enabled but the Windows sandbox is not
      active on this session (feature gate off)
      Commands will run WITHOUT sandboxing.
    ```

    …and the response still said `sandboxed: true`. The config was accepted; the
    sandbox was not applied. Reporting a boundary that is not there is worse than
    reporting none, because it is the claim a deployer acts on.

    This is the **third** instance of the same failure class in this folder, and
    the pattern is now hard to miss: **F11** (a gate that failed open silently),
    **F14** (`setting_sources=[]` documented as isolation, not delivering it),
    and now F9. In every case the setting was accepted, the intent was correct,
    and only running it showed the effect was absent. *A security setting is
    worth exactly its observed effect.*

    Fixed with `SandboxMonitor`, which reads the CLI's stderr — the SDK forwards
    it to a callback — and treats the CLI as the authority over the config.
    `sandbox_note` returns the reason verbatim. Also set
    `allowUnsandboxedCommands: false`: the SDK's default of `true` lets any
    command opt out via `dangerouslyDisableSandbox`, and the thing choosing
    commands is model output. A control the attacker can request be switched off
    is not a control.

    F9 drops 🔴 High → 🟠 Medium. Not closed: the SDK sandboxes bash on
    macOS/Linux only, so the container guidance stands wherever `sandboxed` is
    false. 7 tests, none platform-skipped — a control must not go untested on
    the machine it is authored on, so the platform check is monkeypatched rather
    than skipped around. Live-verified twice: `sandboxed: false` with the CLI's
    reason attached.

## Found while putting claude-agent-sdk in Docker (2026-07-31)

20. **`tests_passed: true` for a run whose every test invocation failed.** The
    worst defect found in this repo so far, and it was found by containerising
    the approach rather than by any test.

    In the image the shell sandbox *did* engage — but bubblewrap could not start
    inside an unprivileged container, so every `Bash` call died on
    `bwrap: pivot_root: Operation not permitted`. The agent did not stop. It
    retried eight times, reworded, tried "without sandbox", and finally gave up
    and **reasoned from reading its own source** that the implementation was
    correct:

    > *"It seems there's a sandbox issue. Let me verify the implementation is
    > correct by reviewing the code logic … The implementation is solid."*

    17 turns, **$0.99**, no test ever executed — and the API answered
    `tests_passed: true`. `_ran_tests_successfully` asked whether `Bash` was
    *invoked*, never whether it *worked*. Its own docstring promised the agent
    "really executed the tests rather than just asserting they would pass",
    which is precisely what it failed to detect.

    Fixed by requiring **pytest's own `.pytest_cache`** in the workdir: written
    by the tool, in the directory it ran in, so it is evidence rather than the
    agent's account. Regression test reproduces the exact shape of that run
    (both files written, `Bash` called, no error, clean `end_turn`) and asserts
    false.

    **What it still cannot prove is that the tests *passed*.** `lastfailed`
    survives a later green run when test ids change, so it is not usable; and
    the only authoritative check — running the tests ourselves — would execute
    model-written code *outside* the sandbox the agent's own shell runs in,
    trading a real boundary for a nicer status field. Recorded as a known limit
    rather than papered over. **This is the cost of not owning the loop**: the
    SDK does not surface tool *results* (see the `not_captured` list in
    `trace.py`), so "did that command succeed?" is a question this approach
    structurally cannot answer directly.

21. **Docker needed no Node and no CLI install** — the Dockerfile said it did.
    The SDK ships the Claude Code CLI as a native binary in a platform-tagged
    wheel (`_bundled/claude`), and the committed lockfiles already carry the
    manylinux wheels. The agent-SDK image built **with no Dockerfile change at
    all**; `claude --version` in it returns 2.1.220, the version the SDK pins.
    The restriction had been assumed from the SDK docs and never tested.

22. **Installing bubblewrap made things strictly worse.** The obvious follow-up
    to finding 19 was to add `bubblewrap` + `socat` so the container's sandbox
    would be real. Measured both ways on the same task:

    | image | result |
    |---|---|
    | without bwrap | CLI downgrades, says so; **4 turns, $0.16**, tests genuinely run |
    | with bwrap installed | every Bash call fails; **17 turns, $0.99**, nothing proved |

    `--privileged` did not fix it either — the error just moved
    (`apply-seccomp: write /proc/self/uid_map: Operation not permitted`, Rancher
    Desktop / Lima). So the image deliberately does **not** install them: a
    half-working sandbox is worse than an absent one that reports itself. In a
    container, the container is the boundary.

## Dependency: chromadb CVE-2026-45829 (assessed 2026-07-31)

Two critical Dependabot alerts (`langchain/01-rag`, `langgraph/01-rag`).
**Dismissed as `not_used`, not ignored.** The advisory is a *pre-authentication
code injection in Chroma's HTTP server* — `/api/v2/tenants/{t}/databases/{d}/collections`
with `trust_remote_code`. These projects use **embedded** chromadb
(`persist_directory` only): no `HttpClient`, no `chroma run`, no bound port, so
the vulnerable endpoint is never served.

There is also **nothing to upgrade to**: 1.5.9 is both the latest release and
vulnerable (`first_patched_version: null`), and `langchain-chroma` requires
`chromadb>=1.3.5`, so downgrading below the affected range is not available
either. `raw-api/01-rag` sits on 0.6.3 and is outside the range already.

CI's `pip-audit` gate carries the matching `--ignore-vuln` with the same
rationale, so the decision is enforced in one place and recorded in two.
Re-open if these examples ever switch to Chroma's client/server mode.

## Status
- **10/10 use cases × 4 approaches = 40 projects built.**
- `claude-agent-sdk`: **176 offline unit tests green**. **All 14 integration tests verified live and passing — all 10 use cases.** Two live passes found 8 defects plus 1 test bug; every one is fixed, and the security-relevant ones (F10, F11, F14) are in `docs/security-review.md`.
- raw-api / langchain / langgraph: re-pointed at `:8094` and **all 30 live-run 2026-07-30 — 30/30 passing** (18 free-local, 12 cloud). The sweep found one defect (finding 11 above) affecting 6 projects.
- **Running total: 22 findings recorded; 21 of them surfaced by *running* the code, not by testing it.** The exception is finding 18 (`stop_reason` never reaching a response), which was found by reading. Across both the agent-SDK build and the older 30. Three — F9, F11, F14 — were security controls that were accepted, configured correctly, and simply did not take effect.
