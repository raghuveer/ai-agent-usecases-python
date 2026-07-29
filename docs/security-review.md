# Security Review — ai-agent-usecases-python

**Date:** 2026-06-25 · **Version reviewed:** v0.1.0 (+ post-release hardening) ·
**Reviewer:** Raghuveer Dendukuri with Claude Code (Opus)

> **⚠️ Scope note.** §1–§10 below cover the **30** services built with raw-api /
> langchain / langgraph. The fourth approach, **`claude-agent-sdk/`** (10 further
> services), was added later and is reviewed in **[§11 — Addendum](#11-addendum--claude-agent-sdk-approach-2026-07-29)**.
> Read §11 before deploying anything from that folder: it has a **materially
> larger blast radius** than the other three (shell execution enabled by default,
> filesystem writes, a subprocess runtime), and it raises one finding to High.

This review assesses the 30 example services against **NIST** secure-development and
GenAI guidance, the **OWASP Top 10 for LLM Applications (2025)**, and the **OWASP
Web Application Security Top 10 (2021)**. It combines automated tooling (SAST, dependency
CVE audit, IaC/secret scanning, and DAST) with a manual code review.

> **Posture:** these are **reference examples, not production systems** (see `SECURITY.md`).
> No High/Critical issue was found in the application *code*. The material findings are
> (1) **vulnerable third-party dependencies** (LangChain/LangGraph/Starlette/Chroma) and
> (2) **intentionally-omitted web hardening** (auth, CORS, rate-limit, TLS) that any
> deployer must add. Safe hardening was applied as part of this review; the dependency
> upgrades are a tracked follow-up because they require breaking major-version migrations.

---

## 1. Methodology & tooling

| Tool | Class | Scope | Result |
|---|---|---|---|
| **Bandit** 1.9.4 | Python SAST | all `app/` code | 9 Low (no High/Medium) |
| **pip-audit** 2.10.1 + **OSV.dev** | Dependency CVE audit | 151 pinned versions across all venvs | **21 advisories** (9 packages) |
| **Trivy** 0.70 | IaC/secret/vuln | repo + `.github` config | CI workflow **clean**; deps cross-checked (fs scan deferred to OSV — see §8) |
| **OWASP ZAP** (baseline) | DAST | running `raw-api/01-rag` | 1 Informational, 0 vulns (shallow — see note) |
| **PortSwigger Dastardly** | DAST | running `raw-api/01-rag` | 0 findings (shallow — see note) |
| Manual review | Code/threat | all 10 use cases × 3 approaches | see §3–§5 |

**DAST coverage note:** the services are JSON APIs with no crawlable HTML/links, and the
OpenAPI spec was not imported into the scanners, so ZAP/Dastardly spiders only reached `/`
and `/robots.txt`. The DAST runs confirm there is no obvious web-layer vulnerability on the
reachable surface and no information leakage, but for deep DAST you should import each
service's `/openapi.json` (e.g. ZAP's OpenAPI add-on) and provide a valid `LLM_GATEWAY_KEY`.

**Severity legend:** 🔴 High · 🟠 Medium · 🟡 Low · 🔵 Info · ⚪ By-design (documented).

---

## 2. Findings summary

| # | Finding | Severity | Framework | Status |
|---|---|---|---|---|
| F1 | Vulnerable dependencies (LangChain/LangGraph/Starlette/Chroma/pytest) | 🟠 Medium | LLM03, A06 | **Remediated in v0.2.0** — upgraded to patched majors; 1 no-fix residual (chromadb, accepted) (§6) |
| F2 | Arbitrary code execution in code-gen smoke-check when `RUN_CODE_CHECK=1` | 🟠 Medium (when enabled) | LLM05/LLM06, A03 | **Hardened** — off by default; RCE warning + sandbox guidance added |
| F3 | No authentication / authorization on endpoints | ⚪ By-design | LLM06, A01/A07 | Documented; production guidance in §7 |
| F4 | No request rate-limiting / unbounded LLM consumption | 🟡 Low | LLM10 | **Partially fixed** — input length/bounds caps added; rate-limit is deploy-time |
| F5 | No CORS / security headers / TLS | ⚪ By-design | A02/A05 | Documented; production guidance in §7 |
| F6 | Prompt injection via untrusted input reaching tools/LLM | 🟡 Low (mitigated) | LLM01 | Mitigated by tool allow-lists/validators (§9); residual risk documented |
| F7 | `try/except/pass` swallows errors (3 sites) | 🟡 Low | A09 | Benign cleanup paths; noted |
| F8 | OpenAPI docs (`/docs`, `/openapi.json`) exposed | 🔵 Info | A05 | Acceptable for examples; disable in prod |

No secrets are committed (`.env` is gitignored; verified by manual scan and the repo's
own pre-commit checks). No SQL injection, no SSRF in app code, no unsafe deserialization
in app code, and no `eval`/`shell=True` were found.

---

## 3. OWASP Top 10 for LLM Applications (2025)

| ID | Risk | Assessment |
|---|---|---|
| **LLM01** Prompt Injection | 🟡 **Mitigated.** All use cases pass untrusted text to the model. Impact is bounded because tools are constrained: the SQL agent only runs validated single read-only `SELECT`s, the ReAct calculator is an AST allow-list (no `eval`), search tools are read-only over a local corpus, and the code executor is off by default. Residual risk: a model could be steered to produce misleading output (see LLM09) or call an allowed tool with attacker-chosen (but safe) inputs. |
| **LLM02** Sensitive Information Disclosure | 🟢 **Good.** Secrets live only in gitignored `.env`; system prompts contain no secrets; error responses return controlled messages, not stack traces. The gateway additionally applies PII redaction (observed during the build). |
| **LLM03** Supply Chain | 🟠 **Finding F1.** 21 advisories in pinned dependencies (see §6). Models are pulled from the configured gateway only. |
| **LLM04** Data & Model Poisoning | 🟢 **Low exposure.** RAG/recommendation corpora are small, bundled, version-controlled fixtures — no ingestion of untrusted external documents at runtime. |
| **LLM05** Improper Output Handling | 🟠 **Finding F2 (mitigated).** The one place model output is *executed* is the code-gen smoke-check (off by default, now carrying an RCE warning + sandbox guidance). SQL output is validated before execution. Other outputs are returned as data, not executed/rendered as HTML. |
| **LLM06** Excessive Agency | 🟠 **Bounded.** Agents have a deliberately small, read-only/safe tool set and capped iteration counts (ReAct `max_steps`, multi-agent revise loop). The highest-agency action (code execution) is gated. HITL (UC10) is the explicit human-approval pattern. |
| **LLM07** System Prompt Leakage | 🟡 **Low.** System prompts carry no secrets; leakage would expose only benign instructions. |
| **LLM08** Vector / Embedding Weaknesses | 🟡 **Low.** Embedded Chroma with a local ONNX model; no multi-tenant vector store, no external embedding ingestion. (Note the chromadb CVE in §6 applies to its server mode, which is not used.) |
| **LLM09** Misinformation | 🟡 **Inherent.** RAG/research answers are grounded in the bundled corpus with source citations, reducing but not eliminating hallucination. Documented as inherent to LLM output. |
| **LLM10** Unbounded Consumption | 🟡 **Finding F4 (partially fixed).** `LLM_MAX_TOKENS` caps output; input length/bounds caps were added this review. No request rate-limiting or concurrency control — add at the gateway/proxy for production. |

---

## 4. OWASP Web Application Security Top 10 (2021)

| ID | Risk | Assessment |
|---|---|---|
| **A01** Broken Access Control | ⚪ **F3 (by-design).** No authn/authz on demo endpoints. Must add before exposure. |
| **A02** Cryptographic Failures | ⚪ **F5 (by-design).** No TLS in the examples (HTTP localhost). No secrets at rest beyond the gitignored `.env`. Terminate TLS at a proxy in production. |
| **A03** Injection | 🟢 **Good.** SQL agent uses a strict single-`SELECT` validator + a read-only SQLite authorizer; no string-built SQL reaches the DB unchecked. No OS command injection (`subprocess` uses an argv list, never `shell=True`). No template injection. |
| **A04** Insecure Design | 🟢 **Reasonable for scope.** Safe-by-default choices (code-exec off, read-only DB, capped loops, HITL gating). |
| **A05** Security Misconfiguration | 🔵 **F8.** `/docs` + `/openapi.json` exposed; no security headers (CSP/HSTS/X-Content-Type-Options). Low risk for a JSON API; disable docs and add headers via proxy in prod. |
| **A06** Vulnerable & Outdated Components | 🟠 **F1.** See §6. |
| **A07** Identification & Auth Failures | ⚪ **F3.** No identity layer (by-design). |
| **A08** Software & Data Integrity Failures | 🟡 **Low.** CI actions are tag-pinned (`@v5`), not SHA-pinned — consider SHA-pinning for stronger supply-chain integrity. No untrusted deserialization in app code (the LangGraph checkpoint-deserialization CVEs are dependency issues; we use in-process `MemorySaver`). |
| **A09** Security Logging & Monitoring | 🟡 **F7.** Minimal logging; 3 `try/except/pass` sites. Add structured logging/auditing for production. |
| **A10** SSRF | 🟢 **Good (app code).** The only outbound URL is the operator-configured `LLM_BASE_URL`, not user-controlled at request time. (Note: some LangChain CVEs in §6 are SSRF issues in library features we do not invoke — URL splitters, image-URL token counting.) |

---

## 5. NIST mapping

**NIST SSDF (SP 800-218)** — secure development practices:
- *PW.4 (reuse secure components):* dependency audit performed (§6); CVEs tracked.
- *PW.5/PW.6 (secure coding, static analysis):* Bandit SAST in this review; CI runs the test suite on every push.
- *PW.7/PW.8 (review & test):* unit + integration tests per project; this manual review.
- *PS.1 (protect code & secrets):* secrets gitignored; secret scanning performed; MIT-licensed.
- *RV.1 (vuln identification):* SAST + DAST + dependency + IaC scanning, documented here.
- **Gap:** dependency remediation (PW.4) pending — see §6.

**NIST AI RMF / SP 800-218A (Generative AI profile)** — AI-specific:
- *Govern/Map:* use cases and their risk levels are documented (`TRACKING.md`, this report).
- *Measure:* prompt-injection blast radius is constrained by tool design; outputs grounded/cited where applicable.
- *Manage:* HITL approval pattern (UC10), human-in-the-loop gating for high-impact actions, capped agency. Residual risks (misinformation, prompt injection) are documented rather than claimed solved.

---

## 6. Dependency audit (Finding F1)

21 advisories across 9 packages (via pip-audit + OSV.dev over the installed versions).
**Exploitability in this project is mostly low** because the vulnerable code paths
(Chroma *server* mode, LangGraph checkpoint deserialization of *untrusted* data, LangChain
URL/image features) are **not exercised** by these examples — but they should still be
upgraded for hygiene and because downstream users may use those paths.

| Package | Installed | Advisory (theme) | Fixed in | Notes for this repo |
|---|---|---|---|---|
| chromadb | 1.5.9 | CVE-2026-45829 — pre-auth code injection | (patch — see advisory) | Embedded use only; server not run → low real risk |
| langchain-core | 0.3.86 | CVE-2026-26013 SSRF; CVE-2026-34070 path traversal (`load_prompt`) | 1.2.11 | `load_prompt` not used |
| langchain-openai | 0.2.14 | CVE-2026-41488 SSRF (image-URL token counting / DNS rebinding) | 1.1.14 | image-URL token counting not used |
| langchain-text-splitters | 0.3.11 | CVE-2026-41481 SSRF (`split_text_from_url`) | 1.1.2 | URL splitter not used |
| langchain | 0.3.30 | GHSA-gr75-… path traversal / sandbox escape (file-search middleware) | 1.3.9 | that middleware not used |
| langgraph | 0.2.76 | CVE-2026-28277 unsafe msgpack checkpoint deserialization | 1.0.10 | in-process `MemorySaver`, no untrusted checkpoints |
| langgraph-checkpoint | 2.1.2 | CVE-2026-27794 / CVE-2025-64439 — RCE via checkpoint deserialization | 3.0.0 / 4.0.0 | as above |
| starlette | 0.46.2 | Multiple: Host-header poisoning, multipart DoS, Range DoS, StaticFiles SSRF (Windows) | 0.47.2+ | reachable via FastAPI; **most impactful to upgrade** |
| pytest | 8.4.2 | CVE-2025-71176 tmpdir handling | 9.0.3 | dev/test-only |

**Why not auto-fixed in this review:** the LangChain (→1.x) and LangGraph (→1.x) fixes are
**major breaking migrations** (v0→v1 API changes; LangGraph 1.0 changes the interrupt /
checkpoint APIs that UC10 relies on). Bumping them blindly would break the green test suite.

**Remediation — DONE in v0.2.0.** Pyproject pins were bumped and verified by re-running OSV
over the resolved versions and recreating venvs from scratch (CI-faithful). Outcome:

| Package | v0.1.0 | v0.2.0 (patched) | Status |
|---|---|---|---|
| langchain | 0.3.30 | **1.3.11** | ✅ clean |
| langchain-core | 0.3.86 | **1.4.8** | ✅ clean |
| langchain-openai | 0.2.14 | **1.3.3** | ✅ clean |
| langchain-text-splitters | 0.3.11 | **1.1.2** | ✅ clean |
| langgraph | 0.2.76 | **1.2.6** | ✅ clean |
| langgraph-checkpoint | 2.1.2 | **4.1.1** | ✅ clean |
| starlette (via FastAPI) | 0.46.2 | **1.3.1** | ✅ clean |
| pytest (dev) | 8.4.2 | **9.1.1** | ✅ clean |
| chromadb | 1.5.9 | 1.5.9 (no fix published) / 0.6.3 in raw-api | ⚠️ **accepted residual** |

The LangChain/LangGraph v1 migration required **no application-code changes** (the StateGraph /
`interrupt()` / `Command` / `with_structured_output` / LCEL APIs we use are compatible); the only
change was dropping the unmaintained `langchain-community` test-only dep in UC4 (research) and
importing `FakeListChatModel` from its `langchain_core` canonical path. All 33 projects' unit
tests pass; CI re-validates with fresh installs.

**chromadb residual:** the advisory (GHSA-f4j7-r4q5-qw2c) is a *server-side* pre-auth code
injection with **no published fix** as of this review. These examples use **embedded** chromadb
(no server) so it is not exploitable here; raw-api/01-rag stays on the unaffected 0.6.x line.
Tracked for upgrade once a fix ships.

---

## 7. Production hardening checklist (for anyone deploying these patterns)

These are intentionally **absent** from the examples; add them before any real deployment:
- **AuthN/AuthZ** on every endpoint (API key / OAuth / mTLS) — addresses A01/A07/LLM06.
- **TLS** termination at a reverse proxy — A02.
- **Rate limiting & concurrency caps** (per-key budgets at the gateway) — LLM10.
- **CORS** allow-list and **security headers** (CSP, HSTS, X-Content-Type-Options) — A05.
- **Disable `/docs` & `/openapi.json`** (or gate them) in production — A05.
- **Structured logging + audit trail** of prompts, tool calls, and decisions — A09 / NIST RV.
- **Never enable `RUN_CODE_CHECK`** outside a disposable, network-isolated, resource-capped sandbox — F2.
- **Keep dependencies patched** — ✅ **Dependabot is configured** (`.github/dependabot.yml`: weekly pip + github-actions updates, minor/patch grouped, majors individual). Consider also adding `pip-audit` to CI — A06/LLM03.

---

## 8. Fixes applied in this review

- **Input length & bounds caps** (`Field(max_length=…)`, integer `ge/le`) on every request
  model across all 30 projects — bounds LLM10 / basic input validation. (Unit tests: 33/33 green.)
- **Code-execution RCE warning + sandbox guidance** added to UC2 (`codegen.py` docstring +
  README "⚠️ Security" section); default remains `RUN_CODE_CHECK=0`.
- **`SECURITY.md`** disclosure policy + scope added.
- **This report** (`docs/security-review.md`).

## 9. Positive controls observed (good practices already present)

- **SQL agent:** strict single-`SELECT` validator (rejects writes/DDL/multi-statement/CTE-write
  hiding) **plus** a runtime read-only SQLite authorizer (defense in depth).
- **ReAct calculator:** AST allow-list evaluator — rejects `__import__`, calls, names; **no `eval`**.
- **Search tools:** read-only over bundled local corpora; no network.
- **HITL:** unguessable `uuid4` run-ids; thread-safe store; terminal-resume locking.
- **Secrets:** `.env` gitignored and verified clean; only `.env.example` committed.
- **Config:** OpenAI-compatible HTTP, no hardcoded provider/keys; params from env.

## 10. Reproduce the scans

```bash
# SAST
bandit -r raw-api langchain langgraph --exclude '*/.venv/*,*/tests/*'
# Dependency CVEs (per venv, or audit a frozen set against OSV)
uv pip freeze --python <venv>/python | <query OSV.dev or pip-audit -r>
# IaC / config
trivy config .github
# DAST (start a service first, then point a scanner at it; import openapi.json for depth)
uvicorn app.main:app --port 8111
docker run --rm ghcr.io/zaproxy/zaproxy zap-baseline.py -t http://host.docker.internal:8111
docker run --rm -e BURP_START_URL=http://host.docker.internal:8111 public.ecr.aws/portswigger/dastardly
```

---

## 11. Addendum — `claude-agent-sdk` approach (2026-07-29)

**Date:** 2026-07-29 · **Scope:** the 10 services in `claude-agent-sdk/` (added after
the v0.2.0 review) · **Reviewer:** Raghuveer Dendukuri with Claude Code (Opus)

This addendum covers the fourth approach only. It is kept separate rather than
merged because the threat model genuinely differs: the other three approaches
send messages to a model and act on the reply, whereas this one hands an agent a
**shell, a filesystem, and a subprocess runtime**. Several controls that are
"by-design absent" elsewhere become materially riskier here.

### 11.1 Tooling run

| Tool | Class | Scope | Result |
|---|---|---|---|
| **Bandit** 1.9.4 | Python SAST | `claude-agent-sdk/*/app` (3,955 LoC) | **0 High, 0 Medium, 1 Low** (B101 `assert`, non-security path) |
| **pip-audit** + OSV.dev | Dependency CVE audit | resolved tree (claude-agent-sdk, fastapi, uvicorn, pydantic, pydantic-settings, anyio) | **No known vulnerabilities** |
| Manual review | Code/threat | all 10 use cases | §11.2–§11.3 |
| **Live agent runs** | Behavioural | UC02, UC08, UC10 against the gateway | 5 defects found — §11.4 |

Bandit did **not** flag the one f-string reaching SQLite
(`PRAGMA table_info('{table}')` in `06-sql-agent/app/db.py`) because the table
name is validated against the real table list first. That is a deliberate,
guarded exception, and it is unit-tested with an injection payload.

### 11.2 New findings

| # | Finding | Severity | Framework | Status |
|---|---|---|---|---|
| **F9** | **Shell execution enabled by default** in `02-code-generation` | 🔴 **High** (as shipped) | LLM05/LLM06, A03 | Documented + bounded; **not eliminated** |
| **F10** | `cwd` does **not** confine agent file writes | 🟠 Medium | LLM06, A01 | Mitigated (prompt + read-back), not solved |
| **F11** | Human-approval gate fails **open** on a config mistake | 🟠 Medium | LLM06 | **Fixed** + regression test |
| **F12** | Parked HITL runs have no TTL or eviction | 🟡 Low | LLM10 | **Fixed** — deny-on-timeout reaper + capacity cap |
| **F13** | Agent runtime depends on an external CLI resolved from `PATH` | 🟡 Low | LLM03, A08 | Documented |
| **F14** | **Developer `~/.claude` memory leaked into agent context** despite `setting_sources=[]` | 🟠 Medium | LLM02, LLM07 | **Fixed** — `CLAUDE_CONFIG_DIR` isolation |

**F9 — shell execution on by default.** This is the sharpest difference from the
v0.2.0 review. There, arbitrary code execution existed only behind
`RUN_CODE_CHECK=1`, **off by default** (F2). Here, `02-code-generation` grants the
built-in `Bash` tool as its core mechanism — the agent is *supposed* to run
`pytest`. That is remote code execution by design, as the server user, driven by
model output which is in turn driven by untrusted request text (`POST /run`
`{"task": ...}`). Prompt injection in that field is therefore a **command
execution** risk, not merely a misinformation risk.

It is not "fixed", because removing it removes the use case. What is applied: a
fresh per-run temp `cwd`, an allow-list without `WebFetch`/`WebSearch`,
`max_turns` + `max_budget_usd` caps, and explicit RCE warnings in both the module
docstring and the README. **Anyone deploying UC02 must run it inside a
disposable, network-isolated, resource-capped container/VM, or use the SDK's
`sandbox` setting.** The other nine services in this folder do not grant `Bash`.

**F10 — `cwd` is not a sandbox.** Confirmed empirically during live runs, not
assumed: the `Write` tool accepts **absolute** paths, and the model repeatedly
wrote to `/tmp/solution.py` instead of the working directory. `cwd` sets where
the agent *starts*, not where it is *allowed*. Mitigations: the system prompt
mandates bare relative filenames and forbids absolute paths, and artefacts are
read back only from the workdir (so out-of-workdir writes never count as output
and `tests_passed` stays false). Neither is a boundary — the process can still
write anywhere its user can.

**F11 — the approval gate could fail open.** In `10-hitl-approval`, listing the
guarded tool in `allowed_tools` **auto-approves it before `can_use_tool` is
consulted**, so the high-risk action executed with no human involvement. A
security control silently failing open on a one-line config choice is the worst
failure mode in this folder, and it was live-only: the mocked unit tests passed.
Fixed (`allowed_tools=[]`; the tool is still supplied by its MCP server), covered
by a regression test, and the SDK itself warns (`CanUseToolShadowedWarning`).
`setting_sources=[]` is load-bearing for the same reason — allow-rules in a
developer's `~/.claude` or the repo's `.claude/` can shadow the callback
identically, and are **not** visible in that warning.

**F12 — no eviction of parked runs.** A `/run` never resolved by `/resume` leaves
an agent coroutine and its state resident indefinitely. There is no TTL, cap, or
reaper, so repeated unresolved calls are a memory/task-exhaustion vector.
Acceptable for a single-process example; production needs a timeout that denies
and reaps. Related: the design is single-worker by construction — a parked
coroutine cannot be resumed by another process.

**F14 — developer memory leaked into agent context.** `setting_sources=[]` is
documented as "SDK isolation mode", and this repo relied on it. It is **not
sufficient**: it gates `settings.json` files only, and the CLI still loads the
developer's `~/.claude` **project memory** and any parent `CLAUDE.md`. Confirmed
by direct probe — an agent with no tools, asked what memory it could see,
recited this repository's private memory index **verbatim** (five entries). It
was also causing wrong answers: UC07 was responding from that leaked context
instead of searching its own corpus.

Impact is twofold. **Disclosure (LLM02/LLM07):** whatever a developer keeps in
project memory — architecture notes, credentials-adjacent operational detail,
customer names — is injected into every agent run on that machine and can be
echoed into an API response. **Reproducibility:** a run's behaviour depended on
whose laptop it executed on, which for a public reference repo is its own kind of
defect. Fixed by pointing **`CLAUDE_CONFIG_DIR` at a throwaway directory** in
`sdk_env()` (the probe then returns `NONE VISIBLE`). Anyone copying this pattern
should assume `setting_sources=[]` alone leaves memory attached.

**F13 — external CLI dependency.** The Python SDK spawns the Claude Code CLI
(Node) found on `PATH`, and passes the gateway credential to it via subprocess
environment. This is a runtime dependency outside the Python dependency audit: a
hijacked `PATH` or compromised CLI install would see the key. Pin via
`ClaudeAgentOptions.cli_path` in hostile environments.

### 11.3 Framework re-assessment (deltas only)

| ID | Risk | Delta vs the other three approaches |
|---|---|---|
| **LLM01** Prompt Injection | 🔴 **Escalated for UC02.** Elsewhere injection is bounded by read-only tools; with `Bash` the blast radius is command execution. The other 9 services stay bounded (custom tools + validators). |
| **LLM05** Improper Output Handling | 🔴 **Escalated (F9).** Model output is *executed by design* rather than behind a default-off flag. |
| **LLM06** Excessive Agency | 🟠 **Highest in the repo.** Built-in filesystem/shell tools plus subagents. Counterweights: per-subagent tool allow-lists (UC07 gives `analyst`/`writer` **no** tools), an explicit permission gate (UC10), and hard turn/budget caps everywhere. |
| **LLM10** Unbounded Consumption | 🟢 **Stronger than elsewhere.** The only approach capping **both** turns (`max_turns`) and spend (`max_budget_usd`) per run, not just `max_tokens`. F12 is the remaining gap. |
| **LLM02** Sensitive Info Disclosure | 🟠 **Was a real leak (F14), now fixed.** Developer `~/.claude` memory reached agent context despite `setting_sources=[]`; closed via `CLAUDE_CONFIG_DIR` isolation. Positive control retained: `collect()` discards `ThinkingBlock` content so chain-of-thought never reaches API responses. Credentials travel via subprocess env and are not logged (F13 noted). |
| **LLM03** Supply Chain | 🟢 Dependency tree clean (§11.1); F13 is the non-Python addition. |
| **A03** Injection | 🟢 **Good.** The SQL agent keeps two independent defences (syntactic single-`SELECT` validator **plus** driver-level `mode=ro`); UC08's calculator is an AST allow-list with no `eval`, tested against `__import__` / `__subclasses__` payloads. |
| **A08** Integrity | 🟡 F13 (CLI resolved from `PATH`). |

### 11.4 Why the live runs mattered

Eight defects passed the mocked unit-test suite and were caught only by running
real agents (all 10 use cases, across two passes). Three were security-relevant
(**F10**, **F11**, **F14**); the rest were correctness — including a delegation
trace that silently reported nothing because the built-in tool is named `Agent`,
not `Task`. That is direct evidence that for agentic systems,
mocked tests verify *your* logic but cannot verify *the agent's behaviour* — the
gated integration tests are not optional ceremony. Full list in `TRACKING.md` →
Live-run findings.

### 11.5 Deployment checklist (in addition to §7)

- **Never expose `02-code-generation` without a real sandbox** — container/VM, no
  network, CPU/memory/PID caps, non-root, read-only root filesystem (F9).
- **Do not add the guarded tool to `allowed_tools`** in UC10, and keep
  `setting_sources=[]` so local settings cannot shadow the gate (F11).
- ~~Add a TTL/reaper for parked approvals~~ — **now built in** (`APPROVAL_TTL_SECONDS` / `APPROVAL_MAX_PENDING`); still run UC10 as a **single worker** (F12).
- **Pin `cli_path`** and control `PATH` where the runtime is untrusted (F13).
- **Treat `cwd` as ergonomics, not isolation** (F10).
- **Set `CLAUDE_CONFIG_DIR` to a throwaway directory** — `setting_sources=[]`
  alone does **not** detach developer memory or a parent `CLAUDE.md` (F14).
- Keep the per-run `max_turns` / `max_budget_usd` caps — they are the only thing
  bounding an agent loop's cost and runtime.
