# Security Review — ai-agent-usecases-python

**Date:** 2026-06-25 · **Version reviewed:** v0.1.0 (+ post-release hardening) ·
**Reviewer:** Raghuveer Dendukuri with Claude Code (Opus)

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
| F1 | Vulnerable dependencies (LangChain/LangGraph/Starlette/Chroma/pytest) | 🟠 Medium | LLM03, A06 | **Documented — upgrade is a tracked follow-up** (§6) |
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

**Remediation plan (recommended, as a dedicated PR):**
1. Low-risk first: bump FastAPI to pull **Starlette ≥ 0.47.2**; bump **pytest** (dev).
2. Then the **LangChain 1.x / LangGraph 1.x** migration on a branch — update imports and the
   UC10 interrupt/checkpoint code, re-run the full suite + CI, and ship as v0.2.0.

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
- **Keep dependencies patched** (Dependabot/Renovate + CI `pip-audit`) — A06/LLM03.

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
