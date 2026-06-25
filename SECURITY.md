# Security Policy

## Scope & intent

This repository is a **collection of reference/example applications** demonstrating
ten LLM agent use cases across three approaches. It is intentionally **not
production-hardened**: the example services ship without authentication, CORS,
rate limiting, or TLS, and several use cases deliberately expose powerful
capabilities (code execution, SQL execution, tool use) to illustrate the pattern.

**Do not deploy these examples as-is on an untrusted network or with untrusted
input.** See [`docs/security-review.md`](docs/security-review.md) for a full
threat analysis mapped to NIST, the OWASP Top 10 for LLM Applications, and the
OWASP Web Application Security Top 10, including which protections are present,
which are intentionally omitted, and what you must add before production use.

## Supported versions

| Version | Supported |
|---------|-----------|
| `0.1.x` | ✅ (current) |
| `< 0.1` | ❌ |

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue for an
unfixed vulnerability.

- Preferred: GitHub **Private Vulnerability Reporting** — the *Security* tab →
  *Report a vulnerability* on this repository.
- Alternatively: email the maintainer at the address on the commit history.

Please include: affected use case / approach / file, a description, reproduction
steps or a proof of concept, and impact. We aim to acknowledge within a few days.
As this is an examples project maintained on a best-effort basis, please allow
reasonable time for a fix before any public disclosure.

## What is in scope

- Issues in the example application code (`raw-api/`, `langchain/`, `langgraph/`).
- Vulnerable pinned dependencies (see the dependency-audit section of the review).
- Secret leakage in the repository.

## What is out of scope

- The absence of auth/CORS/rate-limiting/TLS on the demo endpoints — this is
  documented and by design (see the review's "intentionally omitted" section).
- The AI Utility Platform / gateway used during local development (separate project).
- Findings that require enabling an explicitly off-by-default, clearly-labelled
  dangerous feature (e.g. `RUN_CODE_CHECK` in the code-generation use case)
  without applying the documented sandboxing guidance.
