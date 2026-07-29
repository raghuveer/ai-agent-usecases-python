# Gateway behaviour notes

- The platform gateway performs Presidio-style PII redaction on inbound
  requests. It is context-sensitive and probabilistic: the same string may pass
  one call and be masked the next.
- Redaction has masked brand names and, notably, the literal ReAct field label
  `Action Input`, replacing it with `<PERSON>`. Renaming the field to
  `Arguments:` avoided the collision.
- The gateway exposes two surfaces: an OpenAI-compatible
  `/v1/chat/completions` and an Anthropic-compatible `/v1/messages`.
- Authentication is `Authorization: Bearer <virtual-key>`. Presenting the key as
  `x-api-key` is rejected with 401.
- Model names must be allow-listed aliases scoped to the virtual key; raw
  Ollama tags are not accepted through the gateway.
