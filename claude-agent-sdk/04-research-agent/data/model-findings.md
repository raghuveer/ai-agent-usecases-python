# Empirical model findings

Notes gathered while building the ai-usecases examples against the local
AI Utility Platform gateway.

- Small local models (`qwen3:1.7b`, `qwen2.5-coder:1.5b`) handle retrieval-
  augmented answering, intent classification, ranking, and SQL generation
  reliably at zero marginal cost.
- The same models could **not** drive a multi-step text ReAct loop. They emitted
  prose instead of `Action:` lines, drew on outside knowledge instead of tool
  results, and hit the step ceiling without ever reaching `Final Answer`.
- Strict-schema JSON extraction was likewise unreliable on the 1.5b/1.7b models;
  output drifted out of schema often enough to need a retry path.
- `claude-haiku` handled both cases without special prompting.
- Adding `stop=["Observation:"]` was necessary for the text-ReAct approaches, or
  the model hallucinated its own observations instead of waiting for real tool
  results.
