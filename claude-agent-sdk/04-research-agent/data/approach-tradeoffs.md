# Approach trade-offs

- **raw-api** makes every byte sent to the model explicit, which is ideal for
  teaching, and correspondingly verbose for coordination-heavy work. Multi-agent
  orchestration and human-in-the-loop approval were both judged impractical to
  build this way beyond a minimal demo.
- **langchain** reaches a working prototype fastest for linear chains, and gets
  awkward once control flow needs cycles or conditional routing.
- **langgraph** pays a structural cost up front — a typed state graph — and
  earns it back on cycles, conditional edges, and durable pause/resume via
  `interrupt()` plus a checkpointer.
- **claude-agent-sdk** supplies the agent harness itself: built-in file and
  shell tools, subagents, hooks, and a permission callback. It is the least code
  for agentic work and the poorest fit where no agent loop is wanted at all.
- No approach is best across all ten use cases; the matrix is the point.
