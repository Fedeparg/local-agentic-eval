# Local Agentic Eval Harness

A **ReAct agent** with a **concurrent evaluation harness**, built in Go with [Ollama](https://ollama.com) as the local LLM backend.

This project was completed as a university practice in Applied AI Engineering, covering three pillars of the discipline: **agentic AI**, **LLM evaluation**, and **idiomatic concurrency in Go**.

---

## What it does

The harness runs a ReAct (Reasoning + Acting) agent against a dataset of evaluation cases and scores each one using two complementary methods:

- **Deterministic scoring** — exact match, contains, regex, JSON schema validation
- **LLM-as-judge** — a second LLM evaluates the agent's reasoning trajectory against a case-specific rubric

All cases run concurrently (goroutines + buffered channel + semaphore), and results are compiled into a Markdown report.

---

## Architecture

```
cmd/harness/main.go
    │
    ├── internal/agent/
    │   ├── agent.go      ← ReAct loop (Thought → Action → Observation)
    │   └── tools.go      ← calculator, lookup, flaky_api
    │
    ├── internal/eval/
    │   ├── runner.go     ← concurrent runner (WaitGroup + buffered channel)
    │   ├── scorer.go     ← deterministic scoring + LLM judge
    │   └── reporter.go   ← Markdown report generator
    │
    └── internal/llm/
        └── ollama.go     ← HTTP client for Ollama /api/generate
```

**Data flow per case:**
```
Case (JSON) → Agent.Run() → AgentResult (trajectory) → Scorer → EvalResult → Reporter
                  ↑
             Tool.Execute()
```

---

## Tools available to the agent

| Tool | Description |
|------|-------------|
| `calculator` | Evaluates arithmetic expressions (`"15 * 23"`) |
| `lookup` | Case-insensitive search over a local knowledge base |
| `flaky_api` | Simulated external API with 20% random failure rate |

---

## ReAct loop

Each agent step follows the ReAct format:

```
Thought: I need to calculate 15 * 23
Action: calculator
Action Input: {"expression": "15 * 23"}
Observation: 345
Thought: I have the result
Final Answer: 345
```

The loop runs up to `maxSteps` iterations. Tool errors are fed back as `Observation` so the agent can recover — only LLM errors abort the run.

---

## Evaluation dataset

14 cases across 5 categories in `data/golden_dataset.json`:

| Category | Cases | What it tests |
|----------|-------|---------------|
| `calc-*` | 3 | Arithmetic reasoning and tool use |
| `lookup-*` | 4 | Knowledge base retrieval |
| `api-*` | 3 | Error handling with a flaky API |
| `multi-*` | 3 | Multi-step tasks combining tools |
| `edge-*` | 1 | Edge cases (no tools needed) |

---

## Concurrency design

```go
results := make(chan EvalResult, len(cases))  // buffered: senders don't block
sem     := make(chan struct{}, maxConcurrency) // semaphore: limits parallel LLM calls

for _, c := range cases {
    wg.Add(1)
    go func(c Case) {           // c passed by value — avoids the closure bug
        defer wg.Done()
        sem <- struct{}{}       // acquire slot
        defer func() { <-sem }()
        results <- runOne(c, ...)
    }(c)
}
wg.Wait()
close(results)
```

> "Don't communicate by sharing memory; share memory by communicating." — Rob Pike

---

## Setup

### Requirements

- Go 1.22+
- [Ollama](https://ollama.com) running locally

```bash
ollama pull qwen2.5:3b
ollama serve
```

### Build and run

```bash
git clone https://github.com/fedeparg/local-agentic-eval
cd local-agentic-eval

go build ./cmd/harness/

# Run a single case
./harness -case calc-001

# Run all cases (default concurrency: 3)
./harness

# Run with custom concurrency and save report
./harness -concurrency 5 -output report.md

# Check for data races
go run -race ./cmd/harness/
```

---

## Sample output

```
✓ Connected to Ollama (http://localhost:11434), model: qwen2.5:3b
→ Running 14 evaluation case(s)...

# Eval Report - 2026-06-20 18:10:03
**Result: 7/14 cases passed**

| ID        | Task                                      | Det | Judge | Pass |
|-----------|-------------------------------------------|-----|-------|------|
| calc-001  | Compute the result of 15 multiplied by... | ✓   | 4.7   | ✓    |
| lookup-003| In what year was the 'Attention is All... | ✓   | 3.3   | ✓    |
| multi-001 | The 'Attention is All You Need' paper...  | ✓   | 4.7   | ✓    |
...
```

---

## Key Go patterns practiced

- `(T, error)` return pattern — never `panic` in tool code
- `sync.WaitGroup` + buffered channel for concurrent fan-out
- Semaphore with `make(chan struct{}, N)` to cap parallelism
- `json.Unmarshal` into anonymous structs with struct tags
- `strings.Builder` for efficient string accumulation
- Closure bug with loop variables and how to avoid it

---

## Known limitations

- The parser is line-based: if the LLM breaks the ReAct format (mixing fields on one line), `parseResponse` degrades gracefully but loses precision
- `qwen2.5:3b` is non-deterministic — results vary between runs (7/14 to 2/14 observed in practice)
- The calculator evaluates left-to-right without operator precedence (`2 + 3 * 4 = 20`, not `14`)
