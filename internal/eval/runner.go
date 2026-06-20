package eval

import (
	"fmt"
	"sort"
	"sync"

	"github.com/fedeparg/local-agentic-eval/internal/agent"
	"github.com/fedeparg/local-agentic-eval/internal/llm"
)

// RunnerConfig controls the runner's behavior.
type RunnerConfig struct {
	MaxConcurrency int     // maximum simultaneous goroutines
	JudgeThreshold float64 // minimum judge score to "pass" (e.g. 3.0 out of 5.0)
}

// DefaultRunnerConfig returns reasonable starting values.
func DefaultRunnerConfig() RunnerConfig {
	return RunnerConfig{
		MaxConcurrency: 3,
		JudgeThreshold: 3.0,
	}
}

// RunAll evaluates all cases concurrently and returns the results.
//
// TODO(7): Implement the concurrent runner. It is the most important Go pattern in the project.
//
// ── Why this concurrency design ──────────────────────────────────────────────
//
// Rejected alternative — mutex + shared slice:
//
//	mu.Lock()
//	results[i] = r    // multiple goroutines write → you need a lock
//	mu.Unlock()
//
// Problem: you share memory. The lock prevents the race condition, but
// now you have shared mutable state across goroutines.
//
// Chosen design — buffered channel:
//
//	results := make(chan EvalResult, len(cases))  // BUFFERED: capacity = number of cases
//	                                              // why buffered: the sender goroutine
//	                                              // doesn't block waiting for the receiver;
//	                                              // it can write and finish.
//
// WaitGroup + buffered channel pattern:
//
//	var wg sync.WaitGroup
//	results := make(chan EvalResult, len(cases))  // ← buffered, without this you'd deadlock
//
//	for _, c := range cases {
//	    wg.Add(1)
//	    go func(c Case) {            // ← c by VALUE to avoid the closure bug:
//	        defer wg.Done()          //   without it, every goroutine would read
//	                                 //   the last value of c when it runs.
//	        r := runOne(c, ag, judge, config)
//	        results <- r             // ← sends to the channel; doesn't block thanks to the buffer
//	    }(c)                         // ← argument passed here, not captured by reference
//	}
//
//	wg.Wait()      // ← wait for ALL goroutines to call Done()
//	close(results) // ← signal that no more values are coming; range can finish
//
//	var out []EvalResult
//	for r := range results {  // ← range over a closed channel: ends when empty + closed
//	    out = append(out, r)
//	}
//
// Semaphore to limit concurrency (bonus):
// If you don't want 50 goroutines calling the LLM simultaneously:
//
//	sem := make(chan struct{}, config.MaxConcurrency)  // capacity = limit
//	// At the start of the goroutine:
//	sem <- struct{}{}          // ← acquire a slot (blocks if MaxConcurrency are already active)
//	defer func() { <-sem }()  // ← release the slot when done
//
// ── What to implement ─────────────────────────────────────────────────────────
//
// Implement using the WaitGroup + buffered channel pattern above.
// Optionally add the semaphore to respect config.MaxConcurrency.
//
// For each case, call runOne (implemented below) and send the result to the channel.
//
// IMPORTANT: preserve the order of the cases in the output slice (sort by CaseID
// or simply accept that the order may vary — document your decision).
func RunAll(cases []Case, ag *agent.Agent, judge *llm.Client, config RunnerConfig) ([]EvalResult, error) {
	if len(cases) == 0 {
		return nil, fmt.Errorf("no cases to evaluate")
	}

	wg := sync.WaitGroup{}
	results := make(chan EvalResult, len(cases))
	sem := make(chan struct{}, config.MaxConcurrency)
	for _, c := range cases {
		wg.Add(1)
		go func(c Case) {
			defer wg.Done()
			sem <- struct{}{}
			r := runOne(c, ag, judge, config)
			<-sem
			results <- r
		}(c)
	}
	wg.Wait()
	close(results)

	output := []EvalResult{}
	for r := range results {
		output = append(output, r)
	}

	sort.Slice(output, func(i, j int) bool { return output[i].Case.ID < output[j].Case.ID })

	return output, nil
}

// runOne runs the full evaluation of a single case.
// Implemented: no TODOs here. Study how it orchestrates agent + scorer.
func runOne(c Case, ag *agent.Agent, judge *llm.Client, config RunnerConfig) EvalResult {
	result := EvalResult{Case: c}

	// 1. Run the agent
	agResult := ag.Run(c.ID, c.Task)
	result.AgentResult = agResult

	if agResult.Error != nil {
		result.Error = fmt.Errorf("agent failed: %w", agResult.Error)
		return result
	}

	// 2. Deterministic scoring
	detOK, err := ScoreDeterministic(agResult, c)
	if err != nil {
		result.Error = fmt.Errorf("deterministic score: %w", err)
		return result
	}
	result.DeterministicOK = detOK

	// 3. LLM-as-judge scoring
	judgeScore, err := ScoreLLMJudge(agResult, c, judge)
	if err != nil {
		result.Error = fmt.Errorf("judge score: %w", err)
		return result
	}
	result.JudgeScore = judgeScore

	// 4. Pass/fail decision: both axes must pass
	result.Passed = detOK && judgeScore.Total >= config.JudgeThreshold

	return result
}
