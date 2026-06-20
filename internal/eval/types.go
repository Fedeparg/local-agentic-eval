// Package eval contains the evaluation harness: types, runner, scorer and reporter.
// This file defines the shared types. No TODOs here.
package eval

import "github.com/fedeparg/local-agentic-eval/internal/agent"

// ─── Dataset types ────────────────────────────────────────────────────────────

// Case is a test case from the golden dataset.
type Case struct {
	ID   string `json:"id"`
	Task string `json:"task"`

	ExpectedOutput ExpectedOutput `json:"expected_output"`
	Scoring        CaseScoring    `json:"scoring"`
}

// ExpectedOutput describes what is expected from the agent.
type ExpectedOutput struct {
	FinalAnswer string   `json:"final_answer"` // expected answer (may be empty for "none" cases)
	ToolsUsed   []string `json:"tools_used"`   // expected tools (informational)
}

// CaseScoring groups both evaluation axes per case.
type CaseScoring struct {
	Deterministic DeterministicScoring `json:"deterministic"`
	LLMJudge      LLMJudgeScoring      `json:"llm_judge"`
}

// DeterministicScoring defines the exact verification logic.
// Type can be: "exact_match", "contains", "regex", "schema", "none".
type DeterministicScoring struct {
	Type  string `json:"type"`  // which kind of comparison to perform
	Value string `json:"value"` // expected value (regex, substring or exact text)
}

// LLMJudgeScoring defines the prompt and axes for the LLM judge.
type LLMJudgeScoring struct {
	Rubric    string   `json:"rubric"`     // case-specific instruction
	ScoreAxes []string `json:"score_axes"` // names of the axes to score (0-5 each)
}

// ─── Result types ─────────────────────────────────────────────────────────────

// JudgeScore holds the LLM judge's evaluation.
type JudgeScore struct {
	Axes        map[string]int `json:"scores"`      // axis_name → score 0-5
	Explanation string         `json:"explanation"` // one sentence from the judge
	Total       float64        `json:"-"`           // average computed locally
}

// EvalResult is the complete result of evaluating a case.
type EvalResult struct {
	Case            Case
	AgentResult     agent.AgentResult
	DeterministicOK bool
	JudgeScore      JudgeScore
	Passed          bool  // true if deterministic OK AND judge total >= threshold
	Error           error // runner error (not the agent's; agent errors go in AgentResult.Error)
}
