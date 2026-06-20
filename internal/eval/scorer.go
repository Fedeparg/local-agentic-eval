package eval

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/fedeparg/local-agentic-eval/internal/agent"
	"github.com/fedeparg/local-agentic-eval/internal/llm"
)

// judgePromptTemplate: the prompt sent to the LLM-as-judge.
//
// TODO(6a — design): Read this template and improve it before implementing ScoreLLMJudge.
// Question: Why do we ask for strict JSON instead of free text?
// Question: What biases can a small (3B parameter) LLM judge have?
//   - Verbosity bias: tends to score longer answers higher.
//   - Self-preference: prefers answers in its own style.
//   - Anchoring: the first option in a prompt tends to score better.
//
// Question: How does the rubric change if the case uses flaky_api vs calculator?
const judgePromptTemplate = `You are a judge for evaluating AI agents. Evaluate the following trajectory.

TASK: %s
EXPECTED ANSWER: %s

AGENT TRAJECTORY:
%s

CASE-SPECIFIC RUBRIC:
%s

Score the following axes from 0 to 5 (0=very bad, 5=excellent):
%s

IMPORTANT: Respond ONLY with this exact JSON, with no text before or after.
The "scores" field is an object with the axes as keys:
{"scores": {%s}, "explanation": "one sentence explaining your evaluation"}
`

// ScoreDeterministic checks whether the agent's answer meets the exact criteria.
//
// TODO(5): Implement the five scoring types.
//
// Types:
//   - "exact_match": both strings (after TrimSpace and ToLower) must be equal.
//   - "contains":    the answer contains the expected value (case-insensitive).
//   - "regex":       regexp.Compile(value).MatchString(answer) — use Compile,
//     not MustCompile, and return an error if the regex is invalid.
//   - "schema":      validate that FinalAnswer contains valid JSON (use extractJSON to
//     pull the JSON out of the agent's text). If c.Scoring.Deterministic.Value is not
//     empty, it is a comma-separated list of required keys ("key1,key2"); verify
//     the JSON contains them as root-level fields (map[string]any).
//   - "none":        always returns (true, nil) — for cases with a random outcome.
//
// Go pattern: the switch has no fall-through by default. Use `return` in each branch.
// Return (false, fmt.Errorf("unknown type: %q", type)) for unrecognized types.
//
// Note on (bool, error): you return both because an invalid regex error
// or a malformed schema.Value are system failures, not agent failures.
func ScoreDeterministic(result agent.AgentResult, c Case) (bool, error) {
	comparaisonType := c.Scoring.Deterministic.Type
	expectedValue := c.Scoring.Deterministic.Value

	switch comparaisonType {
	case "exact_match":
		if strings.EqualFold(strings.TrimSpace(expectedValue), strings.TrimSpace(result.FinalAnswer)) {
			return true, nil
		}
		return false, nil
	case "contains":
		if strings.Contains(strings.ToLower(result.FinalAnswer), strings.ToLower(expectedValue)) {
			return true, nil
		}
		return false, nil
	case "regex":
		regex, err := regexp.Compile(expectedValue)
		if err != nil {
			return false, err
		}
		return regex.MatchString(result.FinalAnswer), nil
	case "schema":
		mapOfKeys := make(map[string]any)
		toExtract := extractJSON(result.FinalAnswer)
		err := json.Unmarshal([]byte(toExtract), &mapOfKeys)
		if err != nil {
			return false, nil
		}
		if expectedValue == "" {
			return true, nil
		}
		keysToCheck := strings.Split(expectedValue, ",")
		for _, key := range keysToCheck {
			if _, ok := mapOfKeys[key]; !ok {
				return false, nil
			}
		}
		return true, nil

	case "none":
		return true, nil
	}

	return false, fmt.Errorf("unknown type: %q", comparaisonType)
}

// ScoreLLMJudge uses an LLM to evaluate the agent's trajectory against a rubric.
//
// TODO(6): Implement the LLM judge. It is the most interesting part of the project.
//
// Steps:
//
//	a) Build the trajectory string with buildTrajectory(result).
//
//	b) Format the score axes for the prompt:
//	   For each axis in c.Scoring.LLMJudge.ScoreAxes:
//	     axesDesc += "- " + axis + "\n"
//	   And the scores JSON template:
//	     scoresJSON += `"` + axis + `": 0-5, `  (example: "correctness": 0-5)
//
//	c) Build the prompt with fmt.Sprintf(judgePromptTemplate, ...).
//
//	d) Call judge.Generate(prompt) → (response, err).
//	   If err != nil: return (JudgeScore{}, fmt.Errorf("LLM judge: %w", err)).
//
//	e) Extract the JSON from the response with extractJSON(response).
//	   The LLM may add text before or after the JSON — extractJSON handles it.
//
//	f) Unmarshal into a local struct:
//	   var raw struct {
//	       Scores      map[string]int `json:"scores"`  // ← NOTE: the JSON template uses key "scores"
//	       Explanation string         `json:"explanation"`
//	   }
//	   If it fails: return (JudgeScore{}, fmt.Errorf("parse judge response: %w", err)).
//
//	g) Compute the average of all axes for JudgeScore.Total.
//
//	h) Return (JudgeScore{Axes: raw.Scores, Explanation: ..., Total: ...}, nil).
//
// Error handling: NEVER return a JudgeScore with Total=0 and no error if the LLM
// actually failed — that would make every case look like it failed.
func ScoreLLMJudge(result agent.AgentResult, c Case, judge *llm.Client) (JudgeScore, error) {
	trajectory := buildTrajectory(result)
	axesDesc := ""
	scoresJSON := ""
	for _, axis := range c.Scoring.LLMJudge.ScoreAxes {
		axesDesc += "- " + axis + "\n"
		scoresJSON += `"` + axis + `": 0-5`
	}
	prompt := fmt.Sprintf(judgePromptTemplate, c.Task, c.ExpectedOutput, trajectory, c.Scoring.LLMJudge.Rubric, axesDesc, scoresJSON)

	response, err := judge.Generate(prompt)
	if err != nil {
		return JudgeScore{}, fmt.Errorf("LLM judge: %w", err)
	}

	jsonResponse := extractJSON(response)

	var raw struct {
		Scores      map[string]int `json:"scores"`
		Explanation string         `json:"explanation"`
	}
	if err := json.Unmarshal([]byte(jsonResponse), &raw); err != nil {
		return JudgeScore{}, fmt.Errorf("parse judge response: %w", err)
	}

	meanScore := 0.0
	for _, score := range raw.Scores {
		meanScore += float64(score)
	}
	meanScore = meanScore / float64(len(raw.Scores))

	return JudgeScore{Axes: raw.Scores, Explanation: raw.Explanation, Total: meanScore}, nil
}

// ─── Implemented helpers ──────────────────────────────────────────────────────

// extractJSON finds the first JSON object {...} in a string.
// Handles the common case where the LLM adds a preamble before the JSON.
// Do not modify this function.
func extractJSON(s string) string {
	start := strings.Index(s, "{")
	end := strings.LastIndex(s, "}")
	if start == -1 || end == -1 || end < start {
		return s // return the original string; Unmarshal will fail with a descriptive error
	}
	return s[start : end+1]
}

// buildTrajectory formats the agent's steps as readable text for the judge.
// Do not modify this function.
func buildTrajectory(result agent.AgentResult) string {
	var sb strings.Builder
	for i, step := range result.Steps {
		fmt.Fprintf(&sb, "Step %d:\n", i+1)
		fmt.Fprintf(&sb, "  Thought: %s\n", step.Thought)
		fmt.Fprintf(&sb, "  Action: %s\n", step.Action)
		fmt.Fprintf(&sb, "  Action Input: %s\n", step.ActionInput)
		fmt.Fprintf(&sb, "  Observation: %s\n\n", step.Observation)
	}
	fmt.Fprintf(&sb, "Final Answer: %s\n", result.FinalAnswer)
	return sb.String()
}

// Ensures the regexp and json imports are used even while the TODOs are empty.
// Remove these lines when you implement the TODOs.
var (
	_ = regexp.Compile
	_ = json.Unmarshal
)
