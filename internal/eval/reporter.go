package eval

import (
	"fmt"
	"strings"
	"time"
)

// GenerateReport produces a Markdown report from the evaluation results.
//
// TODO(8): Implement the report generator. It is the easiest one — pure string logic.
//
// Report structure (follow this order):
//
//  1. Header with timestamp and summary (X/Y cases passed):
//     # Eval Report — 2026-06-16 14:30:00
//     **Result: 8/13 cases passed**
//
//  2. Summary table (one row per case):
//     | ID | Task (summary) | Det | Judge | Pass |
//     |----|----------------|-----|-------|------|
//     | calc-001 | Compute 15 * 23 | ✓ | 4.3 | ✓ |
//     Truncate long tasks: if task > 45 chars, use task[:42]+"..."
//
//  3. Judge per-axis (average across all cases):
//     ## Scores by Axis (average)
//     | Axis | Average |
//     For each axis, sum all scores and divide by the number of cases.
//     Hint: some cases may not have certain axes — handle the nil map case.
//
//  4. Failures section (only cases with Passed==false):
//     ## Failed Cases
//     For each failed case show: ID, why it failed (Det/Judge/Error),
//     and the full trajectory with buildTrajectory(r.AgentResult).
//
// Go pattern: use strings.Builder for efficiency.
//
//	var sb strings.Builder
//	fmt.Fprintf(&sb, "# Eval Report — %s\n\n", time.Now().Format("2006-01-02 15:04:05"))
//	return sb.String()
//
// Go date format: time.Now().Format("2006-01-02 15:04:05")
// The magic date "2006-01-02 15:04:05" is Go's reference time
// (02 Jan 2006 15:04:05 MST) — it is not a placeholder, it is the actual reference.
func GenerateReport(results []EvalResult) string {
	// Silences the compiler while unimplemented:
	currentTime := time.Now()
	sb := strings.Builder{}

	fmt.Fprintf(&sb, "# Eval Report - %v\n", currentTime.Format("2006-01-02 15:04:05"))
	passed := 0
	for _, result := range results {
		if result.Passed {
			passed++
		}
	}
	fmt.Fprintf(&sb, "**Result: %v/%v cases passed**\n", passed, len(results))

	// TABLE
	fmt.Fprintf(&sb, "\n|ID|Task(summary)|Det|Judge|Pass|\n")
	fmt.Fprintf(&sb, "|---|---|---|---|---|\n")

	for _, result := range results {
		task := result.Case.Task
		if len(task) > 45 {
			task = task[:42] + "..."
		}
		fmt.Fprintf(&sb, "|%v|%v|%v|%v|%v|\n", result.Case.ID, task, result.DeterministicOK, result.JudgeScore, result.Passed)
	}

	// AXIS
	fmt.Fprintf(&sb, "## Score by Axis (Average)\n")
	fmt.Fprintf(&sb, "|Axis|Average|\n")
	fmt.Fprintf(&sb, "|---|---|\n")
	scores := make(map[string]int)
	appearances := make(map[string]int)
	for _, result := range results {
		for key, value := range result.JudgeScore.Axes {
			scores[key] += value
			appearances[key]++
		}
	}
	means := make(map[string]float64)
	for key, value := range scores {
		means[key] = float64(value) / float64(appearances[key])
	}
	for key, value := range means {
		fmt.Fprintf(&sb, "|%v|%v|\n", key, value)
	}

	// Falied Cases
	fmt.Fprintf(&sb, "\n\n## Failed Cases\n")
	for _, result := range results {
		if !result.Passed {
			fmt.Fprintf(&sb, "%v - Deterministic: %v - Judge: %v - Errors: %v\n", result.Case.ID, result.DeterministicOK, result.JudgeScore, result.Error)
			fmt.Fprintf(&sb, buildTrajectory(result.AgentResult)+"\n")
		}
	}

	return sb.String()
}
