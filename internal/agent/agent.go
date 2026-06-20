package agent

import (
	"fmt"
	"strings"
	"time"

	"github.com/fedeparg/local-agentic-eval/internal/llm"
)

// reactPromptTemplate is the base system prompt for the ReAct agent.
// Note: small models (3B) are very sensitive to prompt formatting.
// If the agent does not follow the format, adjust the wording (more directive, less text).
const reactPromptTemplate = `You have access to these tools:
%s
Use EXACTLY this format on every iteration:

Thought: <your reasoning>
Action: <tool_name>
Action Input: <valid JSON>
Observation: <result — provided by the system>

Repeat up to 5 times. When you have the final answer, write:
Final Answer: <answer>

Rules:
- Always start with Thought.
- Action Input must be valid JSON.
- If a tool fails, the error appears in Observation; decide whether to retry or answer with what you have.
- ALWAYS end with "Final Answer:" even if you don't have complete information.

Task: %s
`

// ─── Types ────────────────────────────────────────────────────────────────────

// Step represents a complete iteration of the ReAct loop.
type Step struct {
	Thought     string
	Action      string
	ActionInput string
	Observation string
}

// AgentResult holds the full trajectory and result of a run.
// Error here means the agent could not run (not that it failed the task).
type AgentResult struct {
	CaseID      string
	Task        string
	Steps       []Step
	FinalAnswer string
	Error       error
	Duration    time.Duration
}

const maxSteps = 5

// ─── Agent ────────────────────────────────────────────────────────────────────

// Agent runs the ReAct loop using an LLM and a set of tools.
type Agent struct {
	llm      *llm.Client
	registry *ToolRegistry
}

// New builds an Agent. Go pattern: explicit constructor instead of a struct literal.
func New(llmClient *llm.Client, registry *ToolRegistry) *Agent {
	return &Agent{llm: llmClient, registry: registry}
}

// Run executes the ReAct loop for the given task and returns the full trajectory.
//
// TODO(4): Implement the ReAct control loop. It is the core of the project.
//
// Loop architecture:
//
//  1. Build the initial prompt with reactPromptTemplate (use fmt.Sprintf).
//     Pass registry.Descriptions() as the list of tools.
//
//  2. Loop up to maxSteps (or until you find "Final Answer:"):
//
//     a) Call a.llm.Generate(prompt) → (response, err)
//     If err != nil: record the error in result.Error and return immediately.
//     Go pattern: an LLM error is unrecoverable at this level; propagate it.
//
//     b) Call parseResponse(response) to extract:
//     thought, action, actionInput, finalAnswer
//
//     c) If finalAnswer != "": store it in result.FinalAnswer and break the loop.
//
//     d) Execute the tool:
//     tool, ok := a.registry.Get(action)
//     If !ok: observation = "Error: unknown tool: " + action
//     If ok: output, err := tool.Execute(actionInput)
//     If err != nil: observation = "Error: " + err.Error()
//     If err == nil: observation = output
//     Go pattern: do NOT propagate the tool error upward — the agent
//     must see the error as an Observation and decide what to do.
//
//     e) Append the Step to the history:
//     result.Steps = append(result.Steps, Step{...})
//
//     f) Extend the prompt with the completed step for the next iteration:
//     prompt += "\nThought: " + thought + "\nAction: " + action + ...
//     Use a strings.Builder for efficiency if you have many steps.
//
//  3. If maxSteps is reached without "Final Answer:", record in result.FinalAnswer:
//     "Maximum number of steps reached without a final answer."
//
// Design question: what happens if the LLM does not follow the format and returns
// free text without "Action:"? Your parseResponse must handle it gracefully.
// Hint: if action == "" and finalAnswer == "", treat the whole response as
// finalAnswer to avoid getting stuck.
func (a *Agent) Run(caseID, task string) AgentResult {
	start := time.Now()
	result := AgentResult{
		CaseID: caseID,
		Task:   task,
	}

	prompt := fmt.Sprintf(reactPromptTemplate, a.registry.Descriptions(), task)
	gotFinalAnswer := false
	for i := 0; i < 5 && !gotFinalAnswer; i++ {
		response, err := a.llm.Generate(prompt)
		if err != nil { // Irrecoverable at this point
			result.Error = err
			return result
		}

		thought, action, actionInput, finalAnswer := parseResponse(response)
		if finalAnswer != "" {
			result.FinalAnswer = finalAnswer
			gotFinalAnswer = true
			break
		} else if action == "" && finalAnswer == "" {
			result.FinalAnswer = response
			gotFinalAnswer = true
			break
		}

		observation := ""
		tool, ok := a.registry.Get(action)
		if !ok {
			observation = fmt.Sprintf("Error: unknown tool: %v", action)
		} else {
			output, err := tool.Execute(actionInput)
			if err != nil {
				observation = "Error: " + err.Error()
			} else {
				observation = output
			}
		}
		currentStep := Step{Thought: thought, Action: action, ActionInput: actionInput, Observation: observation}
		result.Steps = append(result.Steps, currentStep)

		// Build the new prompt
		sb := strings.Builder{}
		sb.WriteString(prompt)
		sb.WriteString("\n")
		sb.WriteString(fmt.Sprintf("Thought: %v\nAction: %v\nAction Input: %v\nObservation: %v\n", thought, action, actionInput, observation))
		prompt = sb.String()
	}

	if !gotFinalAnswer {
		result.FinalAnswer = "Maximum number of steps reached without a final answer."
	}

	// result.Error = fmt.Errorf("TODO(4): implement Agent.Run")
	result.Duration = time.Since(start)
	return result
}

// parseResponse extracts the ReAct-format fields from the LLM response.
//
// TODO(4c — helper): Implement the parser. It is easier than it looks.
//
// Strategy:
//
//  1. Split the response into lines: lines := strings.Split(response, "\n")
//
//  2. Walk the lines looking for known prefixes:
//     - "Thought:"      → store the rest in thought
//     - "Action:"       → store the rest in action
//     CAREFUL: "Action Input:" starts with "Action" — check "Action Input:" first.
//     - "Action Input:" → store the rest in actionInput (may be multi-line JSON)
//     - "Final Answer:" → store the rest in finalAnswer
//
//  3. For prefixes use strings.HasPrefix(line, "Thought:") and
//     strings.TrimSpace(strings.TrimPrefix(line, "Thought:"))
//
//  4. For multi-line Action Input (JSON with newlines): accumulate lines until
//     the next one starts with "Observation:", "Thought:" or "Final Answer:".
//
// Returns the 4 values; those not present in the response are "".
func parseResponse(response string) (thought, action, actionInput, finalAnswer string) {
	lines := strings.Split(response, "\n")
	field := make(map[string]string)
	currentField := ""

	for _, line := range lines {
		switch {
		case strings.HasPrefix(line, `Thought:`):
			currentField = `thought`
			field[`thought`] += strings.TrimSpace(strings.TrimPrefix(line, `Thought:`))
		case strings.HasPrefix(line, `Action Input:`):
			currentField = `actionInput`
			field[`actionInput`] += strings.TrimSpace(strings.TrimPrefix(line, `Action Input:`))
		case strings.HasPrefix(line, `Action:`):
			currentField = `action`
			field[`action`] += strings.TrimSpace(strings.TrimPrefix(line, `Action:`))
		case strings.HasPrefix(line, `Final Answer:`):
			currentField = `final`
			field[`final`] += strings.TrimSpace(strings.TrimPrefix(line, `Final Answer:`))
		default:
			if currentField != "" {
				field[currentField] += "\n" + line
			}
		}
	}
	return field[`thought`], field[`action`], field[`actionInput`], field[`final`]
}
