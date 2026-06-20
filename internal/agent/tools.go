// Package agent contains the ReAct agent and its tools.
package agent

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

// ─── Tool interface ───────────────────────────────────────────────────────────

// Tool is the contract every agent tool must satisfy.
// In Go, interfaces are kept small: one behavior = one interface.
type Tool interface {
	Name() string        // identifier the LLM uses in "Action:"
	Description() string // text injected into the system prompt
	Execute(input string) (string, error)
}

// ─── ToolRegistry ─────────────────────────────────────────────────────────────

// ToolRegistry keeps a name→tool map.
// Implemented: no TODOs here.
type ToolRegistry struct {
	tools map[string]Tool
}

// NewToolRegistry builds the registry from a variadic list of tools.
func NewToolRegistry(tools ...Tool) *ToolRegistry {
	r := &ToolRegistry{tools: make(map[string]Tool, len(tools))}
	for _, t := range tools {
		r.tools[t.Name()] = t
	}
	return r
}

// Get returns the tool with that name and whether it exists.
func (r *ToolRegistry) Get(name string) (Tool, bool) {
	t, ok := r.tools[name]
	return t, ok
}

// Descriptions returns the text block for the system prompt.
func (r *ToolRegistry) Descriptions() string {
	var sb strings.Builder
	for _, t := range r.tools {
		fmt.Fprintf(&sb, "- %s: %s\n", t.Name(), t.Description())
	}
	return sb.String()
}

// ─── TODO(1): CalculatorTool ──────────────────────────────────────────────────

// CalculatorTool evaluates basic arithmetic expressions.
// Expected JSON input: {"expression": "15 * 23"}
// Note: the calculator requires spaces between tokens ("15 * 23", NOT "15*23").
type CalculatorTool struct{}

func (c *CalculatorTool) Name() string { return "calculator" }
func (c *CalculatorTool) Description() string {
	return `Evaluates simple arithmetic expressions. JSON input: {"expression": "15 * 23"}. Operators: + - * /. Requires spaces between tokens.`
}

// Execute implements the calculator.
//
// TODO(1): Implement this function. Steps:
//
//	a) Parse the input as JSON: {"expression": "<expr>"}
//	   Use json.Unmarshal with an anonymous struct or a map[string]string.
//
//	b) Tokenize the expression with strings.Fields (split on whitespace).
//	   Example: "15 * 23" → ["15", "*", "23"]
//
//	c) Evaluate left to right:
//	   - The first token is the initial number (use strconv.ParseFloat).
//	   - Then iterate over (operator, number) pairs.
//	   - Support: +, -, *, /
//	   - Division by zero → return an error (NOT panic).
//
//	d) Return the result as a string with strconv.FormatFloat.
//	   Use 'f' as the format and -1 precision for clean decimals.
//
// Essential Go pattern: return ("", error) on any invalid input.
// NEVER use panic() here — the agent must be able to recover from a tool error.
//
// Quick manual test (after implementing):
//
//	tool := &CalculatorTool{}
//	got, err := tool.Execute(`{"expression": "15 * 23"}`)
//	// got == "345", err == nil
//	got, err = tool.Execute(`{"expression": "10 / 0"}`)
//	// got == "", err != nil
func (c *CalculatorTool) Execute(input string) (string, error) {
	var jsonParser struct {
		Expression string `json:"expression"` //Take the 'expression field'
	}
	if err := json.Unmarshal([]byte(input), &jsonParser); err != nil {
		return "", err
	}

	tokens := strings.Fields(jsonParser.Expression)

	if len(tokens) == 0 {
		return "", fmt.Errorf("malformed JSON - %v", jsonParser.Expression)
	}

	number, err := strconv.ParseFloat(tokens[0], 64)
	if err != nil {
		return "", err
	}

	for i := 1; i < len(tokens); i += 2 {
		op := tokens[i]
		num2, err := strconv.ParseFloat(tokens[i+1], 64)
		if err != nil {
			return "", err
		}

		switch op {
		case "+":
			number += num2
		case "-":
			number -= num2
		case "*":
			number = number * num2
		case "/":
			if num2 == 0 {
				return "", fmt.Errorf("division by 0")
			}
			number = number / num2
		default:
			return "", fmt.Errorf("operand is not a valid operation - %v", op)
		}
	}

	return strconv.FormatFloat(number, 'f', -1, 64), nil
}

// ─── TODO(2): LookupTool ──────────────────────────────────────────────────────

// LookupTool searches facts in a local JSON knowledge base.
// Expected JSON input: {"query": "capital of France"}
type LookupTool struct {
	knowledgeBase []KnowledgeEntry
}

// KnowledgeEntry is a row of data/knowledge_base.json.
type KnowledgeEntry struct {
	Key      string `json:"key"`
	Question string `json:"question"`
	Answer   string `json:"answer"`
	Category string `json:"category"`
}

// NewLookupTool loads the knowledge base from disk.
// Returns (nil, error) if the file does not exist or has invalid JSON.
// Go pattern: constructors that can fail return (*T, error).
func NewLookupTool(kbPath string) (*LookupTool, error) {
	data, err := os.ReadFile(kbPath)
	if err != nil {
		return nil, fmt.Errorf("read knowledge base %q: %w", kbPath, err)
	}
	var kb struct {
		Facts []KnowledgeEntry `json:"facts"`
	}
	if err := json.Unmarshal(data, &kb); err != nil {
		return nil, fmt.Errorf("parse knowledge base: %w", err)
	}
	return &LookupTool{knowledgeBase: kb.Facts}, nil
}

func (l *LookupTool) Name() string { return "lookup" }
func (l *LookupTool) Description() string {
	return `Searches facts in the local knowledge base. JSON input: {"query": "capital of France"}`
}

// Execute implements the search.
//
// TODO(2): Implement this function. Steps:
//
//	a) Parse the input as JSON: {"query": "<search_text>"}
//
//	b) Lowercase the query with strings.ToLower.
//
//	c) Iterate over l.knowledgeBase with range:
//	   - For each entry, check whether strings.Contains(lower(entry.Question), query)
//	     OR strings.Contains(lower(entry.Key), query).
//	   - On a match, return entry.Answer immediately.
//
//	d) If there is no match, return ("", error) with a descriptive message.
//	   Do NOT return ("not found", nil) — that hides the error from the agent.
//
// Go pattern: use `continue` to skip non-matching entries instead of nesting ifs.
// Return as soon as you find the first match (do not accumulate results).
func (l *LookupTool) Execute(input string) (string, error) {
	var q struct {
		Query string `json:"query"`
	}
	if err := json.Unmarshal([]byte(input), &q); err != nil {
		return "", err
	}
	query := strings.ToLower(q.Query)
	if query == "" {
		return "", fmt.Errorf("Query is empty")
	}
	for _, value := range l.knowledgeBase {
		if strings.Contains(strings.ToLower(value.Question), query) || strings.Contains(strings.ToLower(value.Key), query) {
			return value.Answer, nil
		}
	}
	return "", fmt.Errorf("Could not find that information in the Knowledge Base.")
}

// ─── TODO(3): FlakyAPITool ────────────────────────────────────────────────────

// FlakyAPITool simulates an external API that fails ~20% of the time.
// This forces you to practice idiomatic error handling in Go.
// Expected JSON input: {"endpoint": "weather", "params": {}}
type FlakyAPITool struct {
	failureRate float64
	rng         *rand.Rand
	endpoints   map[string]any // mock data per endpoint
}

// NewFlakyAPITool creates the tool with the given failure rate (0.0–1.0).
// Uses rand.New with a time seed for non-deterministic failures.
func NewFlakyAPITool(failureRate float64) *FlakyAPITool {
	endpoints := map[string]any{
		"weather": map[string]any{
			"city": "Madrid", "temperature_celsius": 22, "condition": "Sunny",
		},
		"price": map[string]any{
			"asset": "BTC", "price_usd": 45000,
		},
		"exchange_rate": map[string]any{
			"from": "EUR", "to": "USD", "rate": 1.08,
		},
		"status": map[string]any{
			"service": "api", "status": "operational", "latency_ms": 45,
		},
	}
	return &FlakyAPITool{
		failureRate: failureRate,
		rng:         rand.New(rand.NewSource(time.Now().UnixNano())),
		endpoints:   endpoints,
	}
}

func (f *FlakyAPITool) Name() string { return "flaky_api" }
func (f *FlakyAPITool) Description() string {
	return `Calls an external API (may fail ~20% of the time). Available endpoints: weather, price, exchange_rate, status. JSON input: {"endpoint": "weather", "params": {}}`
}

// Execute implements the API call with simulated failures.
//
// TODO(3): Implement this function. Steps:
//
//	a) Parse the input: {"endpoint": "<name>", "params": {...}}
//	   Use json.Unmarshal with a struct or map[string]json.RawMessage.
//
//	b) Simulate failure: if f.rng.Float64() < f.failureRate, return a realistic
//	   error such as: fmt.Errorf("connection timeout: endpoint %q not responding", endpoint)
//	   or "503 Service Unavailable".
//	   This error must be returned BEFORE trying to look up the endpoint.
//
//	c) Look up f.endpoints[endpoint]. If it does not exist, return a descriptive error.
//	   ("unknown endpoint: %q — available: weather, price, exchange_rate, status")
//
//	d) Serialize the response to JSON with json.Marshal and return it as a string.
//
// Go pattern: the error is returned; THE CALLER decides what to do with it.
// Design insight: how should the AGENT behave when this tool fails?
// That is the responsibility of the ReAct loop in agent.go, not of the tool.
//
// Go pattern: do NOT use panic() — an API failure is a normal business error,
// not an inconsistent program state.
func (f *FlakyAPITool) Execute(input string) (string, error) {
	var request struct {
		Endpoint string         `json:"endpoint"`
		Params   map[string]any `json:"params"`
	}
	if err := json.Unmarshal([]byte(input), &request); err != nil {
		return "", err
	}

	if f.rng.Float64() < f.failureRate {
		return "", fmt.Errorf("connection timeout: endpoint %q not responding", request.Endpoint)
	}

	data, ok := f.endpoints[request.Endpoint]
	if !ok {
		return "", fmt.Errorf("unknown endpoint: %q — available: weather, price, exchange_rate, status", request.Endpoint)
	}

	result, err := json.Marshal(data)
	if err != nil {
		return "", fmt.Errorf("error serializing the outout")
	}
	return string(result), nil
}
