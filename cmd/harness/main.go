// Main program of the evaluation harness.
// This file is FULLY IMPLEMENTED — it is the boilerplate that wires everything together.
// Study how Go structures CLI programs: flags, initialization, orchestration.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/fedeparg/local-agentic-eval/internal/agent"
	"github.com/fedeparg/local-agentic-eval/internal/eval"
	"github.com/fedeparg/local-agentic-eval/internal/llm"
)

func main() {
	// CLI flags — standard Go pattern: declare, parse, use.
	// Flags have sensible defaults so it works with just ./harness
	var (
		ollamaURL   = flag.String("ollama", "http://localhost:11434", "Ollama base URL")
		model       = flag.String("model", "qwen2.5:3b", "Model name to use")
		datasetPath = flag.String("dataset", "data/golden_dataset.json", "Path to the golden dataset")
		kbPath      = flag.String("kb", "data/knowledge_base.json", "Path to the knowledge base")
		outputPath  = flag.String("output", "report.md", "Path of the output report")
		concurrency = flag.Int("concurrency", 3, "Maximum concurrent goroutines")
		caseID      = flag.String("case", "", "Run only this case ID (empty = all)")
	)
	flag.Parse()

	// 1. Build the LLM client (shared between agent and judge)
	llmClient := llm.NewClient(*ollamaURL, *model)

	// 2. Verify Ollama is available before continuing
	if err := llmClient.Ping(); err != nil {
		log.Fatalf("❌ Cannot connect to Ollama at %s: %v\n"+
			"   Make sure Ollama is running: ollama serve\n"+
			"   And that the model is downloaded: ollama pull %s",
			*ollamaURL, err, *model)
	}
	fmt.Printf("✓ Connected to Ollama (%s), model: %s\n", *ollamaURL, *model)

	// 3. Load the knowledge base and build the tools
	lookupTool, err := agent.NewLookupTool(*kbPath)
	if err != nil {
		log.Fatalf("❌ Error loading knowledge base: %v", err)
	}

	registry := agent.NewToolRegistry(
		&agent.CalculatorTool{},
		lookupTool,
		agent.NewFlakyAPITool(0.20), // 20% simulated failures
	)

	// 4. Build the agent
	ag := agent.New(llmClient, registry)

	// 5. Load the golden dataset
	cases, err := loadCases(*datasetPath)
	if err != nil {
		log.Fatalf("❌ Error loading dataset: %v", err)
	}

	// 6. Optional filter: run a single case for quick debugging
	if *caseID != "" {
		cases = filterByID(cases, *caseID)
		if len(cases) == 0 {
			log.Fatalf("❌ Case %q not found in the dataset", *caseID)
		}
	}

	fmt.Printf("→ Running %d evaluation case(s)...\n\n", len(cases))

	// 7. Run the concurrent harness
	config := eval.RunnerConfig{
		MaxConcurrency: *concurrency,
		JudgeThreshold: 3.0,
	}
	results, err := eval.RunAll(cases, ag, llmClient, config)
	if err != nil {
		log.Fatalf("❌ Error in the runner: %v", err)
	}

	// 8. Generate the Markdown report
	report := eval.GenerateReport(results)

	// 9. Write the report to disk AND print it to stdout
	if err := os.WriteFile(*outputPath, []byte(report), 0o644); err != nil {
		log.Fatalf("❌ Error writing report: %v", err)
	}
	fmt.Println(report)
	fmt.Printf("📄 Report saved to: %s\n", *outputPath)

	// 10. Exit code: 1 if any case failed (useful for CI)
	for _, r := range results {
		if !r.Passed {
			os.Exit(1)
		}
	}
}

// loadCases reads and parses the golden dataset from JSON.
// Returns (nil, error) on any problem — no panic.
func loadCases(path string) ([]eval.Case, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %q: %w", path, err)
	}
	var dataset struct {
		Cases []eval.Case `json:"cases"`
	}
	if err := json.Unmarshal(data, &dataset); err != nil {
		return nil, fmt.Errorf("parse JSON of %q: %w", path, err)
	}
	if len(dataset.Cases) == 0 {
		return nil, fmt.Errorf("the dataset is empty")
	}
	return dataset.Cases, nil
}

// filterByID returns the case with that ID, or nil if it does not exist.
func filterByID(cases []eval.Case, id string) []eval.Case {
	for _, c := range cases {
		if c.ID == id {
			return []eval.Case{c}
		}
	}
	return nil
}
