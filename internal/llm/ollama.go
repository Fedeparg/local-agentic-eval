// Package llm provides a thin HTTP client for the Ollama local inference API.
// This is BOILERPLATE — no TODOs here. Study how HTTP clients are structured
// in Go: a Client type, NewXxx constructors, methods returning (T, error).
package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client wraps the Ollama HTTP API.
// We export the public fields so tests can inspect them.
type Client struct {
	BaseURL    string
	Model      string
	httpClient *http.Client // private: the caller does not need to configure it
}

// GenerateRequest matches the body of Ollama's /api/generate.
type GenerateRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

// GenerateResponse matches the /api/generate response when stream=false.
type GenerateResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

// NewClient creates a Client with a 120s timeout.
// Generous timeout because small models on CPU can be slow.
func NewClient(baseURL, model string) *Client {
	return &Client{
		BaseURL: baseURL,
		Model:   model,
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
	}
}

// Ping verifies that the Ollama server is reachable.
// Returns nil if everything is fine, a descriptive error otherwise.
// Go pattern: verification functions return only `error`, not bool.
func (c *Client) Ping() error {
	resp, err := c.httpClient.Get(c.BaseURL + "/api/tags")
	if err != nil {
		return fmt.Errorf("cannot reach Ollama at %s: %w", c.BaseURL, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ollama returned HTTP %d", resp.StatusCode)
	}
	return nil
}

// Generate sends a prompt to the LLM and returns the response as plain text.
//
// Go pattern: returns (string, error), NEVER panics on network or format errors.
// The caller decides how to handle the error.
func (c *Client) Generate(prompt string) (string, error) {
	reqBody := GenerateRequest{
		Model:  c.Model,
		Prompt: prompt,
		Stream: false, // stream=false → single response, easier to parse
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		// json.Marshal only fails on non-serializable types (channels, functions).
		// Our struct is safe, but we propagate the error anyway.
		return "", fmt.Errorf("marshal request: %w", err)
	}

	resp, err := c.httpClient.Post(
		c.BaseURL+"/api/generate",
		"application/json",
		bytes.NewReader(body),
	)
	if err != nil {
		return "", fmt.Errorf("http post to ollama: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body) // ignore read error of the error body
		return "", fmt.Errorf("ollama HTTP %d: %s", resp.StatusCode, raw)
	}

	var result GenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("decode response: %w", err)
	}

	return result.Response, nil
}
