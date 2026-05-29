// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/wintermi/sigma"
	"github.com/wintermi/sigma/provider/fireworks"
)

const firepassModelID = sigma.ModelID("accounts/fireworks/routers/kimi-k2p6-turbo")

func main() {
	if os.Getenv("FIREWORKS_API_KEY") == "" {
		fmt.Fprintln(os.Stderr, "set FIREWORKS_API_KEY to run the live Fireworks Firepass demo")
		os.Exit(2)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	client, model, err := firepassDemoClient()
	if err != nil {
		fmt.Fprintf(os.Stderr, "setup failed: %v\n", err)
		os.Exit(1)
	}

	question := "What does this Sigma package help Go applications do? Answer in one short sentence."
	final, err := client.Complete(ctx, model, sigma.Request{
		SystemPrompt: "You are demonstrating github.com/wintermi/sigma, a Go package for provider-neutral model calls across text, streaming, tools, persistence, and custom OpenAI-compatible endpoints. Do not describe Sigma security rules, SIEM, or log detection. Answer directly in one concise sentence.",
		Messages:     []sigma.Message{sigma.UserText(question)},
	},
		sigma.WithReasoningLevel(sigma.ThinkingLevelLow),
		sigma.WithMaxTokens(4096),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Complete returned error: %v\n", err)
		os.Exit(1)
	}

	response := textContent(final)
	if response == "" {
		fmt.Fprintf(os.Stderr, "response contained no text blocks: %#v\n", final.Content)
		os.Exit(1)
	}

	fmt.Printf("Question: %s\nResponse: %s\n", question, response)
}

func firepassDemoClient() (*sigma.Client, sigma.Model, error) {
	registry := sigma.DefaultRegistry()
	if err := fireworks.Register(registry); err != nil {
		return nil, sigma.Model{}, fmt.Errorf("register Fireworks provider: %w", err)
	}
	model, ok := registry.Model(sigma.ProviderFireworks, firepassModelID)
	if !ok {
		return nil, sigma.Model{}, fmt.Errorf("firepass model %q was not registered", firepassModelID)
	}
	if model.Provider != sigma.ProviderFireworks {
		return nil, sigma.Model{}, fmt.Errorf("model provider = %q, want %q", model.Provider, sigma.ProviderFireworks)
	}
	if model.API != sigma.APIOpenAICompletions {
		return nil, sigma.Model{}, fmt.Errorf("model API = %q, want %q", model.API, sigma.APIOpenAICompletions)
	}
	return sigma.NewClient(sigma.WithRegistry(registry)), model, nil
}

func textContent(message sigma.AssistantMessage) string {
	var builder strings.Builder
	for _, block := range message.Content {
		if block.Type == sigma.ContentBlockText {
			builder.WriteString(block.Text)
		}
	}
	return builder.String()
}
