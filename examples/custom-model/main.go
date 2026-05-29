// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/wintermi/sigma"
	"github.com/wintermi/sigma/provider/openai"
)

func main() {
	preset := localPreset(envOr("SIGMA_LOCAL_PRESET", "ollama"))
	baseURL := envOr("SIGMA_LOCAL_BASE_URL", preset.baseURL)
	modelID := sigma.ModelID(envOr("SIGMA_LOCAL_MODEL", preset.modelID))

	registry := sigma.NewRegistry()
	if err := openai.Register(registry, preset.provider); err != nil {
		log.Fatal(err)
	}

	model := sigma.OpenAICompatibleModel(sigma.OpenAICompatibleModelConfig{
		ID:              modelID,
		Provider:        preset.provider,
		BaseURL:         baseURL,
		Name:            preset.name + " " + string(modelID),
		ContextWindow:   preset.contextWindow,
		MaxOutputTokens: preset.maxOutputTokens,
		SupportedInputs: []sigma.ContentBlockType{sigma.ContentBlockText},
		SupportsTools:   true,
		OpenAICompletionsCompat: &sigma.OpenAICompletionsCompat{
			SupportsStore:                    sigma.OpenAICompatUnsupported,
			SupportsDeveloperRole:            sigma.OpenAICompatUnsupported,
			SupportsStreamingUsage:           sigma.OpenAICompatUnsupported,
			SupportsStrictTools:              sigma.OpenAICompatUnsupported,
			ReasoningFormat:                  sigma.OpenAICompletionsReasoningUnsupported,
			MaxTokensField:                   sigma.OpenAICompletionsMaxTokens,
			CacheControlFormat:               sigma.OpenAICompletionsCacheControlUnsupported,
			RequiresToolResultName:           sigma.OpenAICompatSupported,
			RequiresAssistantAfterToolResult: sigma.OpenAICompatUnsupported,
		},
	})
	if err := registry.RegisterModel(model); err != nil {
		log.Fatal(err)
	}

	client := sigma.NewClient(sigma.WithRegistry(registry))
	if os.Getenv("SIGMA_LOCAL_BASE_URL") == "" {
		fmt.Printf("registered local OpenAI-compatible model %s/%s\n", model.Provider, model.ID)
		fmt.Println("set SIGMA_LOCAL_BASE_URL to run it against Ollama, vLLM, LM Studio, or another OpenAI-compatible server")
		fmt.Println("optional presets: SIGMA_LOCAL_PRESET=ollama|vllm|lmstudio|generic")
		return
	}

	text, err := client.CompleteText(
		context.Background(),
		model,
		"Reply with one short sentence.",
		sigma.WithAPIKey(envOr("SIGMA_LOCAL_API_KEY", "local")),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(text)
}

type endpointPreset struct {
	provider        sigma.ProviderID
	name            string
	baseURL         string
	modelID         string
	contextWindow   int
	maxOutputTokens int
}

func localPreset(name string) endpointPreset {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "vllm":
		return endpointPreset{
			provider:        "vllm",
			name:            "vLLM",
			baseURL:         "http://localhost:8000/v1",
			modelID:         "meta-llama/Llama-3.1-8B-Instruct",
			contextWindow:   131072,
			maxOutputTokens: 8192,
		}
	case "lmstudio", "lm-studio":
		return endpointPreset{
			provider:        "lm-studio",
			name:            "LM Studio",
			baseURL:         "http://localhost:1234/v1",
			modelID:         "local-model",
			contextWindow:   32768,
			maxOutputTokens: 4096,
		}
	case "generic":
		return endpointPreset{
			provider:        sigma.ProviderCustom,
			name:            "OpenAI-compatible",
			baseURL:         "http://localhost:8080/v1",
			modelID:         "local-model",
			contextWindow:   32768,
			maxOutputTokens: 4096,
		}
	default:
		return endpointPreset{
			provider:        "ollama",
			name:            "Ollama",
			baseURL:         "http://localhost:11434/v1",
			modelID:         "llama3.2",
			contextWindow:   131072,
			maxOutputTokens: 8192,
		}
	}
}

func envOr(name string, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}
