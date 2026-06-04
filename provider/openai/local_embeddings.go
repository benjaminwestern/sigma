// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package openai

import (
	"fmt"
	"net/url"
	"strings"

	"github.com/wintermi/sigma"
)

const (
	defaultLocalEmbeddingProvider = sigma.ProviderID("ollama")
	defaultLocalEmbeddingModel    = sigma.ModelID("nomic-embed-text")
	defaultLocalEmbeddingBaseURL  = "http://127.0.0.1:11434/v1"
)

// LocalEmbeddingConfig configures RegisterLocalEmbeddings for a local or
// private OpenAI-compatible embeddings endpoint.
type LocalEmbeddingConfig struct {
	Provider            sigma.ProviderID
	ID                  sigma.ModelID
	BaseURL             string
	APIKeyEnv           string
	Headers             map[string]string
	Name                string
	DefaultDimensions   int
	MinDimensions       int
	MaxDimensions       int
	MaxInputTokens      int
	MaxBatchInputs      int
	MaxBatchBytes       int
	InputCostPerMillion float64
	CostCurrency        string
	ProviderMetadata    map[string]any
}

// RegisterLocalEmbeddings registers an OpenAI-compatible embeddings provider
// and model for local or private /v1/embeddings endpoints.
func RegisterLocalEmbeddings(registry *sigma.Registry, config LocalEmbeddingConfig) (sigma.EmbeddingModel, error) {
	if registry == nil {
		return sigma.EmbeddingModel{}, &sigma.Error{Code: sigma.ErrorUnsupported, Message: "registry is required"}
	}

	providerID := config.Provider
	if providerID == "" {
		providerID = defaultLocalEmbeddingProvider
	}
	modelID := config.ID
	if modelID == "" {
		modelID = defaultLocalEmbeddingModel
	}

	metadata := copyLocalEmbeddingMetadata(config.ProviderMetadata)
	if apiKeyEnv := strings.TrimSpace(config.APIKeyEnv); apiKeyEnv != "" {
		metadata[sigma.MetadataAPIKeyEnvVars] = []string{apiKeyEnv}
	}

	model := sigma.OpenAICompatibleEmbeddingModel(sigma.OpenAICompatibleEmbeddingModelConfig{
		ID:                  modelID,
		Provider:            providerID,
		BaseURL:             normalizeLocalEmbeddingBaseURL(config.BaseURL),
		Name:                config.Name,
		Headers:             config.Headers,
		DefaultDimensions:   config.DefaultDimensions,
		MinDimensions:       config.MinDimensions,
		MaxDimensions:       config.MaxDimensions,
		MaxInputTokens:      config.MaxInputTokens,
		MaxBatchInputs:      config.MaxBatchInputs,
		MaxBatchBytes:       config.MaxBatchBytes,
		InputCostPerMillion: config.InputCostPerMillion,
		CostCurrency:        config.CostCurrency,
		ProviderMetadata:    metadata,
	})
	if err := RegisterEmbeddings(registry, providerID); err != nil {
		return sigma.EmbeddingModel{}, err
	}
	if err := registry.RegisterEmbeddingModel(model); err != nil {
		return sigma.EmbeddingModel{}, fmt.Errorf("openai local embeddings: register model: %w", err)
	}
	return model, nil
}

func normalizeLocalEmbeddingBaseURL(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		raw = defaultLocalEmbeddingBaseURL
	}
	if !strings.Contains(raw, "://") {
		raw = "http://" + raw
	}
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return raw
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/")
	if !strings.HasSuffix(parsed.Path, "/v1") {
		parsed.Path += "/v1"
	}
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return parsed.String()
}

func copyLocalEmbeddingMetadata(metadata map[string]any) map[string]any {
	out := make(map[string]any, len(metadata)+1)
	for key, value := range metadata {
		out[key] = value
	}
	return out
}
