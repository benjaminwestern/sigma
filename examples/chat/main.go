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

	"github.com/wintermi/sigma"
	"github.com/wintermi/sigma/provider/openai"
	"github.com/wintermi/sigma/sigmatest"
)

func main() {
	ctx := context.Background()
	client, model, opts, err := clientAndModel()
	if err != nil {
		log.Fatal(err)
	}

	text, err := client.CompleteText(ctx, model, "Write one short sentence about Sigma.", opts...)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(text)
}

func clientAndModel() (*sigma.Client, sigma.Model, []sigma.Option, error) {
	if os.Getenv("SIGMA_EXAMPLE_PROVIDER") == "openai" {
		apiKey := os.Getenv("OPENAI_API_KEY")
		if apiKey == "" {
			return nil, sigma.Model{}, nil, fmt.Errorf("set OPENAI_API_KEY or unset SIGMA_EXAMPLE_PROVIDER")
		}

		modelID := sigma.ModelID(envOr("SIGMA_EXAMPLE_MODEL", "gpt-4o-mini"))
		registry := sigma.NewRegistry()
		if err := openai.Register(registry, sigma.ProviderOpenAI); err != nil {
			return nil, sigma.Model{}, nil, err
		}
		model := sigma.Model{
			ID:              modelID,
			Provider:        sigma.ProviderOpenAI,
			API:             sigma.APIOpenAICompletions,
			Name:            string(modelID),
			SupportedInputs: []sigma.ContentBlockType{sigma.ContentBlockText},
			SupportsTools:   true,
		}
		if err := registry.RegisterModel(model); err != nil {
			return nil, sigma.Model{}, nil, err
		}
		return sigma.NewClient(sigma.WithRegistry(registry)), model, []sigma.Option{sigma.WithAPIKey(apiKey)}, nil
	}

	provider := sigmatest.NewFauxProvider(sigmatest.Script{
		Final: sigma.AssistantMessage{
			Content: []sigma.ContentBlock{sigma.Text("Sigma provides provider-neutral model calls for Go.")},
		},
	})
	registry, err := sigmatest.Registry(provider)
	if err != nil {
		return nil, sigma.Model{}, nil, err
	}
	return sigma.NewClient(sigma.WithRegistry(registry)), sigmatest.TextModel(), nil, nil
}

func envOr(name string, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}
