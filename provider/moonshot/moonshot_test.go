// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package moonshot_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/wintermi/sigma"
	"github.com/wintermi/sigma/provider/moonshot"
)

type capturedRequest struct {
	Method  string
	Path    string
	Headers http.Header
}

func TestRegistersReportOpenAICompletionsAPI(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		provider sigma.ProviderID
		register func(*sigma.Registry) error
	}{
		{name: "moonshot", provider: sigma.ProviderMoonshotAI, register: func(registry *sigma.Registry) error {
			return moonshot.Register(registry)
		}},
		{name: "moonshot cn", provider: sigma.ProviderMoonshotAICN, register: func(registry *sigma.Registry) error {
			return moonshot.RegisterCN(registry)
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			registry := sigma.NewRegistry()
			if err := tt.register(registry); err != nil {
				t.Fatalf("register returned error: %v", err)
			}
			if err := registry.RegisterModel(moonshotTestModel(tt.provider)); err != nil {
				t.Fatalf("RegisterModel returned error: %v", err)
			}

			providers := registry.ListProviders()
			if got, want := providers[0].ID, tt.provider; got != want {
				t.Fatalf("provider ID = %q, want %q", got, want)
			}
			if got, want := providers[0].TextAPI, sigma.APIOpenAICompletions; got != want {
				t.Fatalf("provider API = %q, want %q", got, want)
			}
		})
	}
}

func TestCompleteUsesConfiguredOpenAICompatibleBaseURL(t *testing.T) {
	t.Parallel()

	requests := make(chan capturedRequest, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captureRequest(t, requests, r)
		writeCompleted(t, w)
	}))
	t.Cleanup(server.Close)

	model := moonshotTestModel(sigma.ProviderMoonshotAICN)
	registry := sigma.NewRegistry()
	if err := moonshot.RegisterCN(registry, moonshot.WithBaseURL(server.URL+"/v1")); err != nil {
		t.Fatalf("RegisterCN returned error: %v", err)
	}
	if err := registry.RegisterModel(model); err != nil {
		t.Fatalf("RegisterModel returned error: %v", err)
	}
	client := sigma.NewClient(sigma.WithRegistry(registry))

	final, err := client.Complete(
		context.Background(),
		model,
		sigma.Request{Messages: []sigma.Message{sigma.UserText("hi")}},
		sigma.WithAPIKey("request-key"),
	)
	if err != nil {
		t.Fatalf("Complete returned error: %v", err)
	}
	if got, want := final.Content[0].Text, "ok"; got != want {
		t.Fatalf("final text = %q, want %q", got, want)
	}

	request := receiveRequest(t, requests)
	if got, want := request.Method, http.MethodPost; got != want {
		t.Fatalf("method = %q, want %q", got, want)
	}
	if got, want := request.Path, "/v1/chat/completions"; got != want {
		t.Fatalf("path = %q, want %q", got, want)
	}
	if got, want := request.Headers.Get("Authorization"), "Bearer request-key"; got != want {
		t.Fatalf("Authorization header = %q, want %q", got, want)
	}
}

func TestRegistersCatalogK27Models(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		provider sigma.ProviderID
		models   []sigma.ModelID
	}{
		{name: "moonshot", provider: sigma.ProviderMoonshotAI, models: []sigma.ModelID{"kimi-k2.7-code", "kimi-k2.7-code-highspeed"}},
		{name: "moonshot cn", provider: sigma.ProviderMoonshotAICN, models: []sigma.ModelID{"kimi-k2.7-code", "kimi-k2.7-code-highspeed"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			for _, modelID := range tt.models {
				model, ok := sigma.DefaultRegistry().Model(tt.provider, modelID)
				if !ok {
					t.Fatalf("default registry missing %s model %q", tt.provider, modelID)
				}
				registry := sigma.NewRegistry()
				switch tt.provider {
				case sigma.ProviderMoonshotAI:
					if err := moonshot.Register(registry); err != nil {
						t.Fatalf("Register returned error: %v", err)
					}
				case sigma.ProviderMoonshotAICN:
					if err := moonshot.RegisterCN(registry); err != nil {
						t.Fatalf("RegisterCN returned error: %v", err)
					}
				}
				if err := registry.RegisterModel(model); err != nil {
					t.Fatalf("RegisterModel(%s, %s) returned error: %v", tt.provider, modelID, err)
				}
			}
		})
	}
}

func moonshotTestModel(provider sigma.ProviderID) sigma.Model {
	return sigma.Model{
		ID:               "kimi-test",
		Provider:         provider,
		API:              sigma.APIOpenAICompletions,
		Name:             "Kimi Test",
		SupportedInputs:  []sigma.ContentBlockType{sigma.ContentBlockText},
		SupportsTools:    true,
		SupportsThinking: true,
	}
}

func captureRequest(t *testing.T, requests chan<- capturedRequest, r *http.Request) {
	t.Helper()

	requests <- capturedRequest{
		Method:  r.Method,
		Path:    r.URL.Path,
		Headers: r.Header.Clone(),
	}
}

func receiveRequest(t *testing.T, requests <-chan capturedRequest) capturedRequest {
	t.Helper()

	select {
	case request := <-requests:
		return request
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for request")
		return capturedRequest{}
	}
}

func writeCompleted(t *testing.T, w http.ResponseWriter) {
	t.Helper()

	w.Header().Set("Content-Type", "text/event-stream")
	_, _ = io.WriteString(w, `data: {"id":"chatcmpl_test","model":"kimi-test","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}`+"\n\n")
	_, _ = io.WriteString(w, `data: {"id":"chatcmpl_test","model":"kimi-test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`+"\n\n")
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
}
