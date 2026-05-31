// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/wintermi/sigma"
	"github.com/wintermi/sigma/internal/goldentest"
	"github.com/wintermi/sigma/provider/openai"
)

func TestRegisterCodexResponsesReportsCodexResponsesAPI(t *testing.T) {
	t.Parallel()

	registry := sigma.NewRegistry()
	providerID := sigma.ProviderID("openai-codex-responses")
	if err := openai.RegisterCodexResponses(registry, providerID); err != nil {
		t.Fatalf("RegisterCodexResponses returned error: %v", err)
	}
	if err := registry.RegisterModel(codexResponsesTestModel(providerID)); err != nil {
		t.Fatalf("RegisterModel returned error: %v", err)
	}

	providers := registry.ListProviders()
	if got, want := providers[0].TextAPI, sigma.APIOpenAICodexResponses; got != want {
		t.Fatalf("provider API = %q, want %q", got, want)
	}
}

func TestCodexResponsesInjectsBearerTokenAndUsesCodexModelName(t *testing.T) {
	t.Parallel()

	requests := make(chan capturedRequest, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captureRequest(t, requests, r)
		writeResponsesSSE(t, w, responsesCompletedEvent)
	}))
	t.Cleanup(server.Close)

	providerID := sigma.ProviderID("codex-responses-auth-test")
	model := codexResponsesTestModel(providerID)
	model.OpenAICodexResponses.Model = "codex-mini-latest"
	client := codexResponsesTestClient(t, providerID, model, server.URL, codexTokenProvider("codex-oauth-token"))

	final, err := client.Complete(
		context.Background(),
		model,
		sigma.Request{Messages: []sigma.Message{sigma.UserText("hi")}},
		sigma.WithOpenAIOptions(sigma.OpenAIOptions{
			TextVerbosity:        "low",
			PromptCacheRetention: "24h",
		}),
	)
	if err != nil {
		t.Fatalf("Complete returned error: %v", err)
	}
	if got, want := final.ProviderMetadata["id"], "resp_complete"; got != want {
		t.Fatalf("response id = %v, want %v", got, want)
	}

	request := receiveRequest(t, requests)
	if got, want := request.Path, "/responses"; got != want {
		t.Fatalf("path = %q, want %q", got, want)
	}
	assertHeader(t, request.Headers, "Authorization", "Bearer codex-oauth-token")
	assertHeader(t, request.Headers, "chatgpt-account-id", "acct_codex")
	assertHeader(t, request.Headers, "OpenAI-Beta", "responses=experimental")
	assertHeader(t, request.Headers, "originator", "sigma")
	assertHeader(t, request.Headers, "X-Client", "client")
	goldentest.AssertJSON(t, request.Body, "provider/openai/codex_responses/basic_payload.json")
}

func TestCodexResponsesPreservesSystemPromptInstructionsAndForcesStoreFalse(t *testing.T) {
	t.Parallel()

	requests := make(chan capturedRequest, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captureRequest(t, requests, r)
		writeResponsesSSE(t, w, responsesCompletedEvent)
	}))
	t.Cleanup(server.Close)

	providerID := sigma.ProviderID("codex-responses-instructions-test")
	model := codexResponsesTestModel(providerID)
	client := codexResponsesTestClient(t, providerID, model, server.URL, codexTokenProvider("codex-oauth-token"))

	_, err := client.Complete(
		context.Background(),
		model,
		sigma.Request{
			SystemPrompt: "Use concise replies.",
			Messages:     []sigma.Message{sigma.UserText("hi")},
		},
		sigma.WithProviderOption(providerID, "store", true),
	)
	if err != nil {
		t.Fatalf("Complete returned error: %v", err)
	}

	request := receiveRequest(t, requests)
	var payload map[string]any
	if err := json.Unmarshal(request.Body, &payload); err != nil {
		t.Fatalf("Unmarshal request body returned error: %v", err)
	}
	if got, want := payload["instructions"], "Use concise replies."; got != want {
		t.Fatalf("instructions = %v, want %q", got, want)
	}
	if got, want := payload["store"], false; got != want {
		t.Fatalf("store = %v, want %v", got, want)
	}
}

func TestCodexResponsesOmitsUnsupportedMaxOutputTokens(t *testing.T) {
	t.Parallel()

	requests := make(chan capturedRequest, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captureRequest(t, requests, r)
		writeResponsesSSE(t, w, responsesCompletedEvent)
	}))
	t.Cleanup(server.Close)

	providerID := sigma.ProviderID("codex-responses-max-tokens-test")
	model := codexResponsesTestModel(providerID)
	client := codexResponsesTestClient(t, providerID, model, server.URL, codexTokenProvider("codex-oauth-token"))

	_, err := client.Complete(
		context.Background(),
		model,
		sigma.Request{Messages: []sigma.Message{sigma.UserText("hi")}},
		sigma.WithMaxTokens(128),
	)
	if err != nil {
		t.Fatalf("Complete returned error: %v", err)
	}

	request := receiveRequest(t, requests)
	var payload map[string]any
	if err := json.Unmarshal(request.Body, &payload); err != nil {
		t.Fatalf("Unmarshal request body returned error: %v", err)
	}
	if _, ok := payload["max_output_tokens"]; ok {
		t.Fatalf("max_output_tokens was sent in Codex payload: %#v", payload)
	}
}

func TestCodexResponsesDerivesPromptCacheKey(t *testing.T) {
	t.Parallel()

	requests := make(chan capturedRequest, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captureRequest(t, requests, r)
		writeResponsesSSE(t, w, responsesCompletedEvent)
	}))
	t.Cleanup(server.Close)

	providerID := sigma.ProviderID("codex-responses-cache-test")
	model := codexResponsesTestModel(providerID)
	client := codexResponsesTestClient(t, providerID, model, server.URL, codexTokenProvider("codex-oauth-token"))
	sessionID := strings.Repeat("x", 70)

	_, err := client.Complete(
		context.Background(),
		model,
		sigma.Request{Messages: []sigma.Message{sigma.UserText("hi")}},
		sigma.WithSessionID(sessionID),
		sigma.WithCacheRetention(sigma.CacheRetentionPersistent),
	)
	if err != nil {
		t.Fatalf("Complete returned error: %v", err)
	}

	request := receiveRequest(t, requests)
	var payload map[string]any
	if err := json.Unmarshal(request.Body, &payload); err != nil {
		t.Fatalf("Unmarshal request body returned error: %v", err)
	}
	if got, want := payload["prompt_cache_key"], strings.Repeat("x", 64); got != want {
		t.Fatalf("prompt_cache_key = %v, want %q", got, want)
	}
	if got, want := payload["prompt_cache_retention"], "24h"; got != want {
		t.Fatalf("prompt_cache_retention = %v, want %q", got, want)
	}
	if _, ok := payload["previous_response_id"]; ok {
		t.Fatalf("previous_response_id was sent in Codex payload: %#v", payload)
	}
	assertHeader(t, request.Headers, "session-id", sessionID)
	assertHeader(t, request.Headers, "x-client-request-id", sessionID)
}

func TestCodexResponsesMissingOAuthProviderFailsBeforeNetwork(t *testing.T) {
	t.Parallel()

	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		calls++
	}))
	t.Cleanup(server.Close)

	providerID := sigma.ProviderID("codex-responses-missing-oauth-test")
	model := codexResponsesTestModel(providerID)
	client := codexResponsesTestClient(t, providerID, model, server.URL, nil)

	_, err := client.Complete(context.Background(), model, sigma.Request{Messages: []sigma.Message{sigma.UserText("hi")}})
	if !errors.Is(err, sigma.ErrCredentialUnavailable) {
		t.Fatalf("error = %v, want ErrCredentialUnavailable", err)
	}
	if calls != 0 {
		t.Fatalf("server calls = %d, want 0", calls)
	}
}

func TestCodexResponsesMissingAccountIDFailsBeforeNetwork(t *testing.T) {
	t.Parallel()

	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		calls++
	}))
	t.Cleanup(server.Close)

	providerID := sigma.ProviderID("codex-responses-missing-account-test")
	model := codexResponsesTestModel(providerID)
	client := codexResponsesTestClient(t, providerID, model, server.URL, sigma.OAuthTokenProviderFunc(
		func(context.Context, sigma.Model, sigma.Options) (sigma.Credential, error) {
			return sigma.Credential{
				Type:  sigma.CredentialTypeOAuthToken,
				Value: "not-a-jwt",
			}, nil
		},
	))

	_, err := client.Complete(context.Background(), model, sigma.Request{Messages: []sigma.Message{sigma.UserText("hi")}})
	if err == nil {
		t.Fatal("Complete returned nil error")
	}
	if !strings.Contains(err.Error(), "account id") {
		t.Fatalf("error = %q, want account id context", err.Error())
	}
	if calls != 0 {
		t.Fatalf("server calls = %d, want 0", calls)
	}
}

func TestCodexResponsesUnsupportedTransportFailsBeforeNetwork(t *testing.T) {
	t.Parallel()

	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		calls++
	}))
	t.Cleanup(server.Close)

	providerID := sigma.ProviderID("codex-responses-transport-test")
	model := codexResponsesTestModel(providerID)
	client := codexResponsesTestClient(t, providerID, model, server.URL, codexTokenProvider("codex-oauth-token"))

	_, err := client.Complete(
		context.Background(),
		model,
		sigma.Request{Messages: []sigma.Message{sigma.UserText("hi")}},
		sigma.WithTransport(sigma.TransportWebSocket),
	)
	if !errors.Is(err, sigma.ErrInvalidOptions) {
		t.Fatalf("error = %v, want ErrInvalidOptions", err)
	}
	if !strings.Contains(err.Error(), "websocket") {
		t.Fatalf("error = %q, want websocket context", err.Error())
	}
	if calls != 0 {
		t.Fatalf("server calls = %d, want 0", calls)
	}
}

func TestCodexResponsesTextStreamingMapsMetadataAndUsage(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeResponsesSSE(t, w,
			`data: {"type":"response.output_item.added","response_id":"codex_resp","model":"codex-provider-model","output_index":0,"item":{"type":"message","id":"msg_codex","role":"assistant","content":[]}}

data: {"type":"response.output_text.delta","response_id":"codex_resp","model":"codex-provider-model","item_id":"msg_codex","output_index":0,"content_index":0,"delta":"Codex"}

data: {"type":"response.output_text.delta","response_id":"codex_resp","model":"codex-provider-model","item_id":"msg_codex","output_index":0,"content_index":0,"delta":" ready"}

data: {"type":"response.completed","response":{"status":"completed","output":[{"type":"message","id":"msg_codex","role":"assistant","content":[{"type":"output_text","id":"txt_codex","text":"Codex ready"}]}],"usage":{"input_tokens":4,"output_tokens":3,"total_tokens":7}}}
`,
		)
	}))
	t.Cleanup(server.Close)

	providerID := sigma.ProviderID("codex-responses-stream-test")
	model := codexResponsesTestModel(providerID)
	client := codexResponsesTestClient(t, providerID, model, server.URL, codexTokenProvider("codex-oauth-token"))

	stream := client.Stream(context.Background(), model, sigma.Request{Messages: []sigma.Message{sigma.UserText("hi")}})
	events := collectEvents(t, stream)
	if err := stream.Err(); err != nil {
		t.Fatalf("stream error = %v", err)
	}
	final, ok := stream.Final()
	if !ok {
		t.Fatal("stream final was not recorded")
	}

	if got, want := eventKinds(events), []sigma.EventKind{
		sigma.EventKindStart,
		sigma.EventKindTextStart,
		sigma.EventKindTextDelta,
		sigma.EventKindTextDelta,
		sigma.EventKindTextEnd,
		sigma.EventKindDone,
	}; !reflect.DeepEqual(got, want) {
		t.Fatalf("event kinds = %v, want %v", got, want)
	}
	if got, want := final.Content[0].Text, "Codex ready"; got != want {
		t.Fatalf("text = %q, want %q", got, want)
	}
	if got, want := final.ProviderMetadata["id"], "codex_resp"; got != want {
		t.Fatalf("response id = %v, want %v", got, want)
	}
	if got, want := final.ProviderMetadata["model"], "codex-provider-model"; got != want {
		t.Fatalf("provider model = %v, want %v", got, want)
	}
	if final.Usage == nil {
		t.Fatal("final usage was nil")
	}
	if got, want := final.Usage.TotalTokens, 7; got != want {
		t.Fatalf("total tokens = %d, want %d", got, want)
	}
}

func TestCodexResponsesToolCallStreaming(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeResponsesSSE(t, w,
			`data: {"type":"response.output_item.added","response_id":"codex_tool","output_index":0,"item":{"type":"function_call","id":"fc_codex","call_id":"call_codex","name":"shell","arguments":""}}

data: {"type":"response.function_call_arguments.delta","response_id":"codex_tool","item_id":"fc_codex","output_index":0,"delta":"{\"cmd\""}

data: {"type":"response.function_call_arguments.delta","response_id":"codex_tool","item_id":"fc_codex","output_index":0,"delta":":\"go test\"}"}

data: {"type":"response.output_item.done","response_id":"codex_tool","output_index":0,"item":{"type":"function_call","id":"fc_codex","call_id":"call_codex","name":"shell","arguments":"{\"cmd\":\"go test\"}"}}

data: {"type":"response.completed","response":{"status":"completed","output":[{"type":"function_call","id":"fc_codex","call_id":"call_codex","name":"shell","arguments":"{\"cmd\":\"go test\"}"}]}}
`,
		)
	}))
	t.Cleanup(server.Close)

	providerID := sigma.ProviderID("codex-responses-tool-test")
	model := codexResponsesTestModel(providerID)
	client := codexResponsesTestClient(t, providerID, model, server.URL, codexTokenProvider("codex-oauth-token"))

	stream := client.Stream(context.Background(), model, sigma.Request{Messages: []sigma.Message{sigma.UserText("run tests")}})
	events := collectEvents(t, stream)
	if err := stream.Err(); err != nil {
		t.Fatalf("stream error = %v", err)
	}
	final, ok := stream.Final()
	if !ok {
		t.Fatal("stream final was not recorded")
	}

	if got, want := eventKinds(events), []sigma.EventKind{
		sigma.EventKindStart,
		sigma.EventKindToolCallStart,
		sigma.EventKindToolCallDelta,
		sigma.EventKindToolCallDelta,
		sigma.EventKindToolCallDelta,
		sigma.EventKindToolCallEnd,
		sigma.EventKindDone,
	}; !reflect.DeepEqual(got, want) {
		t.Fatalf("event kinds = %v, want %v", got, want)
	}
	if got, want := final.Content[0].ToolCallID, "call_codex"; got != want {
		t.Fatalf("tool call id = %q, want %q", got, want)
	}
	if got, want := final.Content[0].ProviderMetadata["id"], "fc_codex"; got != want {
		t.Fatalf("tool item id = %v, want %v", got, want)
	}
	args := final.Content[0].ToolArguments.(map[string]any)
	if got, want := args["cmd"], "go test"; got != want {
		t.Fatalf("tool cmd = %v, want %v", got, want)
	}
}

func TestCodexResponsesProviderErrorUsesCodexAPIAndRedacts(t *testing.T) {
	t.Parallel()

	const token = "codex-oauth-secret"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("x-request-id", "req_codex")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"error":{"message":"bad token","access_token":"`+token+`"}}`)
	}))
	t.Cleanup(server.Close)

	providerID := sigma.ProviderID("codex-responses-error-test")
	model := codexResponsesTestModel(providerID)
	client := codexResponsesTestClient(t, providerID, model, server.URL, codexTokenProvider(token))

	final, err := client.Complete(context.Background(), model, sigma.Request{Messages: []sigma.Message{sigma.UserText("hi")}})
	if err == nil {
		t.Fatal("Complete returned nil error")
	}
	if !errors.Is(err, sigma.ErrProviderResponse) {
		t.Fatalf("error = %v, want ErrProviderResponse", err)
	}
	if got, want := final.Diagnostics[0].API, sigma.APIOpenAICodexResponses; got != want {
		t.Fatalf("diagnostic API = %q, want %q", got, want)
	}
	if errorsContains(err, token) || strings.Contains(final.Diagnostics[0].BodyPreview, token) {
		t.Fatalf("provider error leaked token: err=%v diagnostic=%+v", err, final.Diagnostics[0])
	}
}

func TestCodexResponsesTokenProviderErrorIsRedacted(t *testing.T) {
	t.Parallel()

	const token = "codex-oauth-secret"
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Fatal("server should not be called")
	}))
	t.Cleanup(server.Close)

	providerID := sigma.ProviderID("codex-responses-token-error-test")
	model := codexResponsesTestModel(providerID)
	client := codexResponsesTestClient(t, providerID, model, server.URL, sigma.OAuthTokenProviderFunc(
		func(context.Context, sigma.Model, sigma.Options) (sigma.Credential, error) {
			return sigma.Credential{}, fmt.Errorf("refresh failed for %s", token)
		},
	))

	_, err := client.Complete(context.Background(), model, sigma.Request{Messages: []sigma.Message{sigma.UserText("hi")}})
	if err == nil {
		t.Fatal("Complete returned nil error")
	}
	if errorsContains(err, token) {
		t.Fatalf("token provider error leaked token: %v", err)
	}
}

func codexResponsesTestClient(t *testing.T, providerID sigma.ProviderID, model sigma.Model, baseURL string, tokenProvider sigma.OAuthTokenProvider) *sigma.Client {
	t.Helper()

	registry := sigma.NewRegistry()
	if err := registry.RegisterTextProvider(providerID, openai.NewCodexResponsesProvider(openai.WithBaseURL(baseURL))); err != nil {
		t.Fatalf("RegisterTextProvider returned error: %v", err)
	}
	if err := registry.RegisterModel(model); err != nil {
		t.Fatalf("RegisterModel returned error: %v", err)
	}
	var defaults []sigma.Option
	if tokenProvider != nil {
		defaults = append(defaults, openai.WithCodexResponsesOAuthTokenProvider(providerID, tokenProvider))
	}
	return sigma.NewClient(
		sigma.WithRegistry(registry),
		sigma.WithDefaultOptions(defaults...),
		sigma.WithDefaultHeader("X-Client", "client"),
	)
}

func codexResponsesTestModel(providerID sigma.ProviderID) sigma.Model {
	return sigma.Model{
		ID:       "codex-test",
		Provider: providerID,
		API:      sigma.APIOpenAICodexResponses,
		SupportedInputs: []sigma.ContentBlockType{
			sigma.ContentBlockText,
			sigma.ContentBlockImage,
		},
		SupportsTools:                true,
		SupportsThinking:             true,
		DefaultTransport:             sigma.TransportSSE,
		OpenAICodexResponses:         &sigma.OpenAICodexResponsesConfig{},
		InputCostPerMillion:          1,
		OutputCostPerMillion:         2,
		ProviderMetadata:             map[string]any{"requiresOAuth": true},
		ThinkingLevelMap:             map[sigma.ThinkingLevel]string{sigma.ThinkingLevelHigh: "high"},
		MaxOutputTokens:              8192,
		CacheReadInputCostPerMillion: 0.5,
	}
}

func codexTokenProvider(token string) sigma.OAuthTokenProvider {
	return sigma.OAuthTokenProviderFunc(func(context.Context, sigma.Model, sigma.Options) (sigma.Credential, error) {
		return sigma.Credential{
			Type:  sigma.CredentialTypeOAuthToken,
			Value: token,
			Metadata: map[string]any{
				"accountID": "acct_codex",
			},
		}, nil
	})
}
