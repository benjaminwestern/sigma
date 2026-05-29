# Custom Models

Use custom model registration for local OpenAI-compatible servers, private
routers, and application-specific model metadata. Keep custom registrations in
an isolated registry unless you intentionally want package-global behavior.

```go
registry := sigma.NewRegistry()
providerID := sigma.ProviderID("ollama")

if err := openai.Register(registry, providerID); err != nil {
	return err
}

model := sigma.OpenAICompatibleModel(sigma.OpenAICompatibleModelConfig{
	ID:              "llama3.2",
	Provider:        providerID,
	BaseURL:         "http://localhost:11434/v1",
	Name:            "Ollama llama3.2",
	ContextWindow:   131072,
	MaxOutputTokens: 8192,
	SupportedInputs: []sigma.ContentBlockType{sigma.ContentBlockText},
	SupportsTools:   true,
	OpenAICompletionsCompat: &sigma.OpenAICompletionsCompat{
		SupportsStore:          sigma.OpenAICompatUnsupported,
		SupportsDeveloperRole:  sigma.OpenAICompatUnsupported,
		SupportsStreamingUsage: sigma.OpenAICompatUnsupported,
		SupportsStrictTools:    sigma.OpenAICompatUnsupported,
		ReasoningFormat:        sigma.OpenAICompletionsReasoningUnsupported,
		MaxTokensField:         sigma.OpenAICompletionsMaxTokens,
		CacheControlFormat:     sigma.OpenAICompletionsCacheControlUnsupported,
	},
})
if err := registry.RegisterModel(model); err != nil {
	return err
}

client := sigma.NewClient(sigma.WithRegistry(registry))
```

`OpenAICompatibleModel` sets the metadata needed by the OpenAI Chat
Completions-compatible provider. It requires an absolute `BaseURL`, a provider
ID, and a model ID.

## Compatibility Controls

`OpenAICompletionsCompat` lets you describe endpoint differences explicitly:

- whether `store` is accepted
- whether developer-role messages are accepted
- how reasoning is represented
- whether streaming usage is emitted
- whether strict tool schemas are accepted
- whether the endpoint expects `max_tokens` or `max_completion_tokens`
- how prompt cache markers are represented
- whether tool-result messages need a tool name
- whether an assistant message is required after a tool result
- OpenRouter and Vercel AI Gateway routing preferences

Leave fields at zero values when provider/base-URL detection is enough. Set
fields for local servers and routers whose compatibility is known to differ
from OpenAI.

## Headers And Credentials

Use `Headers` in `OpenAICompatibleModelConfig` for model-scoped headers that are
not secrets. Use `sigma.WithAPIKey`, `sigma.WithAuthResolver`, or
`sigma.WithProviderAuthResolver` for credentials.

```go
text, err := client.CompleteText(
	ctx,
	model,
	"Reply with one sentence.",
	sigma.WithAPIKey("local"),
)
```

Do not store API keys in `ProviderMetadata`; see [Security](security.md).

## Common Local Endpoints

- Ollama: `http://localhost:11434/v1`
- vLLM: `http://localhost:8000/v1`
- LM Studio: `http://localhost:1234/v1`
- Generic OpenAI-compatible proxy: configure the proxy's `/v1` base URL

Local endpoints vary widely. Start with conservative compatibility settings and
enable strict tools, streaming usage, reasoning, cache control, or developer
roles only after fixture-testing the endpoint.

The runnable [custom model example](../examples/custom-model/main.go) shows
Ollama, vLLM, LM Studio, and generic presets.

## Metadata Only

`sigma.WithMetadataOnly` allows registration of model metadata without a
provider. That is useful for catalog inspection, but calls will fail with
`sigma.ErrNoProvider` until a matching provider is registered.
