# Sigma Examples

These examples are ordinary Go programs. They use deterministic `sigmatest`
providers by default, so they compile and run without live credentials.

Run all examples through the repository test suite from the repository root:

```sh
mise run go:test
```

Run one example:

```sh
go run ./examples/chat
go run ./examples/stream
go run ./examples/tools
go run ./examples/images
go run ./examples/cancel
go run ./examples/custom-model
```

## Live Text Provider

`examples/chat` defaults to `sigmatest`. To run it against OpenAI Chat
Completions instead:

```sh
SIGMA_EXAMPLE_PROVIDER=openai OPENAI_API_KEY=... go run ./examples/chat
```

Optional:

```sh
SIGMA_EXAMPLE_MODEL=gpt-4o-mini
```

## Live Fireworks Firepass Demo

`examples/fireworks` contains a live CLI demo for the Fireworks Firepass Kimi
router. It requires `FIREWORKS_API_KEY`.

```sh
FIREWORKS_API_KEY=... go run ./examples/fireworks
```

## Local OpenAI-Compatible Endpoint

`examples/custom-model` registers a custom OpenAI-compatible model and exits
without making a network call unless `SIGMA_LOCAL_BASE_URL` is set.

Examples:

```sh
SIGMA_LOCAL_BASE_URL=http://localhost:11434/v1 SIGMA_LOCAL_MODEL=llama3.2 go run ./examples/custom-model
SIGMA_LOCAL_BASE_URL=http://localhost:1234/v1 SIGMA_LOCAL_MODEL=local-model go run ./examples/custom-model
```

Optional:

```sh
SIGMA_LOCAL_API_KEY=local
```

The custom model demonstrates compatibility overrides for local
Ollama/vLLM/LM Studio-style endpoints: unsupported store/developer-role fields,
streaming usage, strict tools, reasoning, prompt cache control, and tool-result
message quirks are configured explicitly on `sigma.OpenAICompletionsCompat`.
