# Testing

Sigma tests must be deterministic by default. Do not introduce live LLM calls in
unit tests, examples, or `mise run go:test`.

## Faux Providers

Use `sigmatest` for client-level tests:

```go
provider := sigmatest.NewFauxProvider(sigmatest.Script{
	Final: sigma.AssistantMessage{
		Content: []sigma.ContentBlock{sigma.Text("ok")},
	},
})
registry, err := sigmatest.Registry(provider)
if err != nil {
	t.Fatal(err)
}

client := sigma.NewClient(sigma.WithRegistry(registry))
text, err := client.CompleteText(context.Background(), sigmatest.TextModel(), "hello")
```

`FauxProvider` records request captures so tests can assert behavior without
testing provider transports:

```go
request, ok := provider.LastRequest()
if !ok {
	t.Fatal("expected request")
}
```

Use `sigmatest.NewFauxImageProvider` and `sigmatest.ImageRegistry` for image
generation tests.

## Provider Adapter Tests

Provider package tests should use `httptest.Server`, fake provider clients, or
checked-in fixtures. They should assert behavior that would catch real bugs:

- request payload shape for branching logic
- auth precedence and credential errors
- streaming event translation
- partial tool-call JSON handling
- cancellation behavior
- provider error mapping and redaction
- retry boundaries before stream body consumption

Avoid tests that mirror static struct defaults or simply restate generated model
metadata field-by-field.

## Golden Files

Golden files are useful for provider wire payloads when the payload is a
contract. Keep them deterministic, reviewable, and free of credentials. The
helper in `internal/goldentest` is available for JSON golden comparisons.

## Live Tests

Live tests must be opt-in behind explicit environment variables and skipped by
default. They are not required for ordinary verification and should not run in
`mise run go:test` unless a maintainer intentionally enables them.

## Parallelism

Use `t.Parallel()` when a test only touches local state, `t.TempDir()`, local
buffers, or isolated registries. Do not combine `t.Parallel()` with
`t.Setenv()`, package-global mutation, shared ports, or order-dependent side
effects.

## Verification

Run:

```sh
mise run go:test
```

Docs are covered by a deterministic internal-link test. External links are not
checked so the default test suite stays offline.
