# Release notes: sigma v0.1.0

This is the maintainer-facing release note for the first `sigma` tag. It records
the v0.1.0 summary, scope, and serialization boundary. For the itemized change
list see [CHANGELOG.md](../CHANGELOG.md); for the validation commands and pre-tag
checklist see [RELEASING.md](../RELEASING.md).

## Release summary

`sigma` v0.1.0 is a text-first Go package for provider-neutral model calls. It
ships a root package API for model metadata, request construction, completion,
streaming, tool calls, request persistence, credentials, diagnostics, retries,
and deterministic tests.

The release-ready provider surface is intentionally narrow: OpenAI-compatible
Chat Completions and Anthropic Messages are the MVP provider families. Other
implemented adapters are preview unless promoted by a later release note.

## MVP scope

The first release gate covers the text-first surface:

- The provider-neutral `Client` and `Registry`, plus the core
  request/message/tool/stream/usage/cost/image/auth/retry/persistence/error
  types, with stable JSON shapes (see [Serialization compatibility](#serialization-compatibility)).
- Text completion and streaming through registered providers, with ordered
  stream events and tool calls, including provider-defined tools (for example
  Anthropic web search, web fetch, and code execution).
- Request persistence, deterministic `sigmatest` providers, and redaction-safe
  diagnostics.
- OpenAI-compatible Chat Completions and Anthropic Messages as the two MVP
  provider families, including custom/local OpenAI-compatible endpoints and
  Anthropic-compatible routing.

Release facts:

- Root package name `sigma`; module path `github.com/wintermi/sigma`.
- Standard Major.Minor.Patch versioning, starting at `v0.1.0`. Public APIs may
  change before `v1.0.0`; breaking changes are documented in the changelog,
  release notes, and upgrade guidance.

[CHANGELOG.md](../CHANGELOG.md) holds the full itemized change list.

## Preview features

These features are implemented enough for early adopters, but they are outside
the first release gate and may change before `v1.0.0`:

- OpenAI Responses.
- Azure OpenAI Responses.
- OpenAI Codex Responses.
- Fireworks AI Chat Completions, including reasoning effort and thinking budgets.
- OpenCode Zen and OpenCode Go Chat Completions.
- Google Generative AI.
- Google Vertex AI.
- Mistral Conversations.
- Amazon Bedrock Converse Stream (stdlib HTTP, SigV4 signing, and EventStream
  parsing; no AWS SDK dependency).
- OpenRouter image generation.

## Deferred work

- OpenAI Images provider adapter. Generated metadata exists, but runtime image
  generation through OpenAI Images is not implemented.
- First-class provider rows for DeepSeek, Groq, Cerebras, xAI, Together, GitHub
  Copilot, Kimi, and Xiaomi.
- Interactive OAuth login, credential storage, and token persistence.
- WebSocket transports.
- Tokenizer-based token estimates.
- Browser-specific behavior and agent runtime integration.
- Cross-provider context handoff and capability-loss reporting.

## Serialization compatibility

The MVP treats the public JSON shapes for `Request`, `Message`, `ContentBlock`,
`Tool`, `ToolCall`, `ImageRequest`, `AssistantMessage`, `AssistantImages`,
`Usage`, and `Cost` as the compatibility boundary to preserve before `v1.0.0`
unless a release note says otherwise.

`UnmarshalRequest` rejects unknown top-level request fields. Opaque maps such as
`ProviderMetadata`, `ToolArguments`, tool schemas, provider option maps, and
provider-specific metadata remain intentionally unstable because providers may
need fields that `sigma` does not interpret.

## Validation status

This release was validated with the process in [RELEASING.md](../RELEASING.md).
The full `mise run ci` task is green, including `golangci-lint` — the previously
reported repo-wide lint backlog (`errcheck`, `exhaustive`, `goconst`, `gosec`,
`wrapcheck`, and related rules) has been resolved. The repository secret scan
surfaced only synthetic test fixtures and documented environment variable names.

See [RELEASING.md](../RELEASING.md) for the validation commands and the pre-tag
maintainer checklist.
