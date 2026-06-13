// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package fireworks

import (
	"net/http"

	"github.com/wintermi/sigma"
	"github.com/wintermi/sigma/provider/anthropic"
	"github.com/wintermi/sigma/provider/openai"
)

const DefaultBaseURL = "https://api.fireworks.ai/inference/v1"

// Provider adapts Fireworks AI's OpenAI-compatible Chat Completions endpoint to sigma.
type Provider = openai.Provider

// ProviderOption configures a Provider.
type ProviderOption = openai.ProviderOption

// AnthropicProvider adapts Fireworks AI's Anthropic-compatible Messages endpoint.
type AnthropicProvider = anthropic.Provider

// AnthropicProviderOption configures an AnthropicProvider.
type AnthropicProviderOption = anthropic.ProviderOption

// NewProvider constructs a Fireworks AI provider.
func NewProvider(opts ...ProviderOption) *Provider {
	providerOpts := append([]ProviderOption{openai.WithBaseURL(DefaultBaseURL)}, opts...)
	return openai.NewProvider(providerOpts...)
}

// NewAnthropicProvider constructs a Fireworks AI Anthropic-compatible provider.
func NewAnthropicProvider(opts ...AnthropicProviderOption) *AnthropicProvider {
	providerOpts := append([]AnthropicProviderOption{anthropic.WithBaseURL(DefaultBaseURL)}, opts...)
	return anthropic.NewProvider(providerOpts...)
}

// WithBaseURL configures the provider base URL, for example an httptest server URL.
func WithBaseURL(baseURL string) ProviderOption {
	return openai.WithBaseURL(baseURL)
}

// WithAnthropicBaseURL configures the Anthropic-compatible provider base URL,
// for example an httptest server URL ending in /v1.
func WithAnthropicBaseURL(baseURL string) AnthropicProviderOption {
	return anthropic.WithBaseURL(baseURL)
}

// WithHTTPClient configures the provider fallback HTTP client.
func WithHTTPClient(client *http.Client) ProviderOption {
	return openai.WithHTTPClient(client)
}

// WithAnthropicHTTPClient configures the Anthropic-compatible provider fallback
// HTTP client.
func WithAnthropicHTTPClient(client *http.Client) AnthropicProviderOption {
	return anthropic.WithHTTPClient(client)
}

// WithHeader configures a provider default request header.
func WithHeader(key, value string) ProviderOption {
	return openai.WithHeader(key, value)
}

// WithAnthropicHeader configures an Anthropic-compatible provider default
// request header.
func WithAnthropicHeader(key, value string) AnthropicProviderOption {
	return anthropic.WithHeader(key, value)
}

// WithHeaders configures provider default request headers.
func WithHeaders(headers map[string]string) ProviderOption {
	return openai.WithHeaders(headers)
}

// WithAnthropicHeaders configures Anthropic-compatible provider default request
// headers.
func WithAnthropicHeaders(headers map[string]string) AnthropicProviderOption {
	return anthropic.WithHeaders(headers)
}

// Register adds a Fireworks AI text provider to registry.
func Register(registry *sigma.Registry, opts ...ProviderOption) error {
	if registry == nil {
		return &sigma.Error{Code: sigma.ErrorUnsupported, Message: "registry is required"}
	}
	return registry.RegisterTextProvider(sigma.ProviderFireworks, NewProvider(opts...))
}

// RegisterAnthropic adds a Fireworks AI Anthropic-compatible text provider to
// registry.
func RegisterAnthropic(registry *sigma.Registry, opts ...AnthropicProviderOption) error {
	if registry == nil {
		return &sigma.Error{Code: sigma.ErrorUnsupported, Message: "registry is required"}
	}
	return registry.RegisterTextProvider(sigma.ProviderFireworksAnthropic, NewAnthropicProvider(opts...))
}

// RegisterDefault adds a Fireworks AI text provider to sigma's default registry.
func RegisterDefault(opts ...ProviderOption) error {
	return sigma.RegisterDefaultTextProvider(sigma.ProviderFireworks, NewProvider(opts...))
}

// RegisterDefaultAnthropic adds a Fireworks AI Anthropic-compatible text
// provider to sigma's default registry.
func RegisterDefaultAnthropic(opts ...AnthropicProviderOption) error {
	return sigma.RegisterDefaultTextProvider(sigma.ProviderFireworksAnthropic, NewAnthropicProvider(opts...))
}
