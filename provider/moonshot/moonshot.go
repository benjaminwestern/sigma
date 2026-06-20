// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package moonshot

import (
	"net/http"

	"github.com/wintermi/sigma"
	"github.com/wintermi/sigma/provider/openai"
)

const (
	// DefaultBaseURL is the Moonshot AI OpenAI-compatible API base URL.
	DefaultBaseURL = "https://api.moonshot.ai/v1"
	// DefaultCNBaseURL is the Moonshot AI CN OpenAI-compatible API base URL.
	DefaultCNBaseURL = "https://api.moonshot.cn/v1"
)

// Provider adapts Moonshot's OpenAI-compatible Chat Completions endpoint to sigma.
type Provider = openai.Provider

// ProviderOption configures a Provider.
type ProviderOption = openai.ProviderOption

// NewProvider constructs a Moonshot AI provider.
func NewProvider(opts ...ProviderOption) *Provider {
	providerOpts := append([]ProviderOption{openai.WithBaseURL(DefaultBaseURL)}, opts...)
	return openai.NewProvider(providerOpts...)
}

// NewCNProvider constructs a Moonshot AI CN provider.
func NewCNProvider(opts ...ProviderOption) *Provider {
	providerOpts := append([]ProviderOption{openai.WithBaseURL(DefaultCNBaseURL)}, opts...)
	return openai.NewProvider(providerOpts...)
}

// WithBaseURL configures the provider base URL, for example an httptest server URL.
func WithBaseURL(baseURL string) ProviderOption {
	return openai.WithBaseURL(baseURL)
}

// WithHTTPClient configures the provider fallback HTTP client.
func WithHTTPClient(client *http.Client) ProviderOption {
	return openai.WithHTTPClient(client)
}

// WithHeader configures a provider default request header.
func WithHeader(key, value string) ProviderOption {
	return openai.WithHeader(key, value)
}

// WithHeaders configures provider default request headers.
func WithHeaders(headers map[string]string) ProviderOption {
	return openai.WithHeaders(headers)
}

// Register adds a Moonshot AI text provider to registry.
func Register(registry *sigma.Registry, opts ...ProviderOption) error {
	if registry == nil {
		return &sigma.Error{Code: sigma.ErrorUnsupported, Message: "registry is required"}
	}
	return registry.RegisterTextProvider(sigma.ProviderMoonshotAI, NewProvider(opts...))
}

// RegisterCN adds a Moonshot AI CN text provider to registry.
func RegisterCN(registry *sigma.Registry, opts ...ProviderOption) error {
	if registry == nil {
		return &sigma.Error{Code: sigma.ErrorUnsupported, Message: "registry is required"}
	}
	return registry.RegisterTextProvider(sigma.ProviderMoonshotAICN, NewCNProvider(opts...))
}

// RegisterDefault adds a Moonshot AI text provider to sigma's default registry.
func RegisterDefault(opts ...ProviderOption) error {
	return sigma.RegisterDefaultTextProvider(sigma.ProviderMoonshotAI, NewProvider(opts...))
}

// RegisterDefaultCN adds a Moonshot AI CN text provider to sigma's default registry.
func RegisterDefaultCN(opts ...ProviderOption) error {
	return sigma.RegisterDefaultTextProvider(sigma.ProviderMoonshotAICN, NewCNProvider(opts...))
}
