// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package xiaomi

import (
	"net/http"

	"github.com/wintermi/sigma"
	"github.com/wintermi/sigma/provider/openai"
)

const (
	// DefaultBaseURL is the Xiaomi API-billing OpenAI-compatible API base URL.
	DefaultBaseURL = "https://api.xiaomimimo.com/v1"
	// DefaultTokenPlanCNBaseURL is the Xiaomi Token Plan CN API base URL.
	DefaultTokenPlanCNBaseURL = "https://token-plan-cn.xiaomimimo.com/v1"
	// DefaultTokenPlanAMSBaseURL is the Xiaomi Token Plan AMS API base URL.
	DefaultTokenPlanAMSBaseURL = "https://token-plan-ams.xiaomimimo.com/v1"
	// DefaultTokenPlanSGPBaseURL is the Xiaomi Token Plan SGP API base URL.
	DefaultTokenPlanSGPBaseURL = "https://token-plan-sgp.xiaomimimo.com/v1"
)

// Provider adapts Xiaomi's OpenAI-compatible Chat Completions endpoint to sigma.
type Provider = openai.Provider

// ProviderOption configures a Provider.
type ProviderOption = openai.ProviderOption

// NewProvider constructs a Xiaomi API-billing provider.
func NewProvider(opts ...ProviderOption) *Provider {
	providerOpts := append([]ProviderOption{openai.WithBaseURL(DefaultBaseURL)}, opts...)
	return openai.NewProvider(providerOpts...)
}

// NewTokenPlanCNProvider constructs a Xiaomi Token Plan CN provider.
func NewTokenPlanCNProvider(opts ...ProviderOption) *Provider {
	providerOpts := append([]ProviderOption{openai.WithBaseURL(DefaultTokenPlanCNBaseURL)}, opts...)
	return openai.NewProvider(providerOpts...)
}

// NewTokenPlanAMSProvider constructs a Xiaomi Token Plan AMS provider.
func NewTokenPlanAMSProvider(opts ...ProviderOption) *Provider {
	providerOpts := append([]ProviderOption{openai.WithBaseURL(DefaultTokenPlanAMSBaseURL)}, opts...)
	return openai.NewProvider(providerOpts...)
}

// NewTokenPlanSGPProvider constructs a Xiaomi Token Plan SGP provider.
func NewTokenPlanSGPProvider(opts ...ProviderOption) *Provider {
	providerOpts := append([]ProviderOption{openai.WithBaseURL(DefaultTokenPlanSGPBaseURL)}, opts...)
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

// Register adds a Xiaomi API-billing text provider to registry.
func Register(registry *sigma.Registry, opts ...ProviderOption) error {
	if registry == nil {
		return &sigma.Error{Code: sigma.ErrorUnsupported, Message: "registry is required"}
	}
	return registry.RegisterTextProvider(sigma.ProviderXiaomi, NewProvider(opts...))
}

// RegisterTokenPlanCN adds a Xiaomi Token Plan CN text provider to registry.
func RegisterTokenPlanCN(registry *sigma.Registry, opts ...ProviderOption) error {
	if registry == nil {
		return &sigma.Error{Code: sigma.ErrorUnsupported, Message: "registry is required"}
	}
	return registry.RegisterTextProvider(sigma.ProviderXiaomiTokenPlanCN, NewTokenPlanCNProvider(opts...))
}

// RegisterTokenPlanAMS adds a Xiaomi Token Plan AMS text provider to registry.
func RegisterTokenPlanAMS(registry *sigma.Registry, opts ...ProviderOption) error {
	if registry == nil {
		return &sigma.Error{Code: sigma.ErrorUnsupported, Message: "registry is required"}
	}
	return registry.RegisterTextProvider(sigma.ProviderXiaomiTokenPlanAMS, NewTokenPlanAMSProvider(opts...))
}

// RegisterTokenPlanSGP adds a Xiaomi Token Plan SGP text provider to registry.
func RegisterTokenPlanSGP(registry *sigma.Registry, opts ...ProviderOption) error {
	if registry == nil {
		return &sigma.Error{Code: sigma.ErrorUnsupported, Message: "registry is required"}
	}
	return registry.RegisterTextProvider(sigma.ProviderXiaomiTokenPlanSGP, NewTokenPlanSGPProvider(opts...))
}

// RegisterDefault adds a Xiaomi API-billing text provider to sigma's default registry.
func RegisterDefault(opts ...ProviderOption) error {
	return sigma.RegisterDefaultTextProvider(sigma.ProviderXiaomi, NewProvider(opts...))
}

// RegisterTokenPlanCNDefault adds a Xiaomi Token Plan CN text provider to sigma's default registry.
func RegisterTokenPlanCNDefault(opts ...ProviderOption) error {
	return sigma.RegisterDefaultTextProvider(sigma.ProviderXiaomiTokenPlanCN, NewTokenPlanCNProvider(opts...))
}

// RegisterTokenPlanAMSDefault adds a Xiaomi Token Plan AMS text provider to sigma's default registry.
func RegisterTokenPlanAMSDefault(opts ...ProviderOption) error {
	return sigma.RegisterDefaultTextProvider(sigma.ProviderXiaomiTokenPlanAMS, NewTokenPlanAMSProvider(opts...))
}

// RegisterTokenPlanSGPDefault adds a Xiaomi Token Plan SGP text provider to sigma's default registry.
func RegisterTokenPlanSGPDefault(opts ...ProviderOption) error {
	return sigma.RegisterDefaultTextProvider(sigma.ProviderXiaomiTokenPlanSGP, NewTokenPlanSGPProvider(opts...))
}
