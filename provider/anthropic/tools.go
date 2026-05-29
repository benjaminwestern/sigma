// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package anthropic

import "github.com/wintermi/sigma"

const providerToolOptionTypeKey = "type"

// Tools provides factories for Anthropic server-side provider-defined tools.
var Tools = struct {
	WebSearch     func(opts ...WebSearchOption) sigma.Tool
	WebFetch      func(opts ...WebFetchOption) sigma.Tool
	CodeExecution func() sigma.Tool
}{
	WebSearch:     webSearchTool,
	WebFetch:      webFetchTool,
	CodeExecution: codeExecutionTool,
}

type WebSearchOption func(*webSearchConfig)

type webSearchConfig struct {
	maxUses        int
	allowedDomains []string
	blockedDomains []string
	userLocation   *WebSearchLocation
}

type WebSearchLocation struct {
	City     string
	Region   string
	Country  string
	Timezone string
}

func WithMaxUses(maxUses int) WebSearchOption {
	return func(config *webSearchConfig) { config.maxUses = maxUses }
}

func WithAllowedDomains(domains ...string) WebSearchOption {
	return func(config *webSearchConfig) { config.allowedDomains = domains }
}

func WithBlockedDomains(domains ...string) WebSearchOption {
	return func(config *webSearchConfig) { config.blockedDomains = domains }
}

func WithWebSearchUserLocation(location WebSearchLocation) WebSearchOption {
	return func(config *webSearchConfig) { config.userLocation = &location }
}

func webSearchTool(opts ...WebSearchOption) sigma.Tool {
	config := webSearchConfig{}
	for _, opt := range opts {
		opt(&config)
	}
	return providerTool("web_search", "web_search_20250305", webSearchOptions(config))
}

func webSearchOptions(config webSearchConfig) map[string]any {
	options := make(map[string]any)
	if config.maxUses > 0 {
		options["max_uses"] = config.maxUses
	}
	if len(config.allowedDomains) > 0 {
		options["allowed_domains"] = config.allowedDomains
	}
	if len(config.blockedDomains) > 0 {
		options["blocked_domains"] = config.blockedDomains
	}
	if config.userLocation != nil {
		location := map[string]any{providerToolOptionTypeKey: "approximate"}
		if config.userLocation.City != "" {
			location["city"] = config.userLocation.City
		}
		if config.userLocation.Region != "" {
			location["region"] = config.userLocation.Region
		}
		if config.userLocation.Country != "" {
			location["country"] = config.userLocation.Country
		}
		if config.userLocation.Timezone != "" {
			location["timezone"] = config.userLocation.Timezone
		}
		options["user_location"] = location
	}
	return options
}

type WebFetchOption func(*webFetchConfig)

type webFetchConfig struct {
	maxUses          int
	allowedDomains   []string
	blockedDomains   []string
	citations        *WebFetchCitations
	maxContentTokens int
}

type WebFetchCitations struct {
	Enabled bool
}

func WithWebFetchMaxUses(maxUses int) WebFetchOption {
	return func(config *webFetchConfig) { config.maxUses = maxUses }
}

func WithWebFetchAllowedDomains(domains ...string) WebFetchOption {
	return func(config *webFetchConfig) { config.allowedDomains = domains }
}

func WithWebFetchBlockedDomains(domains ...string) WebFetchOption {
	return func(config *webFetchConfig) { config.blockedDomains = domains }
}

func WithCitations(enabled bool) WebFetchOption {
	return func(config *webFetchConfig) { config.citations = &WebFetchCitations{Enabled: enabled} }
}

func WithMaxContentTokens(tokens int) WebFetchOption {
	return func(config *webFetchConfig) { config.maxContentTokens = tokens }
}

func webFetchTool(opts ...WebFetchOption) sigma.Tool {
	config := webFetchConfig{}
	for _, opt := range opts {
		opt(&config)
	}
	options := make(map[string]any)
	if config.maxUses > 0 {
		options["max_uses"] = config.maxUses
	}
	if len(config.allowedDomains) > 0 {
		options["allowed_domains"] = config.allowedDomains
	}
	if len(config.blockedDomains) > 0 {
		options["blocked_domains"] = config.blockedDomains
	}
	if config.citations != nil {
		options["citations"] = map[string]any{"enabled": config.citations.Enabled}
	}
	if config.maxContentTokens > 0 {
		options["max_content_tokens"] = config.maxContentTokens
	}
	return providerTool("web_fetch", "web_fetch_20260209", options)
}

func codeExecutionTool() sigma.Tool {
	return providerTool("code_execution", "code_execution_20260120", nil)
}

func providerTool(name string, providerType string, options map[string]any) sigma.Tool {
	if len(options) == 0 {
		options = nil
	}
	return sigma.Tool{
		Name:                   name,
		ProviderDefinedType:    providerType,
		ProviderDefinedOptions: options,
	}
}
