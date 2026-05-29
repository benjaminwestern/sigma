// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package google

import "github.com/wintermi/sigma"

// Tools provides factories for Google provider-defined tools.
var Tools = struct {
	GoogleSearch  func(opts ...GoogleSearchOption) sigma.Tool
	URLContext    func() sigma.Tool
	CodeExecution func() sigma.Tool
}{
	GoogleSearch:  googleSearchTool,
	URLContext:    urlContextTool,
	CodeExecution: codeExecutionTool,
}

type GoogleSearchOption func(*googleSearchConfig)

type googleSearchConfig struct {
	webSearch   bool
	imageSearch bool
	startTime   string
	endTime     string
}

func WithWebSearch() GoogleSearchOption {
	return func(config *googleSearchConfig) { config.webSearch = true }
}

func WithImageSearch() GoogleSearchOption {
	return func(config *googleSearchConfig) { config.imageSearch = true }
}

func WithTimeRange(startTime string, endTime string) GoogleSearchOption {
	return func(config *googleSearchConfig) {
		config.startTime = startTime
		config.endTime = endTime
	}
}

func googleSearchTool(opts ...GoogleSearchOption) sigma.Tool {
	config := googleSearchConfig{}
	for _, opt := range opts {
		opt(&config)
	}
	options := make(map[string]any)
	if config.webSearch || config.imageSearch {
		searchTypes := make(map[string]any)
		if config.webSearch {
			searchTypes["webSearch"] = map[string]any{}
		}
		if config.imageSearch {
			searchTypes["imageSearch"] = map[string]any{}
		}
		options["searchTypes"] = searchTypes
	}
	if config.startTime != "" && config.endTime != "" {
		options["timeRangeFilter"] = map[string]any{
			"startTime": config.startTime,
			"endTime":   config.endTime,
		}
	}
	return providerTool("google_search", "google.google_search", options)
}

func urlContextTool() sigma.Tool {
	return providerTool("url_context", "google.url_context", nil)
}

func codeExecutionTool() sigma.Tool {
	return providerTool("code_execution", "google.code_execution", nil)
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
