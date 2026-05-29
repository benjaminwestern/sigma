// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package anthropic_test

import (
	"reflect"
	"testing"

	"github.com/wintermi/sigma/provider/anthropic"
)

func TestToolsWebSearchBuildsProviderDefinedTool(t *testing.T) {
	t.Parallel()

	tool := anthropic.Tools.WebSearch(
		anthropic.WithMaxUses(3),
		anthropic.WithAllowedDomains("example.com"),
		anthropic.WithBlockedDomains("blocked.example"),
		anthropic.WithWebSearchUserLocation(anthropic.WebSearchLocation{
			City:     "Melbourne",
			Region:   "Victoria",
			Country:  "AU",
			Timezone: "Australia/Melbourne",
		}),
	)

	if got, want := tool.Name, "web_search"; got != want {
		t.Fatalf("name = %q, want %q", got, want)
	}
	if got, want := tool.ProviderDefinedType, "web_search_20250305"; got != want {
		t.Fatalf("type = %q, want %q", got, want)
	}
	want := map[string]any{
		"max_uses":        3,
		"allowed_domains": []string{"example.com"},
		"blocked_domains": []string{"blocked.example"},
		"user_location": map[string]any{
			"type":     "approximate",
			"city":     "Melbourne",
			"region":   "Victoria",
			"country":  "AU",
			"timezone": "Australia/Melbourne",
		},
	}
	if !reflect.DeepEqual(tool.ProviderDefinedOptions, want) {
		t.Fatalf("options = %#v, want %#v", tool.ProviderDefinedOptions, want)
	}
}

func TestToolsWebFetchBuildsProviderDefinedTool(t *testing.T) {
	t.Parallel()

	tool := anthropic.Tools.WebFetch(
		anthropic.WithWebFetchMaxUses(2),
		anthropic.WithWebFetchAllowedDomains("docs.example"),
		anthropic.WithWebFetchBlockedDomains("private.example"),
		anthropic.WithCitations(false),
		anthropic.WithMaxContentTokens(4096),
	)

	if got, want := tool.Name, "web_fetch"; got != want {
		t.Fatalf("name = %q, want %q", got, want)
	}
	if got, want := tool.ProviderDefinedType, "web_fetch_20260209"; got != want {
		t.Fatalf("type = %q, want %q", got, want)
	}
	want := map[string]any{
		"max_uses":           2,
		"allowed_domains":    []string{"docs.example"},
		"blocked_domains":    []string{"private.example"},
		"citations":          map[string]any{"enabled": false},
		"max_content_tokens": 4096,
	}
	if !reflect.DeepEqual(tool.ProviderDefinedOptions, want) {
		t.Fatalf("options = %#v, want %#v", tool.ProviderDefinedOptions, want)
	}
}

func TestToolsCodeExecutionBuildsProviderDefinedTool(t *testing.T) {
	t.Parallel()

	tool := anthropic.Tools.CodeExecution()

	if got, want := tool.Name, "code_execution"; got != want {
		t.Fatalf("name = %q, want %q", got, want)
	}
	if got, want := tool.ProviderDefinedType, "code_execution_20260120"; got != want {
		t.Fatalf("type = %q, want %q", got, want)
	}
	if len(tool.ProviderDefinedOptions) != 0 {
		t.Fatalf("options = %#v, want empty", tool.ProviderDefinedOptions)
	}
}
