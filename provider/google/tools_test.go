// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package google_test

import (
	"reflect"
	"testing"

	"github.com/wintermi/sigma/provider/google"
)

func TestToolsGoogleSearchBuildsProviderDefinedTool(t *testing.T) {
	t.Parallel()

	tool := google.Tools.GoogleSearch(
		google.WithWebSearch(),
		google.WithImageSearch(),
		google.WithTimeRange("2026-01-01T00:00:00Z", "2026-01-31T23:59:59Z"),
	)

	if got, want := tool.Name, "google_search"; got != want {
		t.Fatalf("name = %q, want %q", got, want)
	}
	if got, want := tool.ProviderDefinedType, "google.google_search"; got != want {
		t.Fatalf("type = %q, want %q", got, want)
	}
	want := map[string]any{
		"searchTypes": map[string]any{
			"webSearch":   map[string]any{},
			"imageSearch": map[string]any{},
		},
		"timeRangeFilter": map[string]any{
			"startTime": "2026-01-01T00:00:00Z",
			"endTime":   "2026-01-31T23:59:59Z",
		},
	}
	if !reflect.DeepEqual(tool.ProviderDefinedOptions, want) {
		t.Fatalf("options = %#v, want %#v", tool.ProviderDefinedOptions, want)
	}
}

func TestToolsURLContextBuildsProviderDefinedTool(t *testing.T) {
	t.Parallel()

	tool := google.Tools.URLContext()

	if got, want := tool.Name, "url_context"; got != want {
		t.Fatalf("name = %q, want %q", got, want)
	}
	if got, want := tool.ProviderDefinedType, "google.url_context"; got != want {
		t.Fatalf("type = %q, want %q", got, want)
	}
	if len(tool.ProviderDefinedOptions) != 0 {
		t.Fatalf("options = %#v, want empty", tool.ProviderDefinedOptions)
	}
}

func TestToolsCodeExecutionBuildsProviderDefinedTool(t *testing.T) {
	t.Parallel()

	tool := google.Tools.CodeExecution()

	if got, want := tool.Name, "code_execution"; got != want {
		t.Fatalf("name = %q, want %q", got, want)
	}
	if got, want := tool.ProviderDefinedType, "google.code_execution"; got != want {
		t.Fatalf("type = %q, want %q", got, want)
	}
	if len(tool.ProviderDefinedOptions) != 0 {
		t.Fatalf("options = %#v, want empty", tool.ProviderDefinedOptions)
	}
}
