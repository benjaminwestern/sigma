// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package openai_test

import (
	"reflect"
	"testing"

	"github.com/wintermi/sigma/provider/openai"
)

func TestToolsWebSearchBuildsProviderDefinedTool(t *testing.T) {
	t.Parallel()

	tool := openai.Tools.WebSearch(
		openai.WithSearchContextSize("high"),
		openai.WithUserLocation(openai.WebSearchLocation{
			Country:  "AU",
			City:     "Melbourne",
			Region:   "Victoria",
			Timezone: "Australia/Melbourne",
		}),
		openai.WithSearchFilters(openai.WebSearchFilters{AllowedDomains: []string{"example.com"}}),
		openai.WithExternalWebAccess(false),
	)

	if got, want := tool.Name, "web_search"; got != want {
		t.Fatalf("name = %q, want %q", got, want)
	}
	if got, want := tool.ProviderDefinedType, "web_search"; got != want {
		t.Fatalf("type = %q, want %q", got, want)
	}
	want := map[string]any{
		"search_context_size": "high",
		"user_location": map[string]any{
			"type":     "approximate",
			"country":  "AU",
			"city":     "Melbourne",
			"region":   "Victoria",
			"timezone": "Australia/Melbourne",
		},
		"filters":             map[string]any{"allowed_domains": []string{"example.com"}},
		"external_web_access": false,
	}
	if !reflect.DeepEqual(tool.ProviderDefinedOptions, want) {
		t.Fatalf("options = %#v, want %#v", tool.ProviderDefinedOptions, want)
	}
}

func TestToolsCodeInterpreterBuildsProviderDefinedTool(t *testing.T) {
	t.Parallel()

	tool := openai.Tools.CodeInterpreter(openai.WithContainerID("container_123"))

	if got, want := tool.Name, "code_interpreter"; got != want {
		t.Fatalf("name = %q, want %q", got, want)
	}
	if got, want := tool.ProviderDefinedType, "code_interpreter"; got != want {
		t.Fatalf("type = %q, want %q", got, want)
	}
	if got, want := tool.ProviderDefinedOptions["container"], "container_123"; got != want {
		t.Fatalf("container = %v, want %v", got, want)
	}
}

func TestToolsFileSearchBuildsProviderDefinedTool(t *testing.T) {
	t.Parallel()

	tool := openai.Tools.FileSearch(
		openai.WithVectorStoreIDs("vs_1", "vs_2"),
		openai.WithMaxNumResults(8),
		openai.WithRanking(openai.FileSearchRanking{Ranker: "auto", ScoreThreshold: 0.4}),
		openai.WithFileSearchFilters(&openai.FileSearchComparisonFilter{
			Key:   "department",
			Type:  "eq",
			Value: "support",
		}),
	)

	if got, want := tool.Name, "file_search"; got != want {
		t.Fatalf("name = %q, want %q", got, want)
	}
	if got, want := tool.ProviderDefinedType, "file_search"; got != want {
		t.Fatalf("type = %q, want %q", got, want)
	}
	want := map[string]any{
		"vector_store_ids": []string{"vs_1", "vs_2"},
		"max_num_results":  8,
		"ranking_options": map[string]any{
			"ranker":          "auto",
			"score_threshold": 0.4,
		},
		"filters": map[string]any{
			"key":   "department",
			"type":  "eq",
			"value": "support",
		},
	}
	if !reflect.DeepEqual(tool.ProviderDefinedOptions, want) {
		t.Fatalf("options = %#v, want %#v", tool.ProviderDefinedOptions, want)
	}
}

func TestToolsImageGenerationBuildsProviderDefinedTool(t *testing.T) {
	t.Parallel()

	tool := openai.Tools.ImageGeneration(
		openai.WithInputImageMask(openai.ImageGenerationMask{FileID: "file_123"}),
		openai.WithImageModel("gpt-image-1"),
		openai.WithImageQuality("high"),
		openai.WithImageSize("1024x1024"),
		openai.WithOutputFormat("png"),
		openai.WithPartialImages(2),
	)

	if got, want := tool.Name, "image_generation"; got != want {
		t.Fatalf("name = %q, want %q", got, want)
	}
	if got, want := tool.ProviderDefinedType, "image_generation"; got != want {
		t.Fatalf("type = %q, want %q", got, want)
	}
	want := map[string]any{
		"input_image_mask": map[string]any{"file_id": "file_123"},
		"model":            "gpt-image-1",
		"quality":          "high",
		"size":             "1024x1024",
		"output_format":    "png",
		"partial_images":   2,
	}
	if !reflect.DeepEqual(tool.ProviderDefinedOptions, want) {
		t.Fatalf("options = %#v, want %#v", tool.ProviderDefinedOptions, want)
	}
}
