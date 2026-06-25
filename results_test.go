// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigma_test

import (
	"encoding/json"
	"testing"

	"github.com/wintermi/sigma"
)

func TestAssistantMessageSourcesExtractsCopiedProviderMetadata(t *testing.T) {
	t.Parallel()

	message := sigma.AssistantMessage{
		ProviderMetadata: map[string]any{
			"sources": []map[string]any{
				{
					"type":       "url",
					"id":         "citation_0",
					"url":        "https://example.com",
					"title":      "Example",
					"startIndex": 2,
					"endIndex":   9,
					"raw":        "kept",
				},
				{
					"type":  "retrievedContext",
					"uri":   "gs://bucket/doc.pdf",
					"title": "Doc",
				},
			},
		},
	}

	sources := message.Sources()
	if got, want := len(sources), 2; got != want {
		t.Fatalf("source count = %d, want %d", got, want)
	}
	if got, want := sources[0].Type, "url"; got != want {
		t.Fatalf("source type = %q, want %q", got, want)
	}
	if got, want := sources[0].ID, "citation_0"; got != want {
		t.Fatalf("source id = %q, want %q", got, want)
	}
	if got, want := sources[0].URL, "https://example.com"; got != want {
		t.Fatalf("source url = %q, want %q", got, want)
	}
	if sources[0].StartIndex == nil || *sources[0].StartIndex != 2 {
		t.Fatalf("start index = %#v, want 2", sources[0].StartIndex)
	}
	if sources[0].EndIndex == nil || *sources[0].EndIndex != 9 {
		t.Fatalf("end index = %#v, want 9", sources[0].EndIndex)
	}
	if got, want := sources[0].ProviderMetadata["raw"], "kept"; got != want {
		t.Fatalf("provider metadata raw = %v, want %v", got, want)
	}

	sources[0].ProviderMetadata["raw"] = "mutated"
	raw := message.ProviderMetadata["sources"].([]map[string]any)[0]["raw"]
	if got, want := raw, "kept"; got != want {
		t.Fatalf("original provider metadata raw = %v, want %v", got, want)
	}

	if got, want := sources[1].URI, "gs://bucket/doc.pdf"; got != want {
		t.Fatalf("source uri = %q, want %q", got, want)
	}
}

func TestAssistantMessageResponseIDReadsProviderMetadata(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		metadata map[string]any
		want     string
	}{
		{
			name:     "id",
			metadata: map[string]any{"id": "resp_123"},
			want:     "resp_123",
		},
		{
			name:     "response_id fallback",
			metadata: map[string]any{"response_id": "resp_snake"},
			want:     "resp_snake",
		},
		{
			name:     "responseId fallback",
			metadata: map[string]any{"responseId": "resp_camel"},
			want:     "resp_camel",
		},
		{
			name:     "first non-empty",
			metadata: map[string]any{"id": "", "response_id": "resp_fallback"},
			want:     "resp_fallback",
		},
		{
			name:     "non-string ignored",
			metadata: map[string]any{"id": 123, "response_id": ""},
			want:     "",
		},
		{
			name:     "missing",
			metadata: nil,
			want:     "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			message := sigma.AssistantMessage{ProviderMetadata: tt.metadata}
			if got := message.ResponseID(); got != tt.want {
				t.Fatalf("ResponseID() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestContentBlockCitationsExtractsJSONDecodedMetadata(t *testing.T) {
	t.Parallel()

	var decoded map[string]any
	if err := json.Unmarshal([]byte(`{"citations":[{"type":"web_search_result_location","url":"https://example.com","title":"Example","cited_text":"fact","start_index":1,"end_index":5},{"ignored":true},"bad"]}`), &decoded); err != nil {
		t.Fatalf("Unmarshal returned error: %v", err)
	}
	block := sigma.ContentBlock{ProviderMetadata: decoded}

	citations := block.Citations()
	if got, want := len(citations), 2; got != want {
		t.Fatalf("citation count = %d, want %d", got, want)
	}
	if got, want := citations[0].Type, "web_search_result_location"; got != want {
		t.Fatalf("citation type = %q, want %q", got, want)
	}
	if got, want := citations[0].URL, "https://example.com"; got != want {
		t.Fatalf("citation url = %q, want %q", got, want)
	}
	if got, want := citations[0].CitedText, "fact"; got != want {
		t.Fatalf("cited text = %q, want %q", got, want)
	}
	if citations[0].StartIndex == nil || *citations[0].StartIndex != 1 {
		t.Fatalf("start index = %#v, want 1", citations[0].StartIndex)
	}
	if citations[0].EndIndex == nil || *citations[0].EndIndex != 5 {
		t.Fatalf("end index = %#v, want 5", citations[0].EndIndex)
	}
	if got, want := citations[1].ProviderMetadata["ignored"], true; got != want {
		t.Fatalf("second citation metadata = %v, want %v", got, want)
	}
}

func TestAssistantMessageCitationsCollectsContentCitations(t *testing.T) {
	t.Parallel()

	message := sigma.AssistantMessage{
		Content: []sigma.ContentBlock{
			sigma.Text("one"),
			{
				Type: sigma.ContentBlockText,
				Text: "two",
				ProviderMetadata: map[string]any{
					"citations": []map[string]any{{"url": "https://one.example"}},
				},
			},
			{
				Type: sigma.ContentBlockText,
				Text: "three",
				ProviderMetadata: map[string]any{
					"citations": []map[string]any{{"url": "https://two.example"}},
				},
			},
		},
	}

	citations := message.Citations()
	if got, want := len(citations), 2; got != want {
		t.Fatalf("citation count = %d, want %d", got, want)
	}
	if got, want := citations[0].URL, "https://one.example"; got != want {
		t.Fatalf("first citation url = %q, want %q", got, want)
	}
	if got, want := citations[1].URL, "https://two.example"; got != want {
		t.Fatalf("second citation url = %q, want %q", got, want)
	}
}

func TestResultAccessorsIgnoreMalformedMetadata(t *testing.T) {
	t.Parallel()

	message := sigma.AssistantMessage{
		ProviderMetadata: map[string]any{
			"sources": []any{
				"bad",
				map[string]any{},
				map[string]any{"url": "https://example.com"},
			},
		},
		Content: []sigma.ContentBlock{{
			ProviderMetadata: map[string]any{
				"citations": "bad",
			},
		}},
	}

	sources := message.Sources()
	if got, want := len(sources), 1; got != want {
		t.Fatalf("source count = %d, want %d", got, want)
	}
	if got, want := sources[0].URL, "https://example.com"; got != want {
		t.Fatalf("source url = %q, want %q", got, want)
	}
	if citations := message.Citations(); len(citations) != 0 {
		t.Fatalf("citations = %#v, want none", citations)
	}
}
