// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigma

import "encoding/json"

// ResultSource is a normalized source entry reported by a provider response.
type ResultSource struct {
	Type             string         `json:"type,omitempty"`
	ID               string         `json:"id,omitempty"`
	URL              string         `json:"url,omitempty"`
	URI              string         `json:"uri,omitempty"`
	Title            string         `json:"title,omitempty"`
	StartIndex       *int           `json:"startIndex,omitempty"`
	EndIndex         *int           `json:"endIndex,omitempty"`
	ProviderMetadata map[string]any `json:"providerMetadata,omitempty"`
}

// ResultCitation is a normalized citation attached to assistant content.
type ResultCitation struct {
	Type             string         `json:"type,omitempty"`
	ID               string         `json:"id,omitempty"`
	URL              string         `json:"url,omitempty"`
	URI              string         `json:"uri,omitempty"`
	Title            string         `json:"title,omitempty"`
	CitedText        string         `json:"citedText,omitempty"`
	StartIndex       *int           `json:"startIndex,omitempty"`
	EndIndex         *int           `json:"endIndex,omitempty"`
	ProviderMetadata map[string]any `json:"providerMetadata,omitempty"`
}

// Sources returns normalized source entries reported on the assistant message.
func (m AssistantMessage) Sources() []ResultSource {
	return resultSources(m.ProviderMetadata["sources"])
}

// Citations returns normalized citations attached to this content block.
func (b ContentBlock) Citations() []ResultCitation {
	return resultCitations(b.ProviderMetadata["citations"])
}

// Citations returns normalized citations attached to all assistant content
// blocks, preserving content order.
func (m AssistantMessage) Citations() []ResultCitation {
	var citations []ResultCitation
	for _, block := range m.Content {
		citations = append(citations, block.Citations()...)
	}
	return citations
}

func resultSources(value any) []ResultSource {
	items := metadataItems(value)
	if len(items) == 0 {
		return nil
	}
	sources := make([]ResultSource, 0, len(items))
	for _, item := range items {
		source := ResultSource{
			Type:             stringMetadataValue(item, "type"),
			ID:               stringMetadataValue(item, "id"),
			URL:              stringMetadataValue(item, "url"),
			URI:              stringMetadataValue(item, "uri"),
			Title:            stringMetadataValue(item, "title"),
			StartIndex:       intMetadataValue(item, "startIndex", "start_index"),
			EndIndex:         intMetadataValue(item, "endIndex", "end_index"),
			ProviderMetadata: copyStringAnyMap(item),
		}
		sources = append(sources, source)
	}
	return sources
}

func resultCitations(value any) []ResultCitation {
	items := metadataItems(value)
	if len(items) == 0 {
		return nil
	}
	citations := make([]ResultCitation, 0, len(items))
	for _, item := range items {
		citation := ResultCitation{
			Type:             stringMetadataValue(item, "type"),
			ID:               stringMetadataValue(item, "id"),
			URL:              stringMetadataValue(item, "url"),
			URI:              stringMetadataValue(item, "uri"),
			Title:            stringMetadataValue(item, "title"),
			CitedText:        stringMetadataValue(item, "citedText", "cited_text"),
			StartIndex:       intMetadataValue(item, "startIndex", "start_index"),
			EndIndex:         intMetadataValue(item, "endIndex", "end_index"),
			ProviderMetadata: copyStringAnyMap(item),
		}
		citations = append(citations, citation)
	}
	return citations
}

func metadataItems(value any) []map[string]any {
	switch typed := value.(type) {
	case []map[string]any:
		items := make([]map[string]any, 0, len(typed))
		for _, item := range typed {
			if len(item) > 0 {
				items = append(items, item)
			}
		}
		return items
	case []any:
		items := make([]map[string]any, 0, len(typed))
		for _, item := range typed {
			if mapped, ok := item.(map[string]any); ok && len(mapped) > 0 {
				items = append(items, mapped)
			}
		}
		return items
	default:
		return nil
	}
}

func stringMetadataValue(metadata map[string]any, keys ...string) string {
	for _, key := range keys {
		value, ok := metadata[key].(string)
		if ok && value != "" {
			return value
		}
	}
	return ""
}

func intMetadataValue(metadata map[string]any, keys ...string) *int {
	for _, key := range keys {
		value, ok := metadata[key]
		if !ok {
			continue
		}
		if parsed, ok := metadataInt(value); ok {
			return intPtr(parsed)
		}
	}
	return nil
}

func metadataInt(value any) (int, bool) {
	switch typed := value.(type) {
	case int:
		return typed, true
	case int8:
		return int(typed), true
	case int16:
		return int(typed), true
	case int32:
		return int(typed), true
	case int64:
		return int(typed), true
	case float64:
		if typed == float64(int(typed)) {
			return int(typed), true
		}
	case json.Number:
		parsed, err := typed.Int64()
		if err == nil {
			return int(parsed), true
		}
	}
	return 0, false
}
