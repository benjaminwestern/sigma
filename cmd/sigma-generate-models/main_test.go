// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package main

import (
	"bytes"
	"go/format"
	"strings"
	"testing"

	"github.com/wintermi/sigma/internal/modeldata"
)

func TestRenderGeneratedFilesIsDeterministic(t *testing.T) {
	t.Parallel()

	catalog, err := modeldata.Load("../../internal/modeldata/catalog.json")
	if err != nil {
		t.Fatalf("Load returned error: %v", err)
	}

	firstText, err := format.Source(renderTextModels(catalog))
	if err != nil {
		t.Fatalf("format first text models: %v", err)
	}
	secondText, err := format.Source(renderTextModels(catalog))
	if err != nil {
		t.Fatalf("format second text models: %v", err)
	}
	if !bytes.Equal(firstText, secondText) {
		t.Fatal("text model rendering was not deterministic")
	}

	firstImages, err := format.Source(renderImageModels(catalog))
	if err != nil {
		t.Fatalf("format first image models: %v", err)
	}
	secondImages, err := format.Source(renderImageModels(catalog))
	if err != nil {
		t.Fatalf("format second image models: %v", err)
	}
	if !bytes.Equal(firstImages, secondImages) {
		t.Fatal("image model rendering was not deterministic")
	}
	firstEmbeddings, err := format.Source(renderEmbeddingModels(catalog))
	if err != nil {
		t.Fatalf("format first embedding models: %v", err)
	}
	secondEmbeddings, err := format.Source(renderEmbeddingModels(catalog))
	if err != nil {
		t.Fatalf("format second embedding models: %v", err)
	}
	if !bytes.Equal(firstEmbeddings, secondEmbeddings) {
		t.Fatal("embedding model rendering was not deterministic")
	}
	if !strings.Contains(string(firstText), "Source snapshot date: 2026-06-10") {
		t.Fatal("generated text models missing source snapshot date")
	}
	if !strings.Contains(string(firstText), "https://platform.openai.com/docs/models") {
		t.Fatal("generated text models missing source URL")
	}
}

func TestRenderCatalogReportSummarizesProviderAPIBuckets(t *testing.T) {
	t.Parallel()

	report := renderCatalogReport(modeldata.Catalog{
		SnapshotDate: "2026-06-10",
		Sources: []modeldata.Source{
			{Name: "one", URL: "https://example.test/one"},
			{Name: "two", URL: "https://example.test/two"},
		},
		TextModels: []modeldata.TextModel{
			{Provider: "z-provider", API: "chat", SupportsTools: true, SupportsThinking: true},
			{Provider: "a-provider", API: "chat", SupportsTools: true},
			{Provider: "a-provider", API: "responses", ThinkingLevelMap: map[string]string{"high": "high"}},
		},
		ImageModels: []modeldata.ImageModel{
			{Provider: "openrouter", API: "openrouter-images"},
		},
		EmbeddingModels: []modeldata.EmbeddingModel{
			{Provider: "openai", API: "openai-embeddings"},
			{Provider: "openai", API: "openai-embeddings"},
		},
	})

	for _, want := range []string{
		"Catalog snapshot: 2026-06-10\n",
		"Sources: 2\n",
		"Text models: 3 (tools: 2, reasoning: 2)\n",
		"  a-provider / chat: 1\n",
		"  a-provider / responses: 1\n",
		"  z-provider / chat: 1\n",
		"Image models: 1\n",
		"  openrouter / openrouter-images: 1\n",
		"Embedding models: 2\n",
		"  openai / openai-embeddings: 2\n",
	} {
		if !strings.Contains(report, want) {
			t.Fatalf("catalog report missing %q:\n%s", want, report)
		}
	}
	if strings.Index(report, "a-provider / chat") > strings.Index(report, "z-provider / chat") {
		t.Fatalf("catalog report buckets are not sorted:\n%s", report)
	}
}
