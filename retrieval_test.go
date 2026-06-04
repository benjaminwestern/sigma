// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigma_test

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/wintermi/sigma"
	"github.com/wintermi/sigma/sigmatest"
)

func TestSplitRetrievalTextDefaultsAndOverlap(t *testing.T) {
	t.Parallel()

	chunks, err := sigma.SplitRetrievalText(strings.Repeat("a", 1200), sigma.RetrievalSplitterConfig{})
	if err != nil {
		t.Fatalf("SplitRetrievalText returned error: %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("chunks = %d, want 2", len(chunks))
	}
	if got, want := len(chunks[0].Text), 1000; got != want {
		t.Fatalf("first chunk length = %d, want %d", got, want)
	}
	if got, want := chunks[1].StartByte, 800; got != want {
		t.Fatalf("second chunk start = %d, want %d", got, want)
	}
	if got, want := chunks[1].EndByte, 1200; got != want {
		t.Fatalf("second chunk end = %d, want %d", got, want)
	}
}

func TestSplitRetrievalTextUsesHardRuneSafeOverlap(t *testing.T) {
	t.Parallel()

	chunks, err := sigma.SplitRetrievalText("abcdefghij", sigma.RetrievalSplitterConfig{
		ChunkSize:    4,
		ChunkOverlap: 2,
		Separators:   []string{"\n\n"},
	})
	if err != nil {
		t.Fatalf("SplitRetrievalText returned error: %v", err)
	}
	got := chunkTexts(chunks)
	want := []string{"abcd", "cdef", "efgh", "ghij"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chunks = %#v, want %#v", got, want)
	}
}

func TestSplitRetrievalTextPrefersSeparators(t *testing.T) {
	t.Parallel()

	chunks, err := sigma.SplitRetrievalText("alpha beta gamma", sigma.RetrievalSplitterConfig{
		ChunkSize:     10,
		Separators:    []string{" "},
		KeepSeparator: true,
	})
	if err != nil {
		t.Fatalf("SplitRetrievalText returned error: %v", err)
	}
	got := chunkTexts(chunks)
	want := []string{"alpha ", "beta gamma"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chunks = %#v, want %#v", got, want)
	}
}

func TestSplitRetrievalTextKeepsRuneSafeByteOffsets(t *testing.T) {
	t.Parallel()

	chunks, err := sigma.SplitRetrievalText("éééé", sigma.RetrievalSplitterConfig{
		ChunkSize:    2,
		ChunkOverlap: 1,
	})
	if err != nil {
		t.Fatalf("SplitRetrievalText returned error: %v", err)
	}
	got := []sigma.RetrievalChunk{
		{Text: chunks[0].Text, StartByte: chunks[0].StartByte, EndByte: chunks[0].EndByte},
		{Text: chunks[1].Text, StartByte: chunks[1].StartByte, EndByte: chunks[1].EndByte},
		{Text: chunks[2].Text, StartByte: chunks[2].StartByte, EndByte: chunks[2].EndByte},
	}
	want := []sigma.RetrievalChunk{
		{Text: "éé", StartByte: 0, EndByte: 4},
		{Text: "éé", StartByte: 2, EndByte: 6},
		{Text: "éé", StartByte: 4, EndByte: 8},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chunks = %#v, want %#v", got, want)
	}
}

func TestSplitRetrievalTextValidationAndEmptyInput(t *testing.T) {
	t.Parallel()

	chunks, err := sigma.SplitRetrievalText("", sigma.RetrievalSplitterConfig{})
	if err != nil {
		t.Fatalf("SplitRetrievalText empty returned error: %v", err)
	}
	if len(chunks) != 0 {
		t.Fatalf("empty chunks = %#v, want none", chunks)
	}

	tests := []struct {
		name   string
		config sigma.RetrievalSplitterConfig
	}{
		{name: "negative chunk size", config: sigma.RetrievalSplitterConfig{ChunkSize: -1}},
		{name: "negative overlap", config: sigma.RetrievalSplitterConfig{ChunkSize: 10, ChunkOverlap: -1}},
		{name: "overlap too large", config: sigma.RetrievalSplitterConfig{ChunkSize: 10, ChunkOverlap: 10}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			_, err := sigma.SplitRetrievalText("hello", tt.config)
			if !errors.Is(err, sigma.ErrInvalidOptions) {
				t.Fatalf("error = %v, want ErrInvalidOptions", err)
			}
		})
	}
}

func TestSplitRetrievalDocumentsCopiesMetadata(t *testing.T) {
	t.Parallel()

	metadata := map[string]any{"source": "doc"}
	docs := []sigma.RetrievalDocument{{ID: "doc-1", Text: "hello", Metadata: metadata}}
	chunks, err := sigma.SplitRetrievalDocuments(docs, sigma.RetrievalSplitterConfig{ChunkSize: 100})
	if err != nil {
		t.Fatalf("SplitRetrievalDocuments returned error: %v", err)
	}
	metadata["source"] = "mutated"
	if got, want := chunks[0].ID, "doc-1#0"; got != want {
		t.Fatalf("chunk id = %q, want %q", got, want)
	}
	if got, want := chunks[0].DocumentID, "doc-1"; got != want {
		t.Fatalf("document id = %q, want %q", got, want)
	}
	if got, want := chunks[0].Metadata["source"], "doc"; got != want {
		t.Fatalf("metadata source = %v, want %v", got, want)
	}
}

func TestInMemoryRetrievalIndexAddsDocumentsAndSearches(t *testing.T) {
	t.Parallel()

	provider := sigmatest.NewFauxEmbeddingProvider(
		sigmatest.EmbeddingScript{Response: sigma.Embeddings{Vectors: []sigma.Embedding{
			{Index: 0, Vector: []float32{1, 0}},
			{Index: 1, Vector: []float32{0, 1}},
		}}},
		sigmatest.EmbeddingScript{Response: sigma.Embeddings{Vectors: []sigma.Embedding{
			{Index: 0, Vector: []float32{1, 0}},
		}}},
	)
	client := retrievalTestClient(t, provider)
	index := sigma.NewInMemoryRetrievalIndex(client, sigmatest.EmbeddingModel(), sigma.InMemoryRetrievalIndexConfig{
		Splitter:   sigma.RetrievalSplitterConfig{ChunkSize: 100},
		Dimensions: 2,
	})

	err := index.AddDocuments(context.Background(), []sigma.RetrievalDocument{
		{ID: "alpha", Text: "alpha text", Metadata: map[string]any{"kind": "match"}},
		{ID: "beta", Text: "beta text", Metadata: map[string]any{"kind": "miss"}},
	})
	if err != nil {
		t.Fatalf("AddDocuments returned error: %v", err)
	}
	results, err := index.Search(context.Background(), "find alpha", 2)
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("results = %d, want 2", len(results))
	}
	if got, want := results[0].Chunk.ID, "alpha#0"; got != want {
		t.Fatalf("first result id = %q, want %q", got, want)
	}
	if got, want := results[0].Chunk.Metadata["kind"], "match"; got != want {
		t.Fatalf("metadata kind = %v, want %v", got, want)
	}
	if results[0].Score <= results[1].Score {
		t.Fatalf("scores = %#v, want descending match first", results)
	}

	requests := provider.Requests()
	if len(requests) != 2 {
		t.Fatalf("provider requests = %d, want 2", len(requests))
	}
	if got, want := requests[0].Request.InputType, sigma.EmbeddingInputTypeDocument; got != want {
		t.Fatalf("add input type = %q, want %q", got, want)
	}
	if got, want := requests[1].Request.InputType, sigma.EmbeddingInputTypeQuery; got != want {
		t.Fatalf("query input type = %q, want %q", got, want)
	}
	if got, want := requests[0].Request.Dimensions, 2; got != want {
		t.Fatalf("dimensions = %d, want %d", got, want)
	}
}

func TestInMemoryRetrievalIndexAddsChunksStableSearchAndClones(t *testing.T) {
	t.Parallel()

	provider := sigmatest.NewFauxEmbeddingProvider(
		sigmatest.EmbeddingScript{Response: sigma.Embeddings{Vectors: []sigma.Embedding{
			{Index: 0, Vector: []float32{1, 0}},
			{Index: 1, Vector: []float32{1, 0}},
		}}},
		sigmatest.EmbeddingScript{Response: sigma.Embeddings{Vectors: []sigma.Embedding{
			{Index: 0, Vector: []float32{1, 0}},
		}}},
		sigmatest.EmbeddingScript{Response: sigma.Embeddings{Vectors: []sigma.Embedding{
			{Index: 0, Vector: []float32{1, 0}},
		}}},
	)
	client := retrievalTestClient(t, provider)
	index := sigma.NewInMemoryRetrievalIndex(client, sigmatest.EmbeddingModel(), sigma.InMemoryRetrievalIndexConfig{})
	chunks := []sigma.RetrievalChunk{
		{ID: "first", DocumentID: "doc", Text: "first", Metadata: map[string]any{"rank": "one"}},
		{ID: "second", DocumentID: "doc", Text: "second", Metadata: map[string]any{"rank": "two"}},
	}
	if err := index.AddChunks(context.Background(), chunks); err != nil {
		t.Fatalf("AddChunks returned error: %v", err)
	}
	chunks[0].Text = "mutated"
	chunks[0].Metadata["rank"] = "mutated"

	results, err := index.Search(context.Background(), "tie", 1)
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("results = %d, want 1", len(results))
	}
	if got, want := results[0].Chunk.ID, "first"; got != want {
		t.Fatalf("first result id = %q, want stable insertion order %q", got, want)
	}
	if got, want := results[0].Chunk.Text, "first"; got != want {
		t.Fatalf("chunk text = %q, want cloned text %q", got, want)
	}
	if got, want := results[0].Chunk.Metadata["rank"], "one"; got != want {
		t.Fatalf("metadata rank = %v, want cloned metadata %v", got, want)
	}
	results[0].Chunk.Metadata["rank"] = "changed"

	next, err := index.Search(context.Background(), "tie again", 1)
	if err != nil {
		t.Fatalf("second Search returned error: %v", err)
	}
	if got, want := next[0].Chunk.Metadata["rank"], "one"; got != want {
		t.Fatalf("second result metadata = %v, want stored metadata clone %v", got, want)
	}
}

func TestInMemoryRetrievalIndexSearchEmptyAndInvalidLimit(t *testing.T) {
	t.Parallel()

	provider := sigmatest.NewFauxEmbeddingProvider()
	client := retrievalTestClient(t, provider)
	index := sigma.NewInMemoryRetrievalIndex(client, sigmatest.EmbeddingModel(), sigma.InMemoryRetrievalIndexConfig{})

	results, err := index.Search(context.Background(), "unused", 3)
	if err != nil {
		t.Fatalf("Search empty returned error: %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("empty results = %#v, want none", results)
	}
	if len(provider.Requests()) != 0 {
		t.Fatalf("provider requests = %d, want none for empty index", len(provider.Requests()))
	}

	_, err = index.Search(context.Background(), "bad", -1)
	if !errors.Is(err, sigma.ErrInvalidOptions) {
		t.Fatalf("negative limit error = %v, want ErrInvalidOptions", err)
	}
}

func TestInMemoryRetrievalIndexPropagatesVectorCountMismatch(t *testing.T) {
	t.Parallel()

	provider := sigmatest.NewFauxEmbeddingProvider(sigmatest.EmbeddingScript{
		Response: sigma.Embeddings{Vectors: []sigma.Embedding{{Index: 0, Vector: []float32{1}}}},
	})
	client := retrievalTestClient(t, provider)
	index := sigma.NewInMemoryRetrievalIndex(client, sigmatest.EmbeddingModel(), sigma.InMemoryRetrievalIndexConfig{})

	err := index.AddChunks(context.Background(), []sigma.RetrievalChunk{
		{ID: "one", Text: "one"},
		{ID: "two", Text: "two"},
	})
	if err == nil {
		t.Fatal("AddChunks returned nil error")
	}
	if !strings.Contains(err.Error(), "returned 1 vectors for 2 inputs") {
		t.Fatalf("error = %v, want vector-count mismatch", err)
	}
}

func TestRetrievalResultDoesNotExposeVectors(t *testing.T) {
	t.Parallel()

	if _, ok := reflect.TypeOf(sigma.RetrievalResult{}).FieldByName("Vector"); ok {
		t.Fatal("RetrievalResult exposes a raw vector field")
	}
	if _, ok := reflect.TypeOf(sigma.RetrievalChunk{}).FieldByName("Vector"); ok {
		t.Fatal("RetrievalChunk exposes a raw vector field")
	}
}

func retrievalTestClient(t *testing.T, provider *sigmatest.FauxEmbeddingProvider) *sigma.Client {
	t.Helper()

	registry, err := sigmatest.EmbeddingRegistry(provider)
	if err != nil {
		t.Fatalf("EmbeddingRegistry returned error: %v", err)
	}
	return sigma.NewClient(sigma.WithRegistry(registry))
}

func chunkTexts(chunks []sigma.RetrievalChunk) []string {
	texts := make([]string, len(chunks))
	for index, chunk := range chunks {
		texts[index] = chunk.Text
	}
	return texts
}
