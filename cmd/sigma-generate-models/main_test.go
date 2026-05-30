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
	if !strings.Contains(string(firstText), "Source snapshot date: 2026-05-30") {
		t.Fatal("generated text models missing source snapshot date")
	}
	if !strings.Contains(string(firstText), "https://platform.openai.com/docs/models") {
		t.Fatal("generated text models missing source URL")
	}
}
