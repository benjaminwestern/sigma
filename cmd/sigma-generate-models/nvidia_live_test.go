// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package main

import (
	"errors"
	"strings"
	"testing"

	"github.com/wintermi/sigma/internal/modeldata"
)

func TestRenderNVIDIALiveValidationSkipsFetchWhenDisabled(t *testing.T) {
	t.Parallel()

	called := false
	output, err := renderNVIDIALiveValidation(modeldata.Catalog{}, false, func() ([]string, error) {
		called = true
		return nil, errors.New("fetch should not run")
	})
	if err != nil {
		t.Fatalf("renderNVIDIALiveValidation returned error: %v", err)
	}
	if output != "" {
		t.Fatalf("output = %q, want empty output", output)
	}
	if called {
		t.Fatal("fetch ran while NVIDIA live validation was disabled")
	}
}

func TestValidateNVIDIALiveModelIDsMatchesExactAndNormalizedIDs(t *testing.T) {
	t.Parallel()

	report := validateNVIDIALiveModelIDs(nvidiaCatalog("nvidia/Nemotron_Test", "openai/gpt-oss-20b"), []string{
		"nvidia/nemotron.test",
		"openai/gpt-oss-20b",
	})
	if len(report.MissingFromLive) != 0 {
		t.Fatalf("missing = %v, want no missing catalog rows", report.MissingFromLive)
	}
	if len(report.UnreviewedLiveRows) != 0 {
		t.Fatalf("unreviewed = %v, want no unreviewed live rows", report.UnreviewedLiveRows)
	}
}

func TestDecodeNVIDIALiveModelIDsUsesLocalJSONFixture(t *testing.T) {
	t.Parallel()

	ids, err := decodeNVIDIALiveModelIDs(strings.NewReader(`{
		"data": [
			{"id": "nvidia/z-model", "object": "model"},
			{"id": "", "object": "model"},
			{"id": "nvidia/a-model", "object": "model"}
		]
	}`))
	if err != nil {
		t.Fatalf("decodeNVIDIALiveModelIDs returned error: %v", err)
	}
	if got, want := ids, []string{"nvidia/a-model", "nvidia/z-model"}; !equalStrings(got, want) {
		t.Fatalf("ids = %v, want %v", got, want)
	}
}

func TestValidateNVIDIALiveModelIDsReportsCatalogRowsMissingFromLive(t *testing.T) {
	t.Parallel()

	report := validateNVIDIALiveModelIDs(nvidiaCatalog("nvidia/not-live"), []string{"nvidia/live-only"})
	if got, want := report.MissingFromLive, []string{"nvidia/not-live"}; !equalStrings(got, want) {
		t.Fatalf("missing = %v, want %v", got, want)
	}
	if got, want := report.UnreviewedLiveRows, []string{"nvidia/live-only"}; !equalStrings(got, want) {
		t.Fatalf("unreviewed = %v, want %v", got, want)
	}

	output := report.String()
	for _, want := range []string{
		"NVIDIA live catalog validation: 1 Sigma text rows checked against 1 live rows",
		"Sigma NVIDIA text rows missing from live NIM:",
		"- nvidia/not-live",
		"live NIM rows not represented in Sigma text metadata; review before adding:",
		"- nvidia/live-only",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("report missing %q:\n%s", want, output)
		}
	}
}

func TestNVIDIALiveValidationOnlyUsesDirectNVIDIATextRows(t *testing.T) {
	t.Parallel()

	catalog := nvidiaCatalog("nvidia/text-live")
	catalog.TextModels = append(catalog.TextModels, modeldata.TextModel{
		ID:       "nvidia/embedding-live",
		Provider: "nvidia",
		API:      "openai-embeddings",
	})
	catalog.TextModels = append(catalog.TextModels, modeldata.TextModel{
		ID:       "other/live",
		Provider: "other",
		API:      "openai-completions",
	})

	report := validateNVIDIALiveModelIDs(catalog, []string{"nvidia/text-live"})
	if report.CatalogRows != 1 {
		t.Fatalf("catalog rows = %d, want 1", report.CatalogRows)
	}
	if len(report.MissingFromLive) != 0 || len(report.UnreviewedLiveRows) != 0 {
		t.Fatalf("report = %+v, want only matching NVIDIA text row considered", report)
	}
}

func nvidiaCatalog(ids ...string) modeldata.Catalog {
	catalog := modeldata.Catalog{}
	for _, id := range ids {
		catalog.TextModels = append(catalog.TextModels, modeldata.TextModel{
			ID:       id,
			Provider: "nvidia",
			API:      "openai-completions",
		})
	}
	return catalog
}

func equalStrings(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}
