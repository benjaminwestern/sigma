// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/wintermi/sigma/internal/modeldata"
)

const nvidiaLiveModelsURL = "https://integrate.api.nvidia.com/v1/models"

type nvidiaLiveModelFetcher func() ([]string, error)

type nvidiaLiveModelsResponse struct {
	Data []struct {
		ID string `json:"id"`
	} `json:"data"`
}

type nvidiaLiveValidationReport struct {
	CatalogRows        int
	LiveRows           int
	MissingFromLive    []string
	UnreviewedLiveRows []string
}

func renderNVIDIALiveValidation(catalog modeldata.Catalog, enabled bool, fetch nvidiaLiveModelFetcher) (string, error) {
	if !enabled {
		return "", nil
	}
	liveIDs, err := fetch()
	if err != nil {
		return "", err
	}
	return validateNVIDIALiveModelIDs(catalog, liveIDs).String(), nil
}

func fetchNVIDIALiveModelIDs() ([]string, error) {
	client := http.Client{Timeout: 30 * time.Second}
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, nvidiaLiveModelsURL, nil)
	if err != nil {
		return nil, fmt.Errorf("build NVIDIA live models request: %w", err)
	}
	resp, err := client.Do(req) // #nosec G107 -- opt-in developer validation against a fixed provider catalog URL.
	if err != nil {
		return nil, fmt.Errorf("fetch NVIDIA live models: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("GET %s: %s", nvidiaLiveModelsURL, resp.Status)
	}

	return decodeNVIDIALiveModelIDs(resp.Body)
}

func decodeNVIDIALiveModelIDs(r io.Reader) ([]string, error) {
	var payload nvidiaLiveModelsResponse
	if err := json.NewDecoder(r).Decode(&payload); err != nil {
		return nil, fmt.Errorf("decode NVIDIA live models: %w", err)
	}
	ids := make([]string, 0, len(payload.Data))
	for _, model := range payload.Data {
		if model.ID != "" {
			ids = append(ids, model.ID)
		}
	}
	sort.Strings(ids)
	return ids, nil
}

func validateNVIDIALiveModelIDs(catalog modeldata.Catalog, liveIDs []string) nvidiaLiveValidationReport {
	catalogRows := nvidiaTextCatalogIDs(catalog)
	liveRows := normalizedIDSet(liveIDs)

	missing := make([]string, 0)
	for _, id := range catalogRows {
		if _, ok := liveRows[normalizeNVIDIAModelID(id)]; !ok {
			missing = append(missing, id)
		}
	}

	catalogSet := normalizedIDSet(catalogRows)
	unreviewed := make([]string, 0)
	for _, id := range liveIDs {
		if _, ok := catalogSet[normalizeNVIDIAModelID(id)]; !ok {
			unreviewed = append(unreviewed, id)
		}
	}

	sort.Strings(missing)
	sort.Strings(unreviewed)
	return nvidiaLiveValidationReport{
		CatalogRows:        len(catalogRows),
		LiveRows:           len(liveIDs),
		MissingFromLive:    missing,
		UnreviewedLiveRows: unreviewed,
	}
}

func (r nvidiaLiveValidationReport) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "NVIDIA live catalog validation: %d Sigma text rows checked against %d live rows\n", r.CatalogRows, r.LiveRows)
	if len(r.MissingFromLive) == 0 {
		b.WriteString("  all Sigma NVIDIA text rows matched live NIM models\n")
	} else {
		b.WriteString("  Sigma NVIDIA text rows missing from live NIM:\n")
		for _, id := range r.MissingFromLive {
			fmt.Fprintf(&b, "    - %s\n", id)
		}
	}
	if len(r.UnreviewedLiveRows) > 0 {
		b.WriteString("  live NIM rows not represented in Sigma text metadata; review before adding:\n")
		for _, id := range r.UnreviewedLiveRows {
			fmt.Fprintf(&b, "    - %s\n", id)
		}
	}
	return b.String()
}

func nvidiaTextCatalogIDs(catalog modeldata.Catalog) []string {
	ids := make([]string, 0)
	for _, model := range catalog.TextModels {
		if model.Provider == "nvidia" && model.API == "openai-completions" {
			ids = append(ids, model.ID)
		}
	}
	sort.Strings(ids)
	return ids
}

func normalizedIDSet(ids []string) map[string]struct{} {
	set := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		set[normalizeNVIDIAModelID(id)] = struct{}{}
	}
	return set
}

func normalizeNVIDIAModelID(id string) string {
	return strings.ReplaceAll(strings.ToLower(id), "_", ".")
}
