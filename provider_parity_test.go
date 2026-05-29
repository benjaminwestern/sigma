// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigma_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"
)

func TestProviderParityDocsCoverBuiltInAPIs(t *testing.T) {
	t.Parallel()

	parity := readDoc(t, "docs/provider-parity.md")
	sourceMap := readDoc(t, "docs/source-capability-map.md")

	for _, api := range rootAPIConstants(t) {
		assertDocMentions(t, parity, api, "docs/provider-parity.md")
		assertDocMentions(t, sourceMap, api, "docs/source-capability-map.md")
	}
}

func readDoc(t *testing.T, path string) string {
	t.Helper()

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile(%q) returned error: %v", path, err)
	}
	return string(data)
}

func assertDocMentions(t *testing.T, doc string, api string, path string) {
	t.Helper()

	if api == "" {
		return
	}
	if !strings.Contains(doc, "`"+api+"`") {
		t.Fatalf("%s does not mention API %q in backticks", path, api)
	}
}

func rootAPIConstants(t *testing.T) []string {
	t.Helper()

	values := map[string]struct{}{}
	for _, path := range []string{"types.go", "image_models.go"} {
		for _, value := range apiConstantsInFile(t, path) {
			values[value] = struct{}{}
		}
	}

	apis := make([]string, 0, len(values))
	for value := range values {
		apis = append(apis, value)
	}
	sort.Strings(apis)
	return apis
}

func apiConstantsInFile(t *testing.T, path string) []string {
	t.Helper()

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, path, nil, 0)
	if err != nil {
		t.Fatalf("ParseFile(%q) returned error: %v", path, err)
	}

	var values []string
	for _, decl := range file.Decls {
		genDecl, ok := decl.(*ast.GenDecl)
		if !ok || genDecl.Tok != token.CONST {
			continue
		}
		var carriedType string
		for _, spec := range genDecl.Specs {
			valueSpec := spec.(*ast.ValueSpec)
			if valueSpec.Type != nil {
				ident, ok := valueSpec.Type.(*ast.Ident)
				if !ok {
					carriedType = ""
					continue
				}
				carriedType = ident.Name
			}
			if carriedType != "API" && carriedType != "ImageAPI" {
				continue
			}
			for _, expr := range valueSpec.Values {
				lit, ok := expr.(*ast.BasicLit)
				if !ok || lit.Kind != token.STRING {
					continue
				}
				value, err := strconv.Unquote(lit.Value)
				if err != nil {
					t.Fatalf("Unquote(%q) returned error: %v", lit.Value, err)
				}
				values = append(values, value)
			}
		}
	}
	return values
}
