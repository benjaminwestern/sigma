// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigma_test

import (
	"testing"

	"github.com/wintermi/sigma"
)

func TestRootPackageSmoke(t *testing.T) {
	t.Parallel()

	client := sigma.New(sigma.WithRegistry(&sigma.Registry{}))
	if client == nil {
		t.Fatal("New returned nil")
	}

	if err := sigma.ValidateModelRef(sigma.ModelRef{Provider: "openai", ID: "gpt-4o-mini"}); err != nil {
		t.Fatalf("valid model ref returned error: %v", err)
	}
}
