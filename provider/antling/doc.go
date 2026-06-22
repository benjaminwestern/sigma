// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

// Package antling adapts Ant Ling's OpenAI-compatible Chat Completions endpoint
// to Sigma.
//
// The provider reuses Sigma's OpenAI Chat Completions adapter with Ant Ling
// defaults. Credentials resolve through sigma.Options.AuthResolver instead of
// direct environment reads.
package antling
