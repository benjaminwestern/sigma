// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

// Package fireworks adapts Fireworks AI's OpenAI-compatible Chat Completions and
// Anthropic-compatible Messages endpoints to sigma.
//
// The default provider reuses sigma's OpenAI Chat Completions adapter with
// Fireworks defaults. RegisterAnthropic reuses sigma's Anthropic Messages
// adapter under ProviderFireworksAnthropic for Fireworks models served through
// the /messages endpoint. Credentials resolve through sigma.Options.AuthResolver
// instead of direct environment reads.
package fireworks
