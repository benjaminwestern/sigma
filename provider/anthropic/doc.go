// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

// Package anthropic adapts Anthropic Messages-compatible APIs to sigma.
//
// The provider supports streaming text, image input, tools, thinking, prompt
// cache markers, usage, and Anthropic-compatible endpoint variations where the
// upstream API exposes stable fields. Credentials resolve through
// sigma.Options.AuthResolver instead of direct environment reads.
package anthropic
