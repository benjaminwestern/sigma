// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

// Package mistral adapts the Mistral Conversations API to sigma.
//
// The provider currently implements streaming text conversations and function
// tools. Image inputs, thinking blocks, built-in connectors, append, and restart
// are intentionally not implemented. Credentials resolve through
// sigma.Options.AuthResolver instead of direct environment reads.
package mistral
