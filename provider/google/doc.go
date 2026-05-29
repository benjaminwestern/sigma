// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

// Package google adapts Google Generative AI and Google Vertex AI to sigma.
//
// The Gemini API provider uses API-key credentials through sigma auth
// resolvers. The Vertex provider keeps project, location, endpoint, and OAuth or
// ADC-style token plumbing inside this package so the root sigma package does
// not expose Google SDK types.
package google
