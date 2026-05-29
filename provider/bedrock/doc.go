// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

// Package bedrock adapts Amazon Bedrock Converse Stream to sigma.
//
// The provider keeps AWS-specific regions, credential discovery, endpoints,
// model IDs, inference profiles, SigV4 signing, and EventStream parsing inside
// this package. Tests can inject ConverseStreamClient and CredentialDetector
// fakes; production calls use the stdlib HTTP client.
//
// Bedrock model families do not expose identical Converse behavior. Anthropic
// Claude models support inline images, tools, prompt cache points, and extended
// thinking through model-specific fields. Amazon Nova and other families differ
// in tool-result status, image support, cache support, and reasoning controls.
// This adapter maps provider-neutral sigma features only where Converse exposes
// a stable field or a documented model-specific extension hook.
package bedrock
