// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

// Package sse parses provider-neutral Server-Sent Event frames.
//
// Parse understands comments, event names, ids, repeated data fields, LF/CRLF
// endings, EOF-final dispatch, and the OpenAI-style [DONE] sentinel. JSON
// decoding and provider-specific event handling belong to callers.
//
// Parse bounds individual lines and accumulated event frames. Callers may
// override those limits with WithMaxLineBytes and WithMaxEventBytes.
//
// Context cancellation is checked between reads. If a caller needs cancellation
// to interrupt a blocked network read immediately, wrap the response body with
// CloseOnContextDone or otherwise close the reader when the context is done.
package sse
