// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package streamblocks

import (
	"reflect"
	"testing"
)

func TestTextAndThinkingAppendAndReplace(t *testing.T) {
	t.Parallel()

	text := &Text{ContentIndex: 3}
	if got, want := text.Append("hel"), "hel"; got != want {
		t.Fatalf("text after first append = %q, want %q", got, want)
	}
	if got, want := text.Append("lo"), "hello"; got != want {
		t.Fatalf("text after second append = %q, want %q", got, want)
	}
	text.Set("done")
	if got, want := text.String(), "done"; got != want {
		t.Fatalf("text after set = %q, want %q", got, want)
	}

	thinking := &Thinking{}
	thinking.Signature = "sig"
	if got, want := thinking.Append("a"), "a"; got != want {
		t.Fatalf("thinking after first append = %q, want %q", got, want)
	}
	if got, want := thinking.Append("b"), "ab"; got != want {
		t.Fatalf("thinking after second append = %q, want %q", got, want)
	}
	if got, want := thinking.Signature, "sig"; got != want {
		t.Fatalf("signature = %q, want %q", got, want)
	}
}

func TestToolCallPartialMetadataModes(t *testing.T) {
	t.Parallel()

	call := &ToolCall{ContentIndex: 1}
	call.SetID("call_1")
	call.SetName("lookup")
	call.AppendArguments(`{"city":`)

	partial := call.Partial(`{"city":`, ToolPartialArgumentsText)
	if partial.ProviderMetadata == nil {
		t.Fatal("partial metadata missing")
	}
	if got, want := partial.ProviderMetadata["argumentsText"], `{"city":`; got != want {
		t.Fatalf("argumentsText = %v, want %v", got, want)
	}
	if _, ok := partial.ProviderMetadata["arguments"]; ok {
		t.Fatal("partial metadata decoded incomplete JSON")
	}

	call.AppendArguments(`"Paris"}`)
	partial = call.Partial(`"Paris"}`, ToolPartialArgumentsText)
	if got, want := partial.ProviderMetadata["argumentsText"], `{"city":"Paris"}`; got != want {
		t.Fatalf("argumentsText = %v, want %v", got, want)
	}
	if got, want := partial.ProviderMetadata["arguments"], map[string]any{"city": "Paris"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("arguments = %#v, want %#v", got, want)
	}
}

func TestToolCallArgumentsValueFallsBackToRawText(t *testing.T) {
	t.Parallel()

	call := &ToolCall{}
	if got, want := call.ArgumentsValue(), map[string]any{}; !reflect.DeepEqual(got, want) {
		t.Fatalf("empty arguments = %#v, want %#v", got, want)
	}

	call.SetArguments(`{"ok":true}`)
	if got, want := call.ToolCall().Arguments, map[string]any{"ok": true}; !reflect.DeepEqual(got, want) {
		t.Fatalf("decoded arguments = %#v, want %#v", got, want)
	}

	call.SetArguments(`{"ok":`)
	if got, want := call.ArgumentsValue(), `{"ok":`; got != want {
		t.Fatalf("invalid arguments = %#v, want %#v", got, want)
	}
}
