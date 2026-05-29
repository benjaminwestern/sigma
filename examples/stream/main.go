// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package main

import (
	"context"
	"fmt"
	"log"

	"github.com/wintermi/sigma"
	"github.com/wintermi/sigma/sigmatest"
)

func main() {
	provider := sigmatest.NewFauxProvider(sigmatest.Script{
		Events: []sigma.Event{
			{Kind: sigma.EventKindStart},
			{Kind: sigma.EventKindThinkingStart, ContentIndex: intPtr(0)},
			{Kind: sigma.EventKindTextStart, ContentIndex: intPtr(1)},
			{Kind: sigma.EventKindThinkingDelta, ContentIndex: intPtr(0), DeltaText: "checking context"},
			{Kind: sigma.EventKindTextDelta, ContentIndex: intPtr(1), DeltaText: "The answer is "},
			{Kind: sigma.EventKindToolCallStart, ContentIndex: intPtr(2), PartialToolCall: &sigma.PartialToolCall{ID: "call_1", Name: "lookup"}},
			{Kind: sigma.EventKindToolCallDelta, ContentIndex: intPtr(2), PartialToolCall: &sigma.PartialToolCall{ArgumentsDelta: `{"query":"sigma"}`}},
			{Kind: sigma.EventKindToolCallEnd, ContentIndex: intPtr(2), ToolCall: &sigma.ToolCall{ID: "call_1", Name: "lookup", Arguments: map[string]any{"query": "sigma"}}},
			{Kind: sigma.EventKindThinkingEnd, ContentIndex: intPtr(0), Thinking: "checking context"},
			{Kind: sigma.EventKindTextDelta, ContentIndex: intPtr(1), DeltaText: "provider-neutral."},
			{Kind: sigma.EventKindTextEnd, ContentIndex: intPtr(1), Text: "The answer is provider-neutral."},
		},
		Final: sigma.AssistantMessage{
			Content: []sigma.ContentBlock{
				sigma.Thinking("checking context", "sig"),
				sigma.Text("The answer is provider-neutral."),
				sigma.ToolCallBlock("call_1", "lookup", map[string]any{"query": "sigma"}),
			},
			StopReason: sigma.StopReasonToolCalls,
			Usage:      &sigma.Usage{InputTokens: 8, OutputTokens: 6, TotalTokens: 14},
		},
	})
	registry, err := sigmatest.Registry(provider)
	if err != nil {
		log.Fatal(err)
	}

	client := sigma.NewClient(sigma.WithRegistry(registry))
	stream := client.Stream(context.Background(), sigmatest.TextModel(), sigma.Request{
		Messages: []sigma.Message{sigma.UserText("Explain Sigma briefly.")},
	})

	textByIndex := map[int]string{}
	thinkingByIndex := map[int]string{}
	toolByIndex := map[int]sigma.ToolCall{}

	for event := range stream.Events() {
		switch event.Kind {
		case sigma.EventKindStart:
			fmt.Println("stream started")
		case sigma.EventKindTextStart:
			fmt.Printf("text[%d] started\n", contentIndex(event))
		case sigma.EventKindTextDelta:
			index := contentIndex(event)
			textByIndex[index] += event.DeltaText
			fmt.Printf("text[%d] += %q\n", index, event.DeltaText)
		case sigma.EventKindTextEnd:
			fmt.Printf("text[%d] done: %q\n", contentIndex(event), event.Text)
		case sigma.EventKindThinkingStart:
			fmt.Printf("thinking[%d] started\n", contentIndex(event))
		case sigma.EventKindThinkingDelta:
			index := contentIndex(event)
			thinkingByIndex[index] += event.DeltaText
			fmt.Printf("thinking[%d] += %q\n", index, event.DeltaText)
		case sigma.EventKindThinkingEnd:
			fmt.Printf("thinking[%d] done\n", contentIndex(event))
		case sigma.EventKindToolCallStart:
			fmt.Printf("tool[%d] started: %s\n", contentIndex(event), event.PartialToolCall.Name)
		case sigma.EventKindToolCallDelta:
			fmt.Printf("tool[%d] args += %q\n", contentIndex(event), event.PartialToolCall.ArgumentsDelta)
		case sigma.EventKindToolCallEnd:
			index := contentIndex(event)
			toolByIndex[index] = *event.ToolCall
			fmt.Printf("tool[%d] done: %s\n", index, event.ToolCall.Name)
		case sigma.EventKindDone:
			fmt.Printf("done: %s\n", event.StopReason)
		case sigma.EventKindError:
			fmt.Printf("error: %s\n", event.Error)
		default:
			fmt.Printf("unhandled event: %s\n", event.Kind)
		}
	}
	if err := stream.Err(); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("assembled text blocks: %#v\n", textByIndex)
	fmt.Printf("assembled thinking blocks: %#v\n", thinkingByIndex)
	fmt.Printf("assembled tool calls: %#v\n", toolByIndex)
}

func contentIndex(event sigma.Event) int {
	if event.ContentIndex == nil {
		return -1
	}
	return *event.ContentIndex
}

func intPtr(value int) *int {
	return &value
}
