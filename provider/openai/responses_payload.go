// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package openai

import (
	"fmt"

	"github.com/wintermi/sigma"
)

const (
	providerOptionStore          = "store"
	providerOptionPreviousID     = "previous_response_id"
	providerOptionPreviousIDGo   = "previousResponseID"
	providerOptionInclude        = "include"
	providerOptionText           = "text"
	providerOptionToolChoice     = "tool_choice"
	providerOptionToolChoiceGo   = "toolChoice"
	providerOptionTruncation     = "truncation"
	providerOptionPromptCacheKey = "prompt_cache_key"
)

func responsesPayload(model sigma.Model, req sigma.Request, opts sigma.Options) (map[string]any, error) {
	input, err := responsesInput(req)
	if err != nil {
		return nil, err
	}

	payload := map[string]any{
		"model":  string(model.ID),
		"input":  input,
		"stream": true,
	}
	if req.SystemPrompt != "" {
		payload["instructions"] = req.SystemPrompt
	}
	if opts.Temperature != nil {
		payload["temperature"] = *opts.Temperature
	}
	if opts.MaxTokens != nil {
		payload["max_output_tokens"] = *opts.MaxTokens
	}
	if len(opts.Metadata) > 0 {
		payload["metadata"] = copyAnyMap(opts.Metadata)
	}
	if opts.OpenAIOptions != nil && opts.OpenAIOptions.ServiceTier != "" {
		payload["service_tier"] = opts.OpenAIOptions.ServiceTier
	}
	addResponsesReasoning(payload, model, opts)
	if len(req.Tools) > 0 {
		tools, err := responsesTools(req.Tools)
		if err != nil {
			return nil, err
		}
		payload["tools"] = tools
	}
	addResponsesSession(payload, model.Provider, opts)
	addResponsesProviderOptions(payload, model.Provider, opts)
	return payload, nil
}

func responsesInput(req sigma.Request) ([]map[string]any, error) {
	items := make([]map[string]any, 0, len(req.Messages)+1)
	for _, message := range req.Messages {
		converted, err := responsesMessage(message)
		if err != nil {
			return nil, err
		}
		items = append(items, converted...)
	}
	return items, nil
}

func responsesMessage(message sigma.Message) ([]map[string]any, error) {
	switch message.Role {
	case sigma.RoleUser, sigma.RoleDeveloper:
		content, err := responsesInputContent(message)
		if err != nil {
			return nil, err
		}
		return []map[string]any{{
			"role":    string(message.Role),
			"content": content,
		}}, nil
	case sigma.RoleAssistant:
		return responsesAssistantItems(message.Content)
	case sigma.RoleTool:
		return []map[string]any{{
			"type":    "function_call_output", //nolint:goconst
			"call_id": message.ToolCallID,
			"output":  textContent(message.Content),
		}}, nil
	default:
		return nil, fmt.Errorf("openai responses: unsupported message role %q", message.Role)
	}
}

func responsesInputContent(message sigma.Message) ([]map[string]any, error) {
	if len(message.Content) == 0 {
		return []map[string]any{{"type": "input_text", "text": ""}}, nil
	}
	parts := make([]map[string]any, 0, len(message.Content))
	for _, block := range message.Content {
		switch block.Type {
		case sigma.ContentBlockText:
			parts = append(parts, map[string]any{
				"type": "input_text",
				"text": block.Text,
			})
		case sigma.ContentBlockImage:
			if message.Role != sigma.RoleUser {
				return nil, fmt.Errorf("openai responses: image content is only supported for user messages")
			}
			url, err := imageURL(block)
			if err != nil {
				return nil, err
			}
			parts = append(parts, map[string]any{
				"type":      "input_image",
				"image_url": url,
			})
		default:
			return nil, fmt.Errorf("openai responses: unsupported input content block %q", block.Type)
		}
	}
	return parts, nil
}

func responsesAssistantItems(blocks []sigma.ContentBlock) ([]map[string]any, error) {
	var items []map[string]any
	var content []map[string]any
	flushMessage := func() {
		if len(content) == 0 {
			return
		}
		items = append(items, map[string]any{
			"type":    "message",
			"role":    "assistant",
			"content": content,
		})
		content = nil
	}

	for _, block := range blocks {
		switch block.Type {
		case sigma.ContentBlockText:
			part := map[string]any{
				"type": "output_text",
				"text": block.Text,
			}
			addProviderContentID(part, block.ProviderMetadata)
			if block.Signature != "" {
				part["signature"] = block.Signature
			}
			content = append(content, part)
		case sigma.ContentBlockThinking:
			flushMessage()
			item := map[string]any{
				"type": "reasoning",
				"summary": []map[string]any{{
					"type": "summary_text",
					"text": block.ThinkingText,
				}},
			}
			addProviderID(item, block.ProviderMetadata)
			if block.Signature != "" {
				item["signature"] = block.Signature
			}
			if block.ProviderSignature != "" {
				item["encrypted_content"] = block.ProviderSignature
			}
			items = append(items, item)
		case sigma.ContentBlockToolCall:
			flushMessage()
			arguments, err := toolArgumentsString(block.ToolArguments)
			if err != nil {
				return nil, err
			}
			item := map[string]any{
				"type":      "function_call",
				"call_id":   block.ToolCallID,
				"name":      block.ToolName,
				"arguments": arguments,
			}
			addProviderID(item, block.ProviderMetadata)
			if block.ProviderSignature != "" {
				item["encrypted_content"] = block.ProviderSignature
			}
			items = append(items, item)
		default:
			return nil, fmt.Errorf("openai responses: unsupported assistant content block %q", block.Type)
		}
	}
	flushMessage()
	return items, nil
}

func responsesTools(tools []sigma.Tool) ([]map[string]any, error) {
	converted := make([]map[string]any, 0, len(tools))
	for _, tool := range tools {
		if tool.ProviderDefinedType != "" {
			convertedTool := map[string]any{
				"type": tool.ProviderDefinedType,
			}
			for key, value := range tool.ProviderDefinedOptions {
				convertedTool[key] = value
			}
			converted = append(converted, convertedTool)
			continue
		}
		parameters, err := jsonValue(tool.InputSchema)
		if err != nil {
			return nil, fmt.Errorf("openai responses: tool %q schema: %w", tool.Name, err)
		}
		if parameters == nil {
			parameters = map[string]any{"type": "object"}
		}
		convertedTool := map[string]any{
			"type":        "function",
			"name":        tool.Name,
			"description": tool.Description,
			"parameters":  parameters,
		}
		if strict, ok := tool.ProviderMetadata["strict"].(bool); ok {
			convertedTool["strict"] = strict
		}
		converted = append(converted, convertedTool)
	}
	return converted, nil
}

func addResponsesReasoning(payload map[string]any, model sigma.Model, opts sigma.Options) {
	reasoning := make(map[string]any)
	if effort := reasoningEffort(model, opts); effort != "" {
		reasoning["effort"] = effort
	}
	if opts.ThinkingBudgetTokens != nil {
		reasoning["budget_tokens"] = *opts.ThinkingBudgetTokens
	}
	if opts.OpenAIOptions != nil && opts.OpenAIOptions.ReasoningSummary != "" {
		reasoning["summary"] = opts.OpenAIOptions.ReasoningSummary
	}
	if len(reasoning) > 0 {
		payload["reasoning"] = reasoning
	}
}

func addResponsesSession(payload map[string]any, provider sigma.ProviderID, opts sigma.Options) {
	options := providerOptions(opts, provider)
	if previous, ok := stringOption(options, providerOptionPreviousID); ok {
		payload["previous_response_id"] = previous
		return
	}
	if previous, ok := stringOption(options, providerOptionPreviousIDGo); ok {
		payload["previous_response_id"] = previous
		return
	}
	if opts.SessionID != "" {
		payload["previous_response_id"] = opts.SessionID
	}
}

func addResponsesProviderOptions(payload map[string]any, provider sigma.ProviderID, opts sigma.Options) {
	options := providerOptions(opts, provider)
	if value, ok := boolOption(options, providerOptionStore); ok {
		payload["store"] = value
	}
	if value, ok := options[providerOptionInclude]; ok {
		payload["include"] = value
	}
	if value, ok := options[providerOptionText]; ok {
		payload["text"] = value
	}
	if value, ok := options[providerOptionToolChoice]; ok {
		payload["tool_choice"] = value
	} else if value, ok := options[providerOptionToolChoiceGo]; ok {
		payload["tool_choice"] = value
	}
	if value, ok := stringOption(options, providerOptionTruncation); ok {
		payload["truncation"] = value
	}
	if value, ok := stringOption(options, providerOptionPromptCacheKey); ok {
		payload["prompt_cache_key"] = value
	}
	for key, value := range extraBody(opts, provider) {
		payload[key] = value
	}
}

func addProviderID(item map[string]any, metadata map[string]any) {
	if id, ok := stringOption(metadata, "id"); ok {
		item["id"] = id
		return
	}
	if id, ok := stringOption(metadata, "item_id"); ok {
		item["id"] = id
	}
}

func addProviderContentID(item map[string]any, metadata map[string]any) {
	if id, ok := stringOption(metadata, "content_id"); ok {
		item["id"] = id
		return
	}
	addProviderID(item, metadata)
}
