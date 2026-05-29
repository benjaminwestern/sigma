// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package anthropic

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/wintermi/sigma"
	"github.com/wintermi/sigma/internal/transform"
)

const (
	providerOptionBaseURL           = "base_url"
	providerOptionBaseURLCamel      = "baseURL"
	providerOptionEndpoint          = "endpoint"
	providerOptionVersion           = "anthropic_version"
	providerOptionVersionGo         = "anthropicVersion"
	providerOptionBeta              = "anthropic_beta"
	providerOptionBetaGo            = "anthropicBeta"
	providerOptionExtraBody         = "extra_body"
	providerOptionExtraBodyGo       = "extraBody"
	providerOptionSessionHeader     = "session_id_header"
	providerOptionSessionHeaderGo   = "sessionIDHeader"
	providerOptionToolChoice        = "tool_choice"
	providerOptionToolChoiceGo      = "toolChoice"
	providerOptionThinkingDisplay   = "thinking_display"
	providerOptionThinkingDisplayGo = "thinkingDisplay"
)

func messagesPayload(model sigma.Model, req sigma.Request, opts sigma.Options, compat messagesCompat) (map[string]any, error) {
	transformed, err := transform.Transform(transform.Input{
		TargetModel: model,
		Request:     req,
		Compatibility: transform.Compatibility{
			ConvertDeveloperRole: true,
		},
	})
	if err != nil {
		return nil, err
	}

	messages, err := anthropicMessages(transformed, opts.CacheRetention, compat)
	if err != nil {
		return nil, err
	}

	maxTokens := 1024
	if opts.MaxTokens != nil {
		maxTokens = *opts.MaxTokens
	}
	payload := map[string]any{
		"model":      string(model.ID),
		"messages":   messages,
		"max_tokens": maxTokens,
		"stream":     true,
	}
	if transformed.SystemPrompt != "" {
		payload["system"] = anthropicSystem(transformed.SystemPrompt, opts.CacheRetention, compat)
	}
	if opts.Temperature != nil {
		payload["temperature"] = *opts.Temperature
	}
	if len(opts.Metadata) > 0 {
		payload["metadata"] = copyAnyMap(opts.Metadata)
	}
	addThinking(payload, model, opts, compat)
	if len(transformed.Tools) > 0 {
		tools, err := anthropicTools(transformed.Tools, opts.CacheRetention, compat)
		if err != nil {
			return nil, err
		}
		payload["tools"] = tools
	}
	addProviderOptions(payload, model.Provider, opts)
	return payload, nil
}

func anthropicSystem(prompt string, retention sigma.CacheRetention, compat messagesCompat) any {
	if cacheControl(retention, compat) == nil {
		return prompt
	}
	block := map[string]any{
		"type": "text", //nolint:goconst
		"text": prompt,
	}
	addCacheControl(block, retention, compat)
	return []map[string]any{block}
}

func anthropicMessages(req sigma.Request, retention sigma.CacheRetention, compat messagesCompat) ([]map[string]any, error) {
	messages := make([]map[string]any, 0, len(req.Messages))
	for _, message := range req.Messages {
		converted, err := anthropicMessage(message, retention, compat)
		if err != nil {
			return nil, err
		}
		messages = append(messages, converted)
	}
	return messages, nil
}

func anthropicMessage(message sigma.Message, retention sigma.CacheRetention, compat messagesCompat) (map[string]any, error) {
	switch message.Role {
	case sigma.RoleUser, sigma.RoleDeveloper:
		content, err := anthropicInputContent(message.Content, false)
		if err != nil {
			return nil, err
		}
		addCacheControlToLast(content, retention, compat)
		return map[string]any{"role": "user", "content": content}, nil
	case sigma.RoleAssistant:
		content, err := anthropicAssistantContent(message.Content)
		if err != nil {
			return nil, err
		}
		return map[string]any{"role": "assistant", "content": content}, nil
	case sigma.RoleTool:
		content, err := anthropicToolResultContent(message)
		if err != nil {
			return nil, err
		}
		block := map[string]any{
			"type":        "tool_result",
			"tool_use_id": message.ToolCallID,
			"content":     content,
		}
		if message.IsError {
			block["is_error"] = true
		}
		addCacheControl(block, retention, compat)
		return map[string]any{"role": "user", "content": []map[string]any{block}}, nil
	default:
		return nil, fmt.Errorf("anthropic messages: unsupported message role %q", message.Role)
	}
}

func anthropicInputContent(blocks []sigma.ContentBlock, toolResult bool) ([]map[string]any, error) {
	if len(blocks) == 0 {
		return []map[string]any{{"type": "text", "text": ""}}, nil
	}
	content := make([]map[string]any, 0, len(blocks))
	for _, block := range blocks {
		switch block.Type {
		case sigma.ContentBlockText:
			content = append(content, map[string]any{
				"type": "text",
				"text": block.Text,
			})
		case sigma.ContentBlockImage:
			image, err := anthropicImage(block)
			if err != nil {
				return nil, err
			}
			content = append(content, image)
		default:
			if toolResult {
				return nil, fmt.Errorf("anthropic messages: unsupported tool-result content block %q", block.Type)
			}
			return nil, fmt.Errorf("anthropic messages: unsupported user content block %q", block.Type)
		}
	}
	return content, nil
}

func anthropicToolResultContent(message sigma.Message) (any, error) {
	if len(message.Content) == 1 && message.Content[0].Type == sigma.ContentBlockText {
		return message.Content[0].Text, nil
	}
	return anthropicInputContent(message.Content, true)
}

func anthropicAssistantContent(blocks []sigma.ContentBlock) ([]map[string]any, error) {
	content := make([]map[string]any, 0, len(blocks))
	for _, block := range blocks {
		switch block.Type {
		case sigma.ContentBlockText:
			content = append(content, map[string]any{
				"type": "text",
				"text": block.Text,
			})
		case sigma.ContentBlockThinking:
			if block.Redacted {
				data := block.ProviderSignature
				if data == "" {
					data = block.Signature
				}
				content = append(content, map[string]any{
					"type": "redacted_thinking",
					"data": data,
				})
				continue
			}
			thinking := map[string]any{
				"type":     "thinking",
				"thinking": block.ThinkingText,
			}
			if block.Signature != "" {
				thinking["signature"] = block.Signature
			}
			content = append(content, thinking)
		case sigma.ContentBlockToolCall:
			input, err := jsonValue(block.ToolArguments)
			if err != nil {
				return nil, fmt.Errorf("anthropic messages: tool %q input: %w", block.ToolName, err)
			}
			if input == nil {
				input = map[string]any{}
			}
			content = append(content, map[string]any{
				"type":  "tool_use",
				"id":    block.ToolCallID,
				"name":  block.ToolName,
				"input": input,
			})
		default:
			return nil, fmt.Errorf("anthropic messages: unsupported assistant content block %q", block.Type)
		}
	}
	return content, nil
}

func anthropicImage(block sigma.ContentBlock) (map[string]any, error) {
	switch block.ImageSource {
	case "base64":
		if block.Data == "" {
			return nil, fmt.Errorf("anthropic messages: image data is required")
		}
		if _, err := base64.StdEncoding.DecodeString(block.Data); err != nil {
			return nil, fmt.Errorf("anthropic messages: image data must be base64: %w", err)
		}
		mimeType := block.MIMEType
		if mimeType == "" {
			mimeType = "application/octet-stream"
		}
		return map[string]any{
			"type": "image",
			"source": map[string]any{
				"type":       "base64",
				"media_type": mimeType,
				"data":       block.Data,
			},
		}, nil
	case "url":
		if block.URL == "" {
			return nil, fmt.Errorf("anthropic messages: image URL is required")
		}
		return map[string]any{
			"type": "image",
			"source": map[string]any{
				"type": "url",
				"url":  block.URL,
			},
		}, nil
	default:
		return nil, fmt.Errorf("anthropic messages: unsupported image source %q", block.ImageSource)
	}
}

func anthropicTools(tools []sigma.Tool, retention sigma.CacheRetention, compat messagesCompat) ([]map[string]any, error) {
	converted := make([]map[string]any, 0, len(tools))
	for _, tool := range tools {
		if tool.ProviderDefinedType != "" {
			convertedTool := map[string]any{
				"type": tool.ProviderDefinedType,
				"name": tool.Name,
			}
			for key, value := range tool.ProviderDefinedOptions {
				convertedTool[key] = value
			}
			if compat.cacheControlOnTools {
				addCacheControl(convertedTool, retention, compat)
			}
			converted = append(converted, convertedTool)
			continue
		}
		inputSchema, err := jsonValue(tool.InputSchema)
		if err != nil {
			return nil, fmt.Errorf("anthropic messages: tool %q schema: %w", tool.Name, err)
		}
		if inputSchema == nil {
			inputSchema = map[string]any{"type": "object"}
		}
		convertedTool := map[string]any{
			"name":         tool.Name,
			"description":  tool.Description,
			"input_schema": inputSchema,
		}
		if compat.cacheControlOnTools {
			addCacheControl(convertedTool, retention, compat)
		}
		converted = append(converted, convertedTool)
	}
	return converted, nil
}

func addThinking(payload map[string]any, model sigma.Model, opts sigma.Options, compat messagesCompat) {
	budget := thinkingBudget(model, opts, compat)
	if budget <= 0 {
		return
	}
	thinking := map[string]any{
		"type":          "enabled",
		"budget_tokens": budget,
	}
	options := providerOptions(opts, model.Provider)
	if display, ok := stringOption(options, providerOptionThinkingDisplay); ok {
		thinking["display"] = display
	} else if display, ok := stringOption(options, providerOptionThinkingDisplayGo); ok {
		thinking["display"] = display
	}
	payload["thinking"] = thinking
}

func thinkingBudget(model sigma.Model, opts sigma.Options, compat messagesCompat) int {
	if opts.AnthropicOptions != nil && opts.AnthropicOptions.ThinkingBudgetTokens != nil {
		return *opts.AnthropicOptions.ThinkingBudgetTokens
	}
	if opts.ThinkingBudgetTokens != nil {
		return *opts.ThinkingBudgetTokens
	}
	if opts.ReasoningLevel == "" || opts.ReasoningLevel == sigma.ThinkingLevelOff {
		return 0
	}
	if value, ok := model.ProviderThinkingLevel(opts.ReasoningLevel); ok {
		if tokens, err := strconv.Atoi(value); err == nil {
			return tokens
		}
	}
	if !compat.adaptiveThinking {
		return 0
	}
	switch opts.ReasoningLevel {
	case sigma.ThinkingLevelMinimal:
		return 1024
	case sigma.ThinkingLevelLow:
		return 2048
	case sigma.ThinkingLevelMedium:
		return 4096
	case sigma.ThinkingLevelHigh:
		return 8192
	case sigma.ThinkingLevelXHigh:
		return 16384
	default:
		return 0
	}
}

func addProviderOptions(payload map[string]any, provider sigma.ProviderID, opts sigma.Options) {
	options := providerOptions(opts, provider)
	if value, ok := options[providerOptionToolChoice]; ok {
		payload["tool_choice"] = value
	} else if value, ok := options[providerOptionToolChoiceGo]; ok {
		payload["tool_choice"] = value
	}
	for key, value := range extraBody(opts, provider) {
		payload[key] = value
	}
}

func addCacheControlToLast(content []map[string]any, retention sigma.CacheRetention, compat messagesCompat) {
	for i := len(content) - 1; i >= 0; i-- {
		if content[i]["type"] == "thinking" || content[i]["type"] == "redacted_thinking" {
			continue
		}
		addCacheControl(content[i], retention, compat)
		return
	}
}

func addCacheControl(block map[string]any, retention sigma.CacheRetention, compat messagesCompat) {
	cacheControl := cacheControl(retention, compat)
	if cacheControl == nil {
		return
	}
	block["cache_control"] = cacheControl
}

func cacheControl(retention sigma.CacheRetention, compat messagesCompat) map[string]any {
	if !retention.CacheEnabled() {
		return nil
	}
	control := map[string]any{"type": "ephemeral"}
	if retention.CacheLongLived() {
		if !compat.longCacheRetention {
			return nil
		}
		control["ttl"] = "1h"
	}
	return control
}

func providerOptions(opts sigma.Options, provider sigma.ProviderID) map[string]any {
	if len(opts.ProviderOptions) == 0 {
		return nil
	}
	if values := opts.ProviderOptions[provider]; len(values) > 0 {
		return values
	}
	return opts.ProviderOptions[sigma.ProviderAnthropic]
}

func extraBody(opts sigma.Options, provider sigma.ProviderID) map[string]any {
	options := providerOptions(opts, provider)
	if value, ok := mapOption(options, providerOptionExtraBody); ok {
		return value
	}
	if value, ok := mapOption(options, providerOptionExtraBodyGo); ok {
		return value
	}
	return nil
}

func jsonValue(value any) (any, error) {
	switch v := value.(type) {
	case nil:
		return nil, nil
	case json.RawMessage:
		var out any
		if err := json.Unmarshal(v, &out); err != nil {
			return nil, err
		}
		return out, nil
	case []byte:
		var out any
		if err := json.Unmarshal(v, &out); err != nil {
			return nil, err
		}
		return out, nil
	default:
		data, err := json.Marshal(value)
		if err != nil {
			return nil, err
		}
		var out any
		if err := json.Unmarshal(data, &out); err != nil {
			return nil, err
		}
		return out, nil
	}
}

func stringOption(options map[string]any, key string) (string, bool) {
	value, ok := options[key]
	if !ok {
		return "", false
	}
	text, ok := value.(string)
	return text, ok && text != ""
}

func mapOption(options map[string]any, key string) (map[string]any, bool) {
	value, ok := options[key]
	if !ok {
		return nil, false
	}
	values, ok := value.(map[string]any)
	return values, ok
}

func copyAnyMap(values map[string]any) map[string]any {
	if len(values) == 0 {
		return nil
	}
	copied := make(map[string]any, len(values))
	for key, value := range values {
		copied[key] = value
	}
	return copied
}
