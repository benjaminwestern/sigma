// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package mistral

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/wintermi/sigma"
	"github.com/wintermi/sigma/internal/transform"
)

const (
	providerOptionBaseURL            = "base_url"
	providerOptionBaseURLCamel       = "baseURL"
	providerOptionEndpoint           = "endpoint"
	providerOptionExtraBody          = "extra_body"
	providerOptionExtraBodyGo        = "extraBody"
	providerOptionCompletionArgs     = "completion_args"
	providerOptionCompletionArgsGo   = "completionArgs"
	providerOptionStore              = "store"
	providerOptionHandoffExecution   = "handoff_execution"
	providerOptionHandoffExecutionGo = "handoffExecution"
	providerOptionToolChoice         = "tool_choice"
	providerOptionToolChoiceGo       = "toolChoice"
)

func conversationPayload(model sigma.Model, req sigma.Request, opts sigma.Options) (map[string]any, error) {
	if err := validateCapabilities(model, req, opts); err != nil {
		return nil, err
	}

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

	inputs, err := conversationInputs(transformed)
	if err != nil {
		return nil, err
	}

	payload := map[string]any{
		"model":  string(model.ID),
		"inputs": inputs,
		"stream": true,
	}
	if transformed.SystemPrompt != "" {
		payload["instructions"] = transformed.SystemPrompt
	}
	if len(opts.Metadata) > 0 {
		payload["metadata"] = copyAnyMap(opts.Metadata)
	}
	completionArgs := completionArgs(model.Provider, opts)
	if len(completionArgs) > 0 {
		payload["completion_args"] = completionArgs
	}
	if len(transformed.Tools) > 0 {
		tools, err := conversationTools(model, transformed.Tools)
		if err != nil {
			return nil, err
		}
		payload["tools"] = tools
	}
	addProviderOptions(payload, model.Provider, opts)
	return payload, nil
}

func validateCapabilities(model sigma.Model, req sigma.Request, opts sigma.Options) error {
	if len(req.Tools) > 0 && !model.SupportsTools {
		return unsupportedError(model, "target model does not support tools")
	}
	if opts.ReasoningLevel != "" && opts.ReasoningLevel != sigma.ThinkingLevelOff {
		return unsupportedError(model, "mistral conversations does not support thinking options")
	}
	if opts.ThinkingBudgetTokens != nil {
		return unsupportedError(model, "mistral conversations does not support thinking options")
	}
	for messageIndex, message := range req.Messages {
		for _, block := range message.Content {
			switch block.Type { //nolint:exhaustive
			case sigma.ContentBlockImage:
				return unsupportedError(model, fmt.Sprintf("message %d: mistral conversations does not support image content", messageIndex))
			case sigma.ContentBlockThinking:
				return unsupportedError(model, fmt.Sprintf("message %d: mistral conversations does not support thinking content", messageIndex))
			}
		}
	}
	return nil
}

func unsupportedError(model sigma.Model, message string) error {
	return &sigma.Error{
		Code:     sigma.ErrorUnsupported,
		Message:  message,
		Provider: model.Provider,
		Model:    model.ID,
	}
}

func conversationInputs(req sigma.Request) ([]map[string]any, error) {
	inputs := make([]map[string]any, 0, len(req.Messages))
	for _, message := range req.Messages {
		converted, err := conversationInput(message)
		if err != nil {
			return nil, err
		}
		inputs = append(inputs, converted...)
	}
	return inputs, nil
}

func conversationInput(message sigma.Message) ([]map[string]any, error) {
	switch message.Role {
	case sigma.RoleUser, sigma.RoleDeveloper:
		return []map[string]any{{
			"object":  "entry",
			"type":    "message.input",
			"role":    "user",
			"content": textContent(message.Content),
		}}, nil
	case sigma.RoleAssistant:
		return assistantEntries(message.Content)
	case sigma.RoleTool:
		entry := map[string]any{
			"object":       "entry",
			"type":         "function.result",
			"tool_call_id": message.ToolCallID,
			"result":       textContent(message.Content),
		}
		if message.ToolName != "" {
			entry["name"] = message.ToolName
		}
		if message.IsError {
			entry["is_error"] = true
		}
		return []map[string]any{entry}, nil
	default:
		return nil, fmt.Errorf("mistral conversations: unsupported message role %q", message.Role)
	}
}

func assistantEntries(blocks []sigma.ContentBlock) ([]map[string]any, error) {
	var entries []map[string]any
	var text strings.Builder
	flushText := func() {
		if text.Len() == 0 {
			return
		}
		entries = append(entries, map[string]any{
			"object":  "entry",
			"type":    "message.output",
			"role":    "assistant",
			"content": text.String(),
		})
		text.Reset()
	}

	for _, block := range blocks {
		switch block.Type {
		case sigma.ContentBlockText:
			text.WriteString(block.Text)
		case sigma.ContentBlockToolCall:
			flushText()
			arguments, err := toolArgumentsString(block.ToolArguments)
			if err != nil {
				return nil, err
			}
			entries = append(entries, map[string]any{
				"object":       "entry",
				"type":         "function.call",
				"tool_call_id": block.ToolCallID,
				"name":         block.ToolName,
				"arguments":    arguments,
			})
		default:
			return nil, fmt.Errorf("mistral conversations: unsupported assistant content block %q", block.Type)
		}
	}
	flushText()
	if len(entries) == 0 {
		entries = append(entries, map[string]any{
			"object":  "entry",
			"type":    "message.output",
			"role":    "assistant",
			"content": "",
		})
	}
	return entries, nil
}

func conversationTools(model sigma.Model, tools []sigma.Tool) ([]map[string]any, error) {
	converted := make([]map[string]any, 0, len(tools))
	for _, tool := range tools {
		if tool.ProviderDefinedType != "" {
			return nil, unsupportedError(model, fmt.Sprintf("provider-defined tool %q is not supported by mistral conversations", tool.ProviderDefinedType))
		}
		parameters, err := jsonValue(tool.InputSchema)
		if err != nil {
			return nil, fmt.Errorf("mistral conversations: tool %q schema: %w", tool.Name, err)
		}
		if parameters == nil {
			parameters = map[string]any{"type": "object"}
		}
		converted = append(converted, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        tool.Name,
				"description": tool.Description,
				"parameters":  parameters,
			},
		})
	}
	return converted, nil
}

func completionArgs(provider sigma.ProviderID, opts sigma.Options) map[string]any {
	args := copyAnyMap(completionArgsOption(opts, provider))
	if args == nil {
		args = make(map[string]any)
	}
	if opts.Temperature != nil {
		args["temperature"] = *opts.Temperature
	}
	if opts.MaxTokens != nil {
		args["max_tokens"] = *opts.MaxTokens
	}
	options := providerOptions(opts, provider)
	if value, ok := options[providerOptionToolChoice]; ok {
		args["tool_choice"] = value
	} else if value, ok := options[providerOptionToolChoiceGo]; ok {
		args["tool_choice"] = value
	}
	if len(args) == 0 {
		return nil
	}
	return args
}

func addProviderOptions(payload map[string]any, provider sigma.ProviderID, opts sigma.Options) {
	options := providerOptions(opts, provider)
	if value, ok := options[providerOptionStore]; ok {
		payload["store"] = value
	}
	if value, ok := options[providerOptionHandoffExecution]; ok {
		payload["handoff_execution"] = value
	} else if value, ok := options[providerOptionHandoffExecutionGo]; ok {
		payload["handoff_execution"] = value
	}
	for key, value := range extraBody(opts, provider) {
		payload[key] = value
	}
}

func textContent(blocks []sigma.ContentBlock) string {
	var text strings.Builder
	for _, block := range blocks {
		if block.Type == sigma.ContentBlockText {
			text.WriteString(block.Text)
		}
	}
	return text.String()
}

func toolArgumentsString(arguments any) (string, error) {
	if arguments == nil {
		return "{}", nil
	}
	if text, ok := arguments.(string); ok {
		return text, nil
	}
	data, err := json.Marshal(arguments)
	if err != nil {
		return "", fmt.Errorf("mistral conversations: tool arguments: %w", err)
	}
	return string(data), nil
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

func providerOptions(opts sigma.Options, provider sigma.ProviderID) map[string]any {
	if len(opts.ProviderOptions) == 0 {
		return nil
	}
	if values := opts.ProviderOptions[provider]; len(values) > 0 {
		return values
	}
	return opts.ProviderOptions[sigma.ProviderMistral]
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

func completionArgsOption(opts sigma.Options, provider sigma.ProviderID) map[string]any {
	options := providerOptions(opts, provider)
	if value, ok := mapOption(options, providerOptionCompletionArgs); ok {
		return value
	}
	if value, ok := mapOption(options, providerOptionCompletionArgsGo); ok {
		return value
	}
	return nil
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
