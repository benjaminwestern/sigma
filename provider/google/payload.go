// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package google

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/wintermi/sigma"
	"github.com/wintermi/sigma/internal/transform"
)

const (
	providerOptionBaseURL                 = "base_url"
	providerOptionBaseURLCamel            = "baseURL"
	providerOptionEndpoint                = "endpoint"
	providerOptionExtraBody               = "extra_body"
	providerOptionExtraBodyGo             = "extraBody"
	providerOptionToolConfig              = "tool_config"
	providerOptionToolConfigGo            = "toolConfig"
	providerOptionFunctionCallingConfig   = "function_calling_config"
	providerOptionFunctionCallingConfigGo = "functionCallingConfig"
	providerOptionIncludeThoughts         = "include_thoughts"
	providerOptionIncludeThoughtsGo       = "includeThoughts"
	providerOptionSafetySettings          = "safety_settings"
	providerOptionSafetySettingsGo        = "safetySettings"
	providerOptionResponseMIMEType        = "response_mime_type"
	providerOptionResponseMIMETypeGo      = "responseMimeType"
	providerOptionResponseSchema          = "response_schema"
	providerOptionResponseSchemaGo        = "responseSchema"
	providerOptionCandidateCount          = "candidate_count"
	providerOptionCandidateCountGo        = "candidateCount"
	providerOptionTopP                    = "top_p"
	providerOptionTopPGo                  = "topP"
	providerOptionTopK                    = "top_k"
	providerOptionTopKGo                  = "topK"
)

func generativePayload(model sigma.Model, req sigma.Request, opts sigma.Options) (map[string]any, error) {
	transformed, err := transform.Transform(transform.Input{
		TargetModel: model,
		Request:     req,
		Compatibility: transform.Compatibility{
			ConvertDeveloperRole:  true,
			RequireToolResultName: true,
		},
	})
	if err != nil {
		return nil, err
	}

	contents, err := googleContents(transformed)
	if err != nil {
		return nil, err
	}

	payload := map[string]any{
		"contents": contents,
	}
	if transformed.SystemPrompt != "" {
		payload["systemInstruction"] = map[string]any{
			"parts": []map[string]any{{"text": transformed.SystemPrompt}},
		}
	}

	generationConfig, err := googleGenerationConfig(model, opts)
	if err != nil {
		return nil, err
	}
	if len(generationConfig) > 0 {
		payload["generationConfig"] = generationConfig
	}
	if len(transformed.Tools) > 0 {
		tools, err := googleTools(transformed.Tools)
		if err != nil {
			return nil, err
		}
		payload["tools"] = tools
	}
	addGoogleProviderOptions(payload, model.Provider, opts)
	return payload, nil
}

func googleContents(req sigma.Request) ([]map[string]any, error) {
	contents := make([]map[string]any, 0, len(req.Messages))
	for _, message := range req.Messages {
		converted, err := googleContent(message)
		if err != nil {
			return nil, err
		}
		contents = append(contents, converted)
	}
	return contents, nil
}

func googleContent(message sigma.Message) (map[string]any, error) {
	switch message.Role {
	case sigma.RoleUser, sigma.RoleDeveloper:
		parts, err := googleInputParts(message.Content)
		if err != nil {
			return nil, err
		}
		return map[string]any{"role": "user", "parts": parts}, nil
	case sigma.RoleAssistant:
		parts, err := googleAssistantParts(message.Content)
		if err != nil {
			return nil, err
		}
		return map[string]any{"role": "model", "parts": parts}, nil
	case sigma.RoleTool:
		part := map[string]any{
			"functionResponse": map[string]any{
				"name":     message.ToolName,
				"id":       message.ToolCallID,
				"response": googleToolResponse(message),
			},
		}
		return map[string]any{"role": "user", "parts": []map[string]any{part}}, nil
	default:
		return nil, fmt.Errorf("google generative ai: unsupported message role %q", message.Role)
	}
}

func googleInputParts(blocks []sigma.ContentBlock) ([]map[string]any, error) {
	if len(blocks) == 0 {
		return []map[string]any{{"text": ""}}, nil
	}
	parts := make([]map[string]any, 0, len(blocks))
	for _, block := range blocks {
		switch block.Type {
		case sigma.ContentBlockText:
			parts = append(parts, map[string]any{"text": block.Text})
		case sigma.ContentBlockImage:
			image, err := googleImage(block)
			if err != nil {
				return nil, err
			}
			parts = append(parts, image)
		default:
			return nil, fmt.Errorf("google generative ai: unsupported user content block %q", block.Type)
		}
	}
	return parts, nil
}

func googleAssistantParts(blocks []sigma.ContentBlock) ([]map[string]any, error) {
	parts := make([]map[string]any, 0, len(blocks))
	for _, block := range blocks {
		switch block.Type {
		case sigma.ContentBlockText:
			part := map[string]any{"text": block.Text}
			addThoughtSignature(part, block.ProviderSignature)
			parts = append(parts, part)
		case sigma.ContentBlockThinking:
			part := map[string]any{
				"text":    block.ThinkingText,
				"thought": true,
			}
			addThoughtSignature(part, block.ProviderSignature)
			parts = append(parts, part)
		case sigma.ContentBlockToolCall:
			args, err := jsonValue(block.ToolArguments)
			if err != nil {
				return nil, fmt.Errorf("google generative ai: tool %q args: %w", block.ToolName, err)
			}
			if args == nil {
				args = map[string]any{}
			}
			call := map[string]any{
				"name": block.ToolName,
				"args": args,
			}
			if block.ToolCallID != "" {
				call["id"] = block.ToolCallID
			}
			part := map[string]any{"functionCall": call}
			addThoughtSignature(part, block.ProviderSignature)
			parts = append(parts, part)
		default:
			return nil, fmt.Errorf("google generative ai: unsupported assistant content block %q", block.Type)
		}
	}
	return parts, nil
}

func googleToolResponse(message sigma.Message) map[string]any {
	response := map[string]any{"output": textContent(message.Content)}
	if message.IsError {
		response["error"] = true
	}
	return response
}

func googleImage(block sigma.ContentBlock) (map[string]any, error) {
	switch block.ImageSource {
	case "base64":
		if block.Data == "" {
			return nil, fmt.Errorf("google generative ai: image data is required")
		}
		if _, err := base64.StdEncoding.DecodeString(block.Data); err != nil {
			return nil, fmt.Errorf("google generative ai: image data must be base64: %w", err)
		}
		mimeType := block.MIMEType
		if mimeType == "" {
			mimeType = "application/octet-stream"
		}
		return map[string]any{
			"inlineData": map[string]any{
				"mimeType": mimeType,
				"data":     block.Data,
			},
		}, nil
	case "url":
		if block.URL == "" {
			return nil, fmt.Errorf("google generative ai: image URL is required")
		}
		fileData := map[string]any{
			"fileUri": block.URL,
		}
		part := map[string]any{
			"fileData": fileData,
		}
		if block.MIMEType != "" {
			fileData["mimeType"] = block.MIMEType
		}
		return part, nil
	default:
		return nil, fmt.Errorf("google generative ai: unsupported image source %q", block.ImageSource)
	}
}

func googleTools(tools []sigma.Tool) ([]map[string]any, error) {
	declarations := make([]map[string]any, 0, len(tools))
	providerTools := make([]map[string]any, 0, len(tools))
	for _, tool := range tools {
		if tool.ProviderDefinedType != "" {
			providerTools = append(providerTools, googleProviderTool(tool))
			continue
		}
		parameters, err := jsonValue(tool.InputSchema)
		if err != nil {
			return nil, fmt.Errorf("google generative ai: tool %q schema: %w", tool.Name, err)
		}
		if parameters == nil {
			parameters = map[string]any{"type": "object"}
		}
		declarations = append(declarations, map[string]any{
			"name":        tool.Name,
			"description": tool.Description,
			"parameters":  parameters,
		})
	}
	converted := make([]map[string]any, 0, 1+len(providerTools))
	if len(declarations) > 0 {
		converted = append(converted, map[string]any{"functionDeclarations": declarations})
	}
	converted = append(converted, providerTools...)
	return converted, nil
}

func googleProviderTool(tool sigma.Tool) map[string]any {
	toolType := strings.TrimPrefix(tool.ProviderDefinedType, "google.")
	key := snakeToCamel(toolType)
	converted := make(map[string]any, len(tool.ProviderDefinedOptions))
	for optionKey, value := range tool.ProviderDefinedOptions {
		converted[optionKey] = value
	}
	return map[string]any{key: converted}
}

func snakeToCamel(value string) string {
	parts := strings.Split(value, "_")
	for index := 1; index < len(parts); index++ {
		if parts[index] == "" {
			continue
		}
		parts[index] = strings.ToUpper(parts[index][:1]) + parts[index][1:]
	}
	return strings.Join(parts, "")
}

func googleGenerationConfig(model sigma.Model, opts sigma.Options) (map[string]any, error) {
	config := make(map[string]any)
	if opts.Temperature != nil {
		config["temperature"] = *opts.Temperature
	}
	if opts.MaxTokens != nil {
		config["maxOutputTokens"] = *opts.MaxTokens
	}
	addGenerationConfigProviderOptions(config, model.Provider, opts)
	thinking, err := googleThinkingConfig(model, opts)
	if err != nil {
		return nil, err
	}
	if len(thinking) > 0 {
		config["thinkingConfig"] = thinking
	}
	return config, nil
}

func googleThinkingConfig(model sigma.Model, opts sigma.Options) (map[string]any, error) {
	thinking := make(map[string]any)
	options := providerOptions(opts, model.Provider)

	if include, ok := boolOption(options, providerOptionIncludeThoughts); ok {
		thinking["includeThoughts"] = include
	} else if include, ok := boolOption(options, providerOptionIncludeThoughtsGo); ok {
		thinking["includeThoughts"] = include
	}

	if opts.GoogleOptions != nil && opts.GoogleOptions.ThinkingBudgetTokens != nil {
		thinking["thinkingBudget"] = *opts.GoogleOptions.ThinkingBudgetTokens
		if _, ok := thinking["includeThoughts"]; !ok {
			thinking["includeThoughts"] = true
		}
		return thinking, nil
	}
	if opts.ThinkingBudgetTokens != nil {
		thinking["thinkingBudget"] = *opts.ThinkingBudgetTokens
		if _, ok := thinking["includeThoughts"]; !ok {
			thinking["includeThoughts"] = true
		}
		return thinking, nil
	}
	if opts.ReasoningLevel == "" || opts.ReasoningLevel == sigma.ThinkingLevelOff {
		return thinking, nil
	}
	value, ok := model.ProviderThinkingLevel(opts.ReasoningLevel)
	if !ok {
		return nil, &sigma.Error{
			Code:     sigma.ErrorInvalidOptions,
			Message:  fmt.Sprintf("thinking level %q is not supported by model metadata", opts.ReasoningLevel),
			Provider: model.Provider,
			Model:    model.ID,
			Err:      sigma.ErrInvalidOptions,
		}
	}
	if tokens, err := strconv.Atoi(value); err == nil {
		thinking["thinkingBudget"] = tokens
	} else {
		level, ok := googleThinkingLevel(value)
		if !ok {
			return nil, &sigma.Error{
				Code:     sigma.ErrorInvalidOptions,
				Message:  fmt.Sprintf("google thinking level %q is not supported", value),
				Provider: model.Provider,
				Model:    model.ID,
				Err:      sigma.ErrInvalidOptions,
			}
		}
		thinking["thinkingLevel"] = level
	}
	if _, ok := thinking["includeThoughts"]; !ok {
		thinking["includeThoughts"] = true
	}
	return thinking, nil
}

func googleThinkingLevel(value string) (string, bool) {
	switch strings.ToUpper(strings.ReplaceAll(value, "-", "_")) {
	case "MINIMAL":
		return "MINIMAL", true
	case "LOW":
		return "LOW", true
	case "MEDIUM":
		return "MEDIUM", true
	case "HIGH":
		return "HIGH", true
	default:
		return "", false
	}
}

func addGenerationConfigProviderOptions(config map[string]any, provider sigma.ProviderID, opts sigma.Options) {
	options := providerOptions(opts, provider)
	copyOption(config, options, providerOptionResponseMIMEType, "responseMimeType")
	copyOption(config, options, providerOptionResponseMIMETypeGo, "responseMimeType")
	copyOption(config, options, providerOptionResponseSchema, "responseSchema")
	copyOption(config, options, providerOptionResponseSchemaGo, "responseSchema")
	copyOption(config, options, providerOptionCandidateCount, "candidateCount")
	copyOption(config, options, providerOptionCandidateCountGo, "candidateCount")
	copyOption(config, options, providerOptionTopP, "topP")
	copyOption(config, options, providerOptionTopPGo, "topP")
	copyOption(config, options, providerOptionTopK, "topK")
	copyOption(config, options, providerOptionTopKGo, "topK")
}

func addGoogleProviderOptions(payload map[string]any, provider sigma.ProviderID, opts sigma.Options) {
	options := providerOptions(opts, provider)
	if value, ok := options[providerOptionToolConfig]; ok {
		payload["toolConfig"] = value
	} else if value, ok := options[providerOptionToolConfigGo]; ok {
		payload["toolConfig"] = value
	} else if value, ok := options[providerOptionFunctionCallingConfig]; ok {
		payload["toolConfig"] = map[string]any{"functionCallingConfig": value}
	} else if value, ok := options[providerOptionFunctionCallingConfigGo]; ok {
		payload["toolConfig"] = map[string]any{"functionCallingConfig": value}
	}
	if value, ok := options[providerOptionSafetySettings]; ok {
		payload["safetySettings"] = value
	} else if value, ok := options[providerOptionSafetySettingsGo]; ok {
		payload["safetySettings"] = value
	}
	for key, value := range extraBody(opts, provider) {
		payload[key] = value
	}
}

func addThoughtSignature(part map[string]any, signature string) {
	if signature != "" {
		part["thoughtSignature"] = signature
	}
}

func textContent(blocks []sigma.ContentBlock) string {
	var text string
	for _, block := range blocks {
		if block.Type == sigma.ContentBlockText {
			text += block.Text
		}
	}
	return text
}

func providerOptions(opts sigma.Options, provider sigma.ProviderID) map[string]any {
	if len(opts.ProviderOptions) == 0 {
		return nil
	}
	if values := opts.ProviderOptions[provider]; len(values) > 0 {
		return values
	}
	return opts.ProviderOptions[sigma.ProviderGoogle]
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

func boolOption(options map[string]any, key string) (bool, bool) {
	value, ok := options[key]
	if !ok {
		return false, false
	}
	boolean, ok := value.(bool)
	return boolean, ok
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

func copyOption(target map[string]any, source map[string]any, sourceKey string, targetKey string) {
	if value, ok := source[sourceKey]; ok {
		target[targetKey] = value
	}
}
