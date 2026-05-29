// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package bedrock

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"mime"
	"strings"

	"github.com/wintermi/sigma"
	"github.com/wintermi/sigma/internal/transform"
)

const (
	providerOptionRegion                       = "region"
	providerOptionModelID                      = "model_id"
	providerOptionModelIDGo                    = "modelID"
	providerOptionInferenceProfileARN          = "inference_profile_arn"
	providerOptionInferenceProfileARNGo        = "inferenceProfileARN"
	providerOptionEndpoint                     = "endpoint"
	providerOptionCredentialSource             = "credential_source"
	providerOptionCredentialSourceGo           = "credentialSource"
	providerOptionAdditionalModelRequestFields = "additional_model_request_fields"
	providerOptionAdditionalModelRequestGo     = "additionalModelRequestFields"
	providerOptionStopSequences                = "stop_sequences"
	providerOptionStopSequencesGo              = "stopSequences"
	providerOptionTopP                         = "top_p"
	providerOptionTopPGo                       = "topP"
)

const (
	converseBlockText       = "text"
	converseBlockImage      = "image"
	converseBlockToolUse    = "tool_use"
	converseBlockToolResult = "tool_result"
	converseBlockReasoning  = "reasoning"
	converseBlockCachePoint = "cache_point"
)

// ConverseStreamClient is the narrow fakeable Bedrock runtime seam.
type ConverseStreamClient interface {
	ConverseStream(context.Context, ConverseRequest) (ConverseStream, error)
}

// ConverseRequest is the provider-owned request shape sent through the client seam.
type ConverseRequest struct {
	ModelID                           string                   `json:"modelId"`
	System                            []ConverseContentBlock   `json:"system,omitempty"`
	Messages                          []ConverseMessage        `json:"messages,omitempty"`
	InferenceConfig                   *ConverseInferenceConfig `json:"inferenceConfig,omitempty"`
	Tools                             []ConverseTool           `json:"tools,omitempty"`
	AdditionalModelRequestFields      map[string]any           `json:"additionalModelRequestFields,omitempty"`
	AdditionalModelResponseFieldPaths []string                 `json:"additionalModelResponseFieldPaths,omitempty"`
	RequestMetadata                   map[string]string        `json:"requestMetadata,omitempty"`
}

// ConverseMessage is a Bedrock Converse message without AWS SDK types.
type ConverseMessage struct {
	Role    string                 `json:"role"`
	Content []ConverseContentBlock `json:"content"`
}

// ConverseContentBlock is a provider-owned Converse content block.
type ConverseContentBlock struct {
	Type       string                   `json:"type"`
	Text       string                   `json:"text,omitempty"`
	Image      *ConverseImageBlock      `json:"image,omitempty"`
	ToolUse    *ConverseToolUseBlock    `json:"toolUse,omitempty"`
	ToolResult *ConverseToolResultBlock `json:"toolResult,omitempty"`
	Reasoning  *ConverseReasoningBlock  `json:"reasoning,omitempty"`
}

// ConverseImageBlock carries inline image bytes as base64 text.
type ConverseImageBlock struct {
	Format string `json:"format"`
	Data   string `json:"data"`
}

// ConverseToolUseBlock carries an assistant tool request.
type ConverseToolUseBlock struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	Input any    `json:"input,omitempty"`
}

// ConverseToolResultBlock carries a user tool result.
type ConverseToolResultBlock struct {
	ToolUseID string `json:"toolUseId"`
	Text      string `json:"text"`
	Status    string `json:"status,omitempty"`
}

// ConverseReasoningBlock carries replayed reasoning content.
type ConverseReasoningBlock struct {
	Text              string `json:"text,omitempty"`
	Signature         string `json:"signature,omitempty"`
	ProviderSignature string `json:"providerSignature,omitempty"`
	Redacted          bool   `json:"redacted,omitempty"`
}

// ConverseInferenceConfig carries portable Converse inference parameters.
type ConverseInferenceConfig struct {
	MaxTokens     *int     `json:"maxTokens,omitempty"`
	Temperature   *float64 `json:"temperature,omitempty"`
	StopSequences []string `json:"stopSequences,omitempty"`
}

// ConverseTool carries a function tool spec.
type ConverseTool struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	InputSchema any    `json:"inputSchema,omitempty"`
}

func conversePayload(model sigma.Model, req sigma.Request, opts sigma.Options, config Config) (ConverseRequest, error) {
	if err := validateCapabilities(model, req, opts); err != nil {
		return ConverseRequest{}, err
	}

	transformed, err := transform.Transform(transform.Input{
		TargetModel: model,
		Request:     req,
		Compatibility: transform.Compatibility{
			ConvertDeveloperRole: true,
		},
	})
	if err != nil {
		return ConverseRequest{}, err
	}

	messages, err := converseMessages(transformed)
	if err != nil {
		return ConverseRequest{}, err
	}
	payload := ConverseRequest{
		ModelID:                           bedrockModelID(config, model),
		Messages:                          messages,
		InferenceConfig:                   inferenceConfig(opts, model.Provider),
		AdditionalModelRequestFields:      additionalModelRequestFields(opts, model.Provider),
		RequestMetadata:                   requestMetadata(opts.Metadata),
		AdditionalModelResponseFieldPaths: responseFieldPaths(opts, model.Provider),
	}
	if payload.ModelID == "" {
		return ConverseRequest{}, unsupportedError(model, "bedrock converse stream: model id is required")
	}
	if transformed.SystemPrompt != "" {
		payload.System = append(payload.System, ConverseContentBlock{Type: converseBlockText, Text: transformed.SystemPrompt})
	}
	if opts.CacheRetention != "" && opts.CacheRetention != sigma.CacheRetentionNone {
		if len(payload.System) > 0 {
			payload.System = append(payload.System, ConverseContentBlock{Type: converseBlockCachePoint})
		} else if len(payload.Messages) > 0 {
			last := len(payload.Messages) - 1
			payload.Messages[last].Content = append(payload.Messages[last].Content, ConverseContentBlock{Type: converseBlockCachePoint})
		}
	}
	if opts.ThinkingBudgetTokens != nil {
		if payload.AdditionalModelRequestFields == nil {
			payload.AdditionalModelRequestFields = make(map[string]any)
		}
		payload.AdditionalModelRequestFields["thinking"] = map[string]any{
			"type":          "enabled",
			"budget_tokens": *opts.ThinkingBudgetTokens,
		}
	}
	if len(transformed.Tools) > 0 {
		tools, err := converseTools(model, transformed.Tools)
		if err != nil {
			return ConverseRequest{}, err
		}
		payload.Tools = tools
	}
	return payload, nil
}

func validateCapabilities(model sigma.Model, req sigma.Request, opts sigma.Options) error {
	if len(req.Tools) > 0 && !model.SupportsTools {
		return unsupportedError(model, "target model does not support tools")
	}
	if opts.ThinkingBudgetTokens != nil && !model.SupportsThinking {
		return unsupportedError(model, "target model does not support thinking options")
	}
	if opts.ReasoningLevel != "" && opts.ReasoningLevel != sigma.ThinkingLevelOff && !model.SupportsThinking {
		return unsupportedError(model, "target model does not support thinking options")
	}
	for messageIndex, message := range req.Messages {
		for _, block := range message.Content {
			if block.Type == sigma.ContentBlockImage && !supportsInput(model, sigma.ContentBlockImage) {
				return unsupportedError(model, fmt.Sprintf("message %d: target model does not declare image input support", messageIndex))
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

func converseMessages(req sigma.Request) ([]ConverseMessage, error) {
	messages := make([]ConverseMessage, 0, len(req.Messages))
	for _, message := range req.Messages {
		converted, err := converseMessage(message)
		if err != nil {
			return nil, err
		}
		messages = append(messages, converted)
	}
	return messages, nil
}

func converseMessage(message sigma.Message) (ConverseMessage, error) {
	switch message.Role {
	case sigma.RoleUser, sigma.RoleDeveloper:
		content, err := converseInputContent(message.Content)
		return ConverseMessage{Role: "user", Content: content}, err
	case sigma.RoleAssistant:
		content, err := converseAssistantContent(message.Content)
		return ConverseMessage{Role: "assistant", Content: content}, err
	case sigma.RoleTool:
		return ConverseMessage{
			Role: "user",
			Content: []ConverseContentBlock{{
				Type: converseBlockToolResult,
				ToolResult: &ConverseToolResultBlock{
					ToolUseID: message.ToolCallID,
					Text:      textContent(message.Content),
					Status:    toolResultStatus(message.IsError),
				},
			}},
		}, nil
	default:
		return ConverseMessage{}, fmt.Errorf("bedrock converse stream: unsupported message role %q", message.Role)
	}
}

func converseInputContent(blocks []sigma.ContentBlock) ([]ConverseContentBlock, error) {
	if len(blocks) == 0 {
		return []ConverseContentBlock{{Type: converseBlockText, Text: ""}}, nil
	}
	content := make([]ConverseContentBlock, 0, len(blocks))
	for _, block := range blocks {
		switch block.Type {
		case sigma.ContentBlockText:
			content = append(content, ConverseContentBlock{Type: converseBlockText, Text: block.Text})
		case sigma.ContentBlockImage:
			image, err := converseImage(block)
			if err != nil {
				return nil, err
			}
			content = append(content, ConverseContentBlock{Type: converseBlockImage, Image: image})
		default:
			return nil, fmt.Errorf("bedrock converse stream: unsupported user content block %q", block.Type)
		}
	}
	return content, nil
}

func converseAssistantContent(blocks []sigma.ContentBlock) ([]ConverseContentBlock, error) {
	content := make([]ConverseContentBlock, 0, len(blocks))
	for _, block := range blocks {
		switch block.Type {
		case sigma.ContentBlockText:
			content = append(content, ConverseContentBlock{Type: converseBlockText, Text: block.Text})
		case sigma.ContentBlockThinking:
			reasoning := &ConverseReasoningBlock{
				Text:              block.ThinkingText,
				Signature:         block.Signature,
				ProviderSignature: firstNonEmpty(block.ProviderSignature, block.Signature),
				Redacted:          block.Redacted,
			}
			content = append(content, ConverseContentBlock{Type: converseBlockReasoning, Reasoning: reasoning})
		case sigma.ContentBlockToolCall:
			input, err := jsonValue(block.ToolArguments)
			if err != nil {
				return nil, fmt.Errorf("bedrock converse stream: tool %q input: %w", block.ToolName, err)
			}
			if input == nil {
				input = map[string]any{}
			}
			content = append(content, ConverseContentBlock{
				Type: converseBlockToolUse,
				ToolUse: &ConverseToolUseBlock{
					ID:    block.ToolCallID,
					Name:  block.ToolName,
					Input: input,
				},
			})
		default:
			return nil, fmt.Errorf("bedrock converse stream: unsupported assistant content block %q", block.Type)
		}
	}
	if len(content) == 0 {
		content = append(content, ConverseContentBlock{Type: converseBlockText, Text: ""})
	}
	return content, nil
}

func converseImage(block sigma.ContentBlock) (*ConverseImageBlock, error) {
	if block.ImageSource != "base64" {
		return nil, fmt.Errorf("bedrock converse stream: unsupported image source %q", block.ImageSource)
	}
	if block.Data == "" {
		return nil, fmt.Errorf("bedrock converse stream: image data is required")
	}
	if _, err := base64.StdEncoding.DecodeString(block.Data); err != nil {
		return nil, fmt.Errorf("bedrock converse stream: image data must be base64: %w", err)
	}
	format := imageFormat(block.MIMEType)
	if format == "" {
		return nil, fmt.Errorf("bedrock converse stream: unsupported image MIME type %q", block.MIMEType)
	}
	return &ConverseImageBlock{Format: format, Data: block.Data}, nil
}

func converseTools(model sigma.Model, tools []sigma.Tool) ([]ConverseTool, error) {
	converted := make([]ConverseTool, 0, len(tools))
	for _, tool := range tools {
		if tool.ProviderDefinedType != "" {
			return nil, unsupportedError(model, fmt.Sprintf("provider-defined tool %q is not supported by bedrock converse stream", tool.ProviderDefinedType))
		}
		schema, err := jsonValue(tool.InputSchema)
		if err != nil {
			return nil, fmt.Errorf("bedrock converse stream: tool %q schema: %w", tool.Name, err)
		}
		if schema == nil {
			schema = map[string]any{"type": "object"}
		}
		converted = append(converted, ConverseTool{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: schema,
		})
	}
	return converted, nil
}

func bedrockModelID(config Config, model sigma.Model) string {
	if config.InferenceProfileARN != "" {
		return config.InferenceProfileARN
	}
	if config.ModelID != "" {
		return config.ModelID
	}
	return string(model.ID)
}

func inferenceConfig(opts sigma.Options, provider sigma.ProviderID) *ConverseInferenceConfig {
	config := &ConverseInferenceConfig{}
	if opts.MaxTokens != nil {
		config.MaxTokens = opts.MaxTokens
	}
	if opts.Temperature != nil {
		config.Temperature = opts.Temperature
	}
	options := providerOptions(opts, provider)
	if values, ok := stringSliceOption(options, providerOptionStopSequences); ok {
		config.StopSequences = values
	} else if values, ok := stringSliceOption(options, providerOptionStopSequencesGo); ok {
		config.StopSequences = values
	}
	if config.MaxTokens == nil && config.Temperature == nil && len(config.StopSequences) == 0 {
		return nil
	}
	return config
}

func additionalModelRequestFields(opts sigma.Options, provider sigma.ProviderID) map[string]any {
	options := providerOptions(opts, provider)
	fields := mapOption(options, providerOptionAdditionalModelRequestFields)
	if len(fields) == 0 {
		fields = mapOption(options, providerOptionAdditionalModelRequestGo)
	}
	if len(fields) == 0 {
		return nil
	}
	return copyAnyMap(fields)
}

func responseFieldPaths(opts sigma.Options, provider sigma.ProviderID) []string {
	options := providerOptions(opts, provider)
	return stringSlice(options["additional_model_response_field_paths"])
}

func requestMetadata(metadata map[string]any) map[string]string {
	if len(metadata) == 0 {
		return nil
	}
	converted := make(map[string]string, len(metadata))
	for key, value := range metadata {
		if text, ok := value.(string); ok {
			converted[key] = text
		}
	}
	if len(converted) == 0 {
		return nil
	}
	return converted
}

func providerOptions(opts sigma.Options, provider sigma.ProviderID) map[string]any {
	if len(opts.ProviderOptions) == 0 {
		return nil
	}
	if values := opts.ProviderOptions[provider]; len(values) > 0 {
		return values
	}
	return opts.ProviderOptions[sigma.ProviderAmazonBedrock]
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

func toolResultStatus(isError bool) string {
	if isError {
		return "error"
	}
	return "success"
}

func imageFormat(mimeType string) string {
	mediaType, _, err := mime.ParseMediaType(mimeType)
	if err != nil {
		mediaType = mimeType
	}
	switch strings.ToLower(mediaType) {
	case "image/png":
		return "png"
	case "image/jpeg", "image/jpg":
		return "jpeg"
	case "image/gif":
		return "gif"
	case "image/webp":
		return "webp"
	default:
		return ""
	}
}

func supportsInput(model sigma.Model, blockType sigma.ContentBlockType) bool {
	if len(model.SupportedInputs) == 0 {
		return true
	}
	for _, supported := range model.SupportedInputs {
		if supported == blockType {
			return true
		}
	}
	return false
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

func mapOption(options map[string]any, key string) map[string]any {
	if len(options) == 0 {
		return nil
	}
	value, _ := options[key].(map[string]any)
	return value
}

func stringOption(options map[string]any, key string) (string, bool) {
	if len(options) == 0 {
		return "", false
	}
	value, ok := options[key].(string)
	return value, ok && value != ""
}

func stringSliceOption(options map[string]any, key string) ([]string, bool) {
	if len(options) == 0 {
		return nil, false
	}
	values := stringSlice(options[key])
	return values, len(values) > 0
}

func stringSlice(value any) []string {
	switch v := value.(type) {
	case []string:
		return append([]string(nil), v...)
	case []any:
		values := make([]string, 0, len(v))
		for _, item := range v {
			if text, ok := item.(string); ok {
				values = append(values, text)
			}
		}
		return values
	default:
		return nil
	}
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
