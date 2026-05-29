// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package transform

import (
	"encoding/json"
	"fmt"

	"github.com/wintermi/sigma"
)

const (
	defaultThinkingStartDelimiter = "<thinking>"
	defaultThinkingEndDelimiter   = "</thinking>"
)

// Input describes a provider-neutral request transformation.
type Input struct {
	TargetModel   sigma.Model
	Request       sigma.Request
	Compatibility Compatibility
	Policy        Policy
}

// Compatibility describes target-provider constraints that are not represented
// by generic model metadata.
type Compatibility struct {
	ThinkingAsText                  bool
	AssistantAfterToolResultRepair  bool
	AssistantAfterToolResultMessage sigma.Message
	RequireToolResultName           bool
	ConvertDeveloperRole            bool
}

// Policy controls lossless request normalization choices.
type Policy struct {
	ThinkingStartDelimiter string
	ThinkingEndDelimiter   string
	AllowUnsupportedImages bool
}

// Transform returns a transformed copy of input.Request for input.TargetModel.
// It never mutates the caller's request, message content, tool schemas, or
// provider metadata.
func Transform(input Input) (sigma.Request, error) {
	if err := sigma.ValidateModelRef(sigma.ModelRef{Provider: input.TargetModel.Provider, ID: input.TargetModel.ID}); err != nil {
		return sigma.Request{}, err
	}

	policy := input.Policy.withDefaults()
	output := sigma.Request{
		SystemPrompt: input.Request.SystemPrompt,
		Tools:        cloneTools(input.Request.Tools),
	}

	toolNamesByID := make(map[string]string)
	for index, message := range input.Request.Messages {
		transformed, err := transformMessage(message, messageContext{
			target:       input.TargetModel,
			compat:       input.Compatibility,
			policy:       policy,
			toolNames:    toolNamesByID,
			messageIndex: index,
		})
		if err != nil {
			return sigma.Request{}, err
		}

		if input.Compatibility.AssistantAfterToolResultRepair &&
			len(output.Messages) > 0 &&
			output.Messages[len(output.Messages)-1].Role == sigma.RoleTool &&
			transformed.Role == sigma.RoleAssistant {
			output.Messages = append(output.Messages, repairMessage(input.Compatibility.AssistantAfterToolResultMessage))
		}

		output.Messages = append(output.Messages, transformed)
		recordToolCalls(toolNamesByID, transformed)
	}

	return output, nil
}

type messageContext struct {
	target       sigma.Model
	compat       Compatibility
	policy       Policy
	toolNames    map[string]string
	messageIndex int
}

func transformMessage(message sigma.Message, ctx messageContext) (sigma.Message, error) {
	transformed := cloneMessage(message)
	if ctx.compat.ConvertDeveloperRole && transformed.Role == sigma.RoleDeveloper {
		transformed.Role = sigma.RoleUser
	}

	for index, block := range message.Content {
		if err := validateImageSupport(block, ctx); err != nil {
			return sigma.Message{}, err
		}
		if transformed.Role == sigma.RoleAssistant && block.Type == sigma.ContentBlockThinking && shouldConvertThinking(message, ctx) {
			transformed.Content[index] = sigma.Text(wrapThinking(block.ThinkingText, ctx.policy))
			continue
		}
		transformed.Content[index] = cloneContentBlock(block)
	}

	if transformed.Role == sigma.RoleTool && ctx.compat.RequireToolResultName && transformed.ToolName == "" {
		toolName, ok := ctx.toolNames[transformed.ToolCallID]
		if !ok {
			return sigma.Message{}, transformError(ctx, "tool result requires a tool name but no matching assistant tool call was found")
		}
		transformed.ToolName = toolName
	}

	return transformed, nil
}

func shouldConvertThinking(message sigma.Message, ctx messageContext) bool {
	if ctx.compat.ThinkingAsText {
		return true
	}
	if !ctx.target.SupportsReasoning() {
		return true
	}
	if message.Provider != "" && message.Provider != ctx.target.Provider {
		return true
	}
	return message.API != "" && ctx.target.API != "" && message.API != ctx.target.API
}

func validateImageSupport(block sigma.ContentBlock, ctx messageContext) error {
	if block.Type != sigma.ContentBlockImage || ctx.policy.AllowUnsupportedImages || ctx.target.SupportsImages() {
		return nil
	}
	return transformError(ctx, "target model does not support image content")
}

func wrapThinking(text string, policy Policy) string {
	return policy.ThinkingStartDelimiter + "\n" + text + "\n" + policy.ThinkingEndDelimiter
}

func recordToolCalls(toolNamesByID map[string]string, message sigma.Message) {
	if message.Role != sigma.RoleAssistant {
		return
	}
	for _, block := range message.Content {
		if block.Type == sigma.ContentBlockToolCall && block.ToolCallID != "" && block.ToolName != "" {
			toolNamesByID[block.ToolCallID] = block.ToolName
		}
	}
}

func repairMessage(message sigma.Message) sigma.Message {
	if message.Role != "" {
		return cloneMessage(message)
	}
	return sigma.UserText("Continue.")
}

func transformError(ctx messageContext, message string) error {
	return &sigma.Error{
		Code:     sigma.ErrorUnsupported,
		Message:  fmt.Sprintf("transform message %d: %s", ctx.messageIndex, message),
		Provider: ctx.target.Provider,
		Model:    ctx.target.ID,
	}
}

func (policy Policy) withDefaults() Policy {
	if policy.ThinkingStartDelimiter == "" {
		policy.ThinkingStartDelimiter = defaultThinkingStartDelimiter
	}
	if policy.ThinkingEndDelimiter == "" {
		policy.ThinkingEndDelimiter = defaultThinkingEndDelimiter
	}
	return policy
}

func cloneMessage(message sigma.Message) sigma.Message {
	message.Content = cloneContentBlocks(message.Content)
	return message
}

func cloneContentBlocks(blocks []sigma.ContentBlock) []sigma.ContentBlock {
	if len(blocks) == 0 {
		return nil
	}
	cloned := make([]sigma.ContentBlock, len(blocks))
	for index, block := range blocks {
		cloned[index] = cloneContentBlock(block)
	}
	return cloned
}

func cloneContentBlock(block sigma.ContentBlock) sigma.ContentBlock {
	block.ToolArguments = cloneAny(block.ToolArguments)
	block.ProviderMetadata = cloneStringAnyMap(block.ProviderMetadata)
	return block
}

func cloneTool(tool sigma.Tool) sigma.Tool {
	tool.InputSchema = cloneAny(tool.InputSchema)
	tool.ProviderDefinedOptions = cloneProviderDefinedOptions(tool.ProviderDefinedOptions)
	tool.ProviderMetadata = cloneStringAnyMap(tool.ProviderMetadata)
	return tool
}

func cloneTools(tools []sigma.Tool) []sigma.Tool {
	if len(tools) == 0 {
		return nil
	}
	cloned := make([]sigma.Tool, len(tools))
	for index, tool := range tools {
		cloned[index] = cloneTool(tool)
	}
	return cloned
}

func cloneStringAnyMap(values map[string]any) map[string]any {
	if len(values) == 0 {
		return nil
	}
	cloned := make(map[string]any, len(values))
	for key, value := range values {
		cloned[key] = cloneAny(value)
	}
	return cloned
}

func cloneProviderDefinedOptions(values map[string]any) map[string]any {
	if values == nil {
		return nil
	}
	cloned := make(map[string]any, len(values))
	for key, value := range values {
		cloned[key] = cloneProviderDefinedOption(value)
	}
	return cloned
}

func cloneProviderDefinedOption(value any) any {
	switch v := value.(type) {
	case nil:
		return nil
	case map[string]any:
		return cloneProviderDefinedOptions(v)
	case sigma.Schema:
		cloned := make(sigma.Schema, len(v))
		for key, value := range v {
			cloned[key] = cloneProviderDefinedOption(value)
		}
		return cloned
	case []any:
		cloned := make([]any, len(v))
		for index, item := range v {
			cloned[index] = cloneProviderDefinedOption(item)
		}
		return cloned
	case []string:
		return append([]string(nil), v...)
	case []byte:
		return append([]byte(nil), v...)
	case json.RawMessage:
		return append(json.RawMessage(nil), v...)
	default:
		return v
	}
}

func cloneAny(value any) any {
	switch v := value.(type) {
	case nil:
		return nil
	case map[string]any:
		return cloneStringAnyMap(v)
	case sigma.Schema:
		cloned := make(sigma.Schema, len(v))
		for key, value := range v {
			cloned[key] = cloneAny(value)
		}
		return cloned
	case []any:
		cloned := make([]any, len(v))
		for index, item := range v {
			cloned[index] = cloneAny(item)
		}
		return cloned
	case []string:
		return append([]string(nil), v...)
	case []byte:
		return append([]byte(nil), v...)
	case json.RawMessage:
		return append(json.RawMessage(nil), v...)
	default:
		return v
	}
}
