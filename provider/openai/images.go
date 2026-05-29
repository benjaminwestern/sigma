// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/wintermi/sigma"
)

const maxImagesResponseBytes = 64 << 20

// ImagesProvider adapts OpenAI's image generation API to sigma.
type ImagesProvider struct {
	base *Provider
}

// NewImagesProvider constructs an OpenAI Images API provider.
func NewImagesProvider(opts ...ProviderOption) *ImagesProvider {
	return &ImagesProvider{base: NewProvider(opts...)}
}

// RegisterImages adds an OpenAI Images API provider to registry.
func RegisterImages(registry *sigma.Registry, providerID sigma.ProviderID, opts ...ProviderOption) error {
	if registry == nil {
		return &sigma.Error{Code: sigma.ErrorUnsupported, Message: "registry is required"}
	}
	return registry.RegisterImageProvider(providerID, NewImagesProvider(opts...))
}

// RegisterImagesDefault adds an OpenAI Images API provider to sigma's default registry.
func RegisterImagesDefault(providerID sigma.ProviderID, opts ...ProviderOption) error {
	return sigma.RegisterDefaultImageProvider(providerID, NewImagesProvider(opts...))
}

// API reports the OpenAI Images API surface.
func (p *ImagesProvider) API() sigma.ImageAPI {
	return sigma.ImageAPIOpenAIImages
}

// Generate sends req to OpenAI's non-streaming image generation endpoint.
func (p *ImagesProvider) Generate(ctx context.Context, model sigma.ImageModel, req sigma.ImageRequest, opts sigma.Options) (sigma.AssistantImages, error) {
	ctx, cancel := sigma.ContextWithRequestTimeout(ctx, opts)
	defer cancel()

	resp, err := sigma.DoHTTPWithRetry(
		ctx,
		p.base.httpClient(opts),
		opts,
		func(ctx context.Context) (*http.Request, error) {
			return p.newRequest(ctx, model, req, opts)
		},
		func(resp *http.Response) *sigma.ProviderError {
			return imagesResponseError(resp, model)
		},
		sigma.ImageResponseDebugHTTPHook(ctx, opts, model.Provider, sigma.ImageAPIOpenAIImages, model.ID),
	)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || ctx.Err() != nil {
			return sigma.AssistantImages{StopReason: sigma.StopReasonAborted}, contextError(ctx, err)
		}
		return sigma.AssistantImages{StopReason: sigma.StopReasonError}, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxImagesResponseBytes))
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || ctx.Err() != nil {
			return sigma.AssistantImages{StopReason: sigma.StopReasonAborted}, contextError(ctx, err)
		}
		return sigma.AssistantImages{StopReason: sigma.StopReasonError}, err
	}
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return sigma.AssistantImages{StopReason: sigma.StopReasonError}, imagesProviderError(resp, model, body, nil)
	}

	return decodeImagesResponse(body, model, req)
}

func (p *ImagesProvider) newRequest(ctx context.Context, model sigma.ImageModel, req sigma.ImageRequest, opts sigma.Options) (*http.Request, error) {
	payload, err := imagesPayload(model, req, opts)
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("openai images: encode request: %w", err)
	}

	endpoint, err := p.endpoint(model, opts)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")
	httpReq.Header.Set("User-Agent", "sigma/openai-images")

	p.addProviderHeaders(httpReq, model.Provider, opts)
	for key, value := range p.base.headers {
		httpReq.Header.Set(key, value)
	}
	addImageModelHeaders(httpReq, model)
	for key, value := range opts.Headers {
		httpReq.Header.Set(key, value)
	}
	if err := p.addAuthHeader(ctx, httpReq, model, opts); err != nil {
		return nil, err
	}
	if err := sigma.RunImagePayloadDebugHooks(ctx, opts, model.Provider, sigma.ImageAPIOpenAIImages, model.ID, body, httpReq.Header); err != nil {
		return nil, err
	}
	return httpReq, nil
}

func imagesPayload(model sigma.ImageModel, req sigma.ImageRequest, opts sigma.Options) (map[string]any, error) {
	if strings.TrimSpace(req.Prompt) == "" {
		return nil, fmt.Errorf("openai images: prompt is required")
	}
	if len(req.Inputs) > 0 {
		return nil, fmt.Errorf("openai images: image inputs require the edits endpoint, which is not implemented")
	}

	payload := map[string]any{
		"model":  string(imageModelID(model, req)),
		"prompt": req.Prompt,
	}
	if req.Count > 0 {
		payload["n"] = req.Count
	}
	if req.Size != "" {
		payload["size"] = req.Size
	}
	if req.Quality != "" {
		payload["quality"] = req.Quality
	}
	if req.MIMEType != "" {
		format, err := outputFormat(req.MIMEType)
		if err != nil {
			return nil, err
		}
		payload["output_format"] = format
	}
	for key, value := range extraBody(opts, model.Provider) {
		payload[key] = value
	}
	return payload, nil
}

func imageModelID(model sigma.ImageModel, req sigma.ImageRequest) sigma.ModelID {
	if req.Model != "" {
		return req.Model
	}
	return model.ID
}

func outputFormat(mimeType string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(mimeType)) {
	case "image/png":
		return "png", nil
	case "image/jpeg":
		return "jpeg", nil
	case "image/webp":
		return "webp", nil
	default:
		return "", fmt.Errorf("openai images: unsupported output MIME type %q", mimeType)
	}
}

func outputMIMEType(format string, fallback string) string {
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "png":
		return "image/png"
	case "jpeg":
		return "image/jpeg"
	case "webp":
		return "image/webp"
	}
	if fallback != "" {
		return fallback
	}
	return "image/png"
}

func (p *ImagesProvider) addProviderHeaders(req *http.Request, provider sigma.ProviderID, opts sigma.Options) {
	options := providerOptions(opts, provider)
	if organization, ok := stringOption(options, providerOptionOrganization); ok {
		req.Header.Set("OpenAI-Organization", organization)
	}
	if project, ok := stringOption(options, providerOptionProject); ok {
		req.Header.Set("OpenAI-Project", project)
	}
}

func (p *ImagesProvider) addAuthHeader(ctx context.Context, req *http.Request, model sigma.ImageModel, opts sigma.Options) error {
	if opts.AuthResolver == nil {
		return &sigma.Error{
			Code:     sigma.ErrorUnsupported,
			Message:  "openai images: auth resolver is required",
			Provider: model.Provider,
			Model:    model.ID,
		}
	}
	credential, err := opts.AuthResolver.Resolve(ctx, imageAuthModel(model), opts)
	if err != nil {
		return err
	}
	if credential.Value != "" {
		req.Header.Set("Authorization", "Bearer "+credential.Value)
	}
	return nil
}

func imageAuthModel(model sigma.ImageModel) sigma.Model {
	return sigma.Model{
		ID:               model.ID,
		Provider:         model.Provider,
		API:              sigma.API(model.API),
		Name:             model.Name,
		ProviderMetadata: copyAnyMap(model.ProviderMetadata),
	}
}

func (p *ImagesProvider) endpoint(model sigma.ImageModel, opts sigma.Options) (string, error) {
	options := providerOptions(opts, model.Provider)
	if endpoint, ok := stringOption(options, providerOptionEndpoint); ok {
		if err := validateImagesEndpoint(endpoint); err != nil {
			return "", err
		}
		return endpoint, nil
	}

	baseURL := p.baseURLForModel(model, opts)
	parsed, err := url.Parse(baseURL)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return "", fmt.Errorf("openai images: invalid base URL %q", baseURL)
	}
	return baseURL + "/images/generations", nil
}

func validateImagesEndpoint(endpoint string) error {
	parsed, err := url.Parse(endpoint)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return fmt.Errorf("openai images: invalid endpoint %q", endpoint)
	}
	return nil
}

func (p *ImagesProvider) baseURLForModel(model sigma.ImageModel, opts sigma.Options) string {
	baseURL := p.base.defaultBaseURL()
	if value, ok := model.ProviderMetadata[sigma.MetadataOpenAICompatibleBaseURL].(string); ok && strings.TrimSpace(value) != "" {
		baseURL = value
	} else if value, ok := model.ProviderMetadata["baseURL"].(string); ok && strings.TrimSpace(value) != "" {
		baseURL = value
	}
	options := providerOptions(opts, model.Provider)
	if value, ok := stringOption(options, providerOptionBaseURL); ok {
		baseURL = value
	} else if value, ok := stringOption(options, providerOptionBaseURLCamel); ok {
		baseURL = value
	}
	return strings.TrimRight(baseURL, "/")
}

func addImageModelHeaders(req *http.Request, model sigma.ImageModel) {
	for key, value := range imageModelHeaders(model) {
		if unsafeCredentialHeader(key) {
			continue
		}
		req.Header.Set(key, value)
	}
}

func imageModelHeaders(model sigma.ImageModel) map[string]string {
	raw := model.ProviderMetadata[sigma.MetadataOpenAICompatibleHeaders]
	if raw == nil {
		raw = model.ProviderMetadata["headers"]
	}
	switch headers := raw.(type) {
	case map[string]string:
		return headers
	case map[string]any:
		copied := make(map[string]string, len(headers))
		for key, value := range headers {
			text, ok := value.(string)
			if !ok {
				continue
			}
			copied[key] = text
		}
		return copied
	default:
		return nil
	}
}

type imagesResponse struct {
	Created      int64           `json:"created"`
	Background   string          `json:"background"`
	Data         []imageData     `json:"data"`
	OutputFormat string          `json:"output_format"`
	Quality      string          `json:"quality"`
	Size         string          `json:"size"`
	Usage        *imagesUsage    `json:"usage"`
	Error        *imagesAPIError `json:"error"`
}

type imageData struct {
	B64JSON       string `json:"b64_json"`
	URL           string `json:"url"`
	RevisedPrompt string `json:"revised_prompt"`
}

type imagesAPIError struct {
	Code    any    `json:"code"`
	Message string `json:"message"`
	Type    string `json:"type"`
}

type imagesUsage struct {
	InputTokens        int `json:"input_tokens"`
	OutputTokens       int `json:"output_tokens"`
	TotalTokens        int `json:"total_tokens"`
	InputTokensDetails struct {
		ImageTokens int `json:"image_tokens"`
		TextTokens  int `json:"text_tokens"`
	} `json:"input_tokens_details"`
	OutputTokensDetails struct {
		ImageTokens int `json:"image_tokens"`
		TextTokens  int `json:"text_tokens"`
	} `json:"output_tokens_details"`
}

func decodeImagesResponse(body []byte, model sigma.ImageModel, req sigma.ImageRequest) (sigma.AssistantImages, error) {
	var decoded imagesResponse
	if err := json.Unmarshal(body, &decoded); err != nil {
		return sigma.AssistantImages{StopReason: sigma.StopReasonError}, fmt.Errorf("openai images: decode response: %w", err)
	}
	if decoded.Error != nil {
		images := sigma.AssistantImages{
			StopReason: sigma.StopReasonError,
			Errors: []sigma.ImageError{{
				Code:    fmt.Sprint(decoded.Error.Code),
				Message: decoded.Error.Message,
				ProviderMetadata: map[string]any{
					providerToolOptionTypeKey: decoded.Error.Type,
				},
			}},
		}
		return images, sigma.NewProviderError(model.Provider, sigma.API(sigma.ImageAPIOpenAIImages), model.ID, http.StatusOK, "", 0, body, nil)
	}

	mimeType := outputMIMEType(decoded.OutputFormat, req.MIMEType)
	images := sigma.AssistantImages{
		Model:            model.ID,
		Provider:         model.Provider,
		StopReason:       sigma.StopReasonEndTurn,
		ProviderMetadata: imagesProviderMetadata(decoded),
	}
	for _, item := range decoded.Data {
		if item.B64JSON != "" {
			images.Images = append(images.Images, sigma.ImageOutputData(mimeType, item.B64JSON))
		}
		if item.URL != "" {
			images.Images = append(images.Images, sigma.ImageOutputURL("", item.URL))
		}
	}
	if decoded.Usage != nil {
		usage := decoded.Usage.sigmaUsage()
		images.Usage = &usage
	}
	return images, nil
}

func imagesProviderMetadata(decoded imagesResponse) map[string]any {
	metadata := make(map[string]any)
	if decoded.Created != 0 {
		metadata["created"] = decoded.Created
	}
	if decoded.Background != "" {
		metadata["background"] = decoded.Background
	}
	if decoded.OutputFormat != "" {
		metadata["output_format"] = decoded.OutputFormat
	}
	if decoded.Quality != "" {
		metadata["quality"] = decoded.Quality
	}
	if decoded.Size != "" {
		metadata["size"] = decoded.Size
	}
	var revisedPrompts []string
	for _, item := range decoded.Data {
		if item.RevisedPrompt != "" {
			revisedPrompts = append(revisedPrompts, item.RevisedPrompt)
		}
	}
	if len(revisedPrompts) == 1 {
		metadata["revised_prompt"] = revisedPrompts[0]
	} else if len(revisedPrompts) > 1 {
		metadata["revised_prompts"] = revisedPrompts
	}
	if decoded.Usage != nil {
		metadata["usage"] = decoded.Usage.providerMetadata()
	}
	if len(metadata) == 0 {
		return nil
	}
	return metadata
}

func (u imagesUsage) sigmaUsage() sigma.Usage {
	return sigma.Usage{
		InputTokens:  u.InputTokens,
		OutputTokens: u.OutputTokens,
		TotalTokens:  u.TotalTokens,
	}
}

func (u imagesUsage) providerMetadata() map[string]any {
	return map[string]any{
		"input_tokens_details": map[string]any{
			"image_tokens": u.InputTokensDetails.ImageTokens,
			"text_tokens":  u.InputTokensDetails.TextTokens,
		},
		"output_tokens_details": map[string]any{
			"image_tokens": u.OutputTokensDetails.ImageTokens,
			"text_tokens":  u.OutputTokensDetails.TextTokens,
		},
	}
}

func imagesResponseError(resp *http.Response, model sigma.ImageModel) *sigma.ProviderError {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	return imagesProviderError(resp, model, body, nil)
}

func imagesProviderError(resp *http.Response, model sigma.ImageModel, body []byte, err error) *sigma.ProviderError {
	return sigma.NewProviderError(
		model.Provider,
		sigma.API(sigma.ImageAPIOpenAIImages),
		model.ID,
		resp.StatusCode,
		requestID(resp.Header),
		sigma.RetryAfter(resp.Header),
		body,
		err,
	)
}
