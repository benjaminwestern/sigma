// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package openai

import "github.com/wintermi/sigma"

const providerToolOptionTypeKey = "type"

// Tools provides factories for OpenAI Responses provider-defined tools.
var Tools = struct {
	WebSearch       func(opts ...WebSearchOption) sigma.Tool
	CodeInterpreter func(opts ...CodeInterpreterOption) sigma.Tool
	FileSearch      func(opts ...FileSearchOption) sigma.Tool
	ImageGeneration func(opts ...ImageGenerationOption) sigma.Tool
}{
	WebSearch:       webSearchTool,
	CodeInterpreter: codeInterpreterTool,
	FileSearch:      fileSearchTool,
	ImageGeneration: imageGenerationTool,
}

// WebSearchOption configures OpenAI web search.
type WebSearchOption func(*webSearchConfig)

type webSearchConfig struct {
	searchContextSize string
	userLocation      *WebSearchLocation
	filters           *WebSearchFilters
	externalWebAccess *bool
}

// WebSearchLocation configures approximate user location for search.
type WebSearchLocation struct {
	Country  string
	City     string
	Region   string
	Timezone string
}

// WebSearchFilters configures search result filtering.
type WebSearchFilters struct {
	AllowedDomains []string
}

func WithSearchContextSize(size string) WebSearchOption {
	return func(config *webSearchConfig) { config.searchContextSize = size }
}

func WithUserLocation(location WebSearchLocation) WebSearchOption {
	return func(config *webSearchConfig) { config.userLocation = &location }
}

func WithSearchFilters(filters WebSearchFilters) WebSearchOption {
	return func(config *webSearchConfig) { config.filters = &filters }
}

func WithExternalWebAccess(enabled bool) WebSearchOption {
	return func(config *webSearchConfig) { config.externalWebAccess = &enabled }
}

func webSearchTool(opts ...WebSearchOption) sigma.Tool {
	config := webSearchConfig{}
	for _, opt := range opts {
		opt(&config)
	}
	options := make(map[string]any)
	if config.searchContextSize != "" {
		options["search_context_size"] = config.searchContextSize
	}
	if config.userLocation != nil {
		location := map[string]any{providerToolOptionTypeKey: "approximate"}
		if config.userLocation.Country != "" {
			location["country"] = config.userLocation.Country
		}
		if config.userLocation.City != "" {
			location["city"] = config.userLocation.City
		}
		if config.userLocation.Region != "" {
			location["region"] = config.userLocation.Region
		}
		if config.userLocation.Timezone != "" {
			location["timezone"] = config.userLocation.Timezone
		}
		options["user_location"] = location
	}
	if config.filters != nil && len(config.filters.AllowedDomains) > 0 {
		options["filters"] = map[string]any{"allowed_domains": config.filters.AllowedDomains}
	}
	if config.externalWebAccess != nil {
		options["external_web_access"] = *config.externalWebAccess
	}
	return providerTool("web_search", "web_search", options)
}

// CodeInterpreterOption configures OpenAI code interpreter.
type CodeInterpreterOption func(*codeInterpreterConfig)

type codeInterpreterConfig struct {
	container any
}

// CodeInterpreterContainer configures an auto-provisioned code interpreter container.
type CodeInterpreterContainer struct {
	FileIDs []string
}

func WithContainerID(id string) CodeInterpreterOption {
	return func(config *codeInterpreterConfig) { config.container = id }
}

func WithContainerFiles(container *CodeInterpreterContainer) CodeInterpreterOption {
	return func(config *codeInterpreterConfig) { config.container = container }
}

func codeInterpreterTool(opts ...CodeInterpreterOption) sigma.Tool {
	config := codeInterpreterConfig{}
	for _, opt := range opts {
		opt(&config)
	}
	options := make(map[string]any)
	switch container := config.container.(type) {
	case string:
		options["container"] = container
	case *CodeInterpreterContainer:
		options["container"] = map[string]any{
			providerToolOptionTypeKey: "auto",
			"file_ids":                container.FileIDs,
		}
	default:
		options["container"] = map[string]any{providerToolOptionTypeKey: "auto"}
	}
	return providerTool("code_interpreter", "code_interpreter", options)
}

// FileSearchOption configures OpenAI file search.
type FileSearchOption func(*fileSearchConfig)

type fileSearchConfig struct {
	vectorStoreIDs []string
	maxNumResults  int
	ranking        *FileSearchRanking
	filters        FileSearchFilter
}

type FileSearchRanking struct {
	Ranker         string
	ScoreThreshold float64
}

type FileSearchComparisonFilter struct {
	Key   string
	Type  string
	Value any
}

type FileSearchCompoundFilter struct {
	Type    string
	Filters []FileSearchFilter
}

type FileSearchFilter interface {
	fileSearchFilter()
}

func (*FileSearchComparisonFilter) fileSearchFilter() {}
func (*FileSearchCompoundFilter) fileSearchFilter()   {}

func WithVectorStoreIDs(ids ...string) FileSearchOption {
	return func(config *fileSearchConfig) { config.vectorStoreIDs = ids }
}

func WithMaxNumResults(results int) FileSearchOption {
	return func(config *fileSearchConfig) { config.maxNumResults = results }
}

func WithRanking(ranking FileSearchRanking) FileSearchOption {
	return func(config *fileSearchConfig) { config.ranking = &ranking }
}

func WithFileSearchFilters(filters FileSearchFilter) FileSearchOption {
	return func(config *fileSearchConfig) { config.filters = filters }
}

func fileSearchTool(opts ...FileSearchOption) sigma.Tool {
	config := fileSearchConfig{}
	for _, opt := range opts {
		opt(&config)
	}
	options := make(map[string]any)
	if len(config.vectorStoreIDs) > 0 {
		options["vector_store_ids"] = config.vectorStoreIDs
	}
	if config.maxNumResults > 0 {
		options["max_num_results"] = config.maxNumResults
	}
	if config.ranking != nil {
		ranking := make(map[string]any)
		if config.ranking.Ranker != "" {
			ranking["ranker"] = config.ranking.Ranker
		}
		if config.ranking.ScoreThreshold > 0 {
			ranking["score_threshold"] = config.ranking.ScoreThreshold
		}
		options["ranking_options"] = ranking
	}
	if config.filters != nil {
		options["filters"] = serializeFileSearchFilter(config.filters)
	}
	return providerTool("file_search", "file_search", options)
}

func serializeFileSearchFilter(filter FileSearchFilter) any {
	switch typed := filter.(type) {
	case *FileSearchComparisonFilter:
		return map[string]any{
			"key":                     typed.Key,
			providerToolOptionTypeKey: typed.Type,
			"value":                   typed.Value,
		}
	case *FileSearchCompoundFilter:
		filters := make([]any, len(typed.Filters))
		for index, nested := range typed.Filters {
			filters[index] = serializeFileSearchFilter(nested)
		}
		return map[string]any{
			providerToolOptionTypeKey: typed.Type,
			"filters":                 filters,
		}
	default:
		return nil
	}
}

// ImageGenerationOption configures OpenAI image generation as a Responses tool.
type ImageGenerationOption func(*imageGenerationConfig)

type imageGenerationConfig struct {
	background        string
	inputFidelity     string
	inputImageMask    *ImageGenerationMask
	model             string
	moderation        string
	outputCompression *int
	outputFormat      string
	partialImages     *int
	quality           string
	size              string
}

type ImageGenerationMask struct {
	FileID   string
	ImageURL string
}

func WithBackground(background string) ImageGenerationOption {
	return func(config *imageGenerationConfig) { config.background = background }
}

func WithInputFidelity(fidelity string) ImageGenerationOption {
	return func(config *imageGenerationConfig) { config.inputFidelity = fidelity }
}

func WithInputImageMask(mask ImageGenerationMask) ImageGenerationOption {
	return func(config *imageGenerationConfig) { config.inputImageMask = &mask }
}

func WithImageModel(model string) ImageGenerationOption {
	return func(config *imageGenerationConfig) { config.model = model }
}

func WithModeration(moderation string) ImageGenerationOption {
	return func(config *imageGenerationConfig) { config.moderation = moderation }
}

func WithOutputCompression(compression int) ImageGenerationOption {
	return func(config *imageGenerationConfig) { config.outputCompression = &compression }
}

func WithOutputFormat(format string) ImageGenerationOption {
	return func(config *imageGenerationConfig) { config.outputFormat = format }
}

func WithPartialImages(images int) ImageGenerationOption {
	return func(config *imageGenerationConfig) { config.partialImages = &images }
}

func WithImageQuality(quality string) ImageGenerationOption {
	return func(config *imageGenerationConfig) { config.quality = quality }
}

func WithImageSize(size string) ImageGenerationOption {
	return func(config *imageGenerationConfig) { config.size = size }
}

func imageGenerationTool(opts ...ImageGenerationOption) sigma.Tool {
	config := imageGenerationConfig{}
	for _, opt := range opts {
		opt(&config)
	}
	options := make(map[string]any)
	if config.background != "" {
		options["background"] = config.background
	}
	if config.inputFidelity != "" {
		options["input_fidelity"] = config.inputFidelity
	}
	if config.inputImageMask != nil {
		mask := make(map[string]any)
		if config.inputImageMask.FileID != "" {
			mask["file_id"] = config.inputImageMask.FileID
		}
		if config.inputImageMask.ImageURL != "" {
			mask["image_url"] = config.inputImageMask.ImageURL
		}
		options["input_image_mask"] = mask
	}
	if config.model != "" {
		options["model"] = config.model
	}
	if config.moderation != "" {
		options["moderation"] = config.moderation
	}
	if config.outputCompression != nil {
		options["output_compression"] = *config.outputCompression
	}
	if config.outputFormat != "" {
		options["output_format"] = config.outputFormat
	}
	if config.partialImages != nil {
		options["partial_images"] = *config.partialImages
	}
	if config.quality != "" {
		options["quality"] = config.quality
	}
	if config.size != "" {
		options["size"] = config.size
	}
	return providerTool("image_generation", "image_generation", options)
}

func providerTool(name string, providerType string, options map[string]any) sigma.Tool {
	if len(options) == 0 {
		options = nil
	}
	return sigma.Tool{
		Name:                   name,
		ProviderDefinedType:    providerType,
		ProviderDefinedOptions: options,
	}
}
