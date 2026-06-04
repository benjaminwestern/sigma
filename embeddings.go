// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package sigma

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// EmbeddingOption configures a single embedding provider request.
type EmbeddingOption func(*Options)

// EmbeddingBatchPhase identifies a resilient embedding batch progress stage.
type EmbeddingBatchPhase string

const (
	EmbeddingBatchPhaseCacheHit     EmbeddingBatchPhase = "cache_hit"
	EmbeddingBatchPhaseBatchStart   EmbeddingBatchPhase = "batch_start"
	EmbeddingBatchPhaseBatchSuccess EmbeddingBatchPhase = "batch_success"
	EmbeddingBatchPhaseBatchError   EmbeddingBatchPhase = "batch_error"
	EmbeddingBatchPhaseSplit        EmbeddingBatchPhase = "split"
)

// EmbeddingBatchProgress reports progress from EmbedBatch.
type EmbeddingBatchProgress struct {
	Phase        EmbeddingBatchPhase
	Attempt      int
	BatchSize    int
	InputIndexes []int
	SplitPart    int
	SplitTotal   int
	Err          error
}

// EmbeddingBatchConfig configures resilient embedding batch behaviour.
type EmbeddingBatchConfig struct {
	ReuseDuplicateInputs bool
	MaxRetries           int
	MaxParallelBatches   int
	SplitOversized       bool
	Progress             func(EmbeddingBatchProgress) error
}

// EmbeddingBatchSummary reports aggregate provider work from EmbedBatch.
type EmbeddingBatchSummary struct {
	RequestCount      int
	TotalRequestCount int
	ErrorCount        int
	VectorCount       int
	StatusBuckets     map[int]int
	RequestIDs        []string
	Attempts          []EmbeddingAttempt
	Usage             *Usage
	Cost              *Cost
}

// EmbeddingBatchResult is ordered embedding output plus batch metadata.
type EmbeddingBatchResult struct {
	Embeddings Embeddings
	Reused     []bool
	Summary    EmbeddingBatchSummary
}

type embeddingBatchJob struct {
	text    string
	indexes []int
}

type embeddingBatcher struct {
	ctx    context.Context
	client *Client
	model  EmbeddingModel
	req    EmbeddingRequest
	config EmbeddingBatchConfig
	opts   []EmbeddingOption

	cache map[string]Embedding
	sem   chan struct{}

	mu      sync.Mutex
	summary EmbeddingBatchSummary
}

// WithEmbeddingAPIKey configures a request-scoped embedding API key override.
func WithEmbeddingAPIKey(apiKey string) EmbeddingOption {
	return embeddingOptionFromOption(WithAPIKey(apiKey))
}

// WithEmbeddingHTTPClient configures the HTTP client exposed to embedding providers.
func WithEmbeddingHTTPClient(httpClient *http.Client) EmbeddingOption {
	return func(options *Options) {
		options.HTTPClient = httpClient
	}
}

// WithEmbeddingAuthResolver configures a request-scoped credential resolver.
func WithEmbeddingAuthResolver(resolver AuthResolver) EmbeddingOption {
	return func(options *Options) {
		options.AuthResolver = resolver
	}
}

// WithEmbeddingHeader adds or replaces an embedding request header.
func WithEmbeddingHeader(key, value string) EmbeddingOption {
	return embeddingOptionFromOption(WithHeader(key, value))
}

// WithEmbeddingHeaders adds or replaces embedding request headers.
func WithEmbeddingHeaders(headers map[string]string) EmbeddingOption {
	return embeddingOptionFromOption(WithHeaders(headers))
}

// WithEmbeddingTimeout configures the per-request embedding provider timeout.
func WithEmbeddingTimeout(timeout time.Duration) EmbeddingOption {
	return embeddingOptionFromOption(WithTimeout(timeout))
}

// WithEmbeddingMaxRetries configures the maximum embedding provider retry attempts.
func WithEmbeddingMaxRetries(maxRetries int) EmbeddingOption {
	return embeddingOptionFromOption(WithMaxRetries(maxRetries))
}

// WithEmbeddingMaxRetryDelay configures the maximum delay between embedding provider retries.
func WithEmbeddingMaxRetryDelay(maxRetryDelay time.Duration) EmbeddingOption {
	return embeddingOptionFromOption(WithMaxRetryDelay(maxRetryDelay))
}

// WithEmbeddingMetadata adds or replaces provider-neutral embedding request metadata.
func WithEmbeddingMetadata(metadata map[string]any) EmbeddingOption {
	return embeddingOptionFromOption(WithMetadata(metadata))
}

// WithEmbeddingMetadataValue adds or replaces one provider-neutral embedding metadata value.
func WithEmbeddingMetadataValue(key string, value any) EmbeddingOption {
	return embeddingOptionFromOption(WithMetadataValue(key, value))
}

// WithEmbeddingProviderOptions adds or replaces advanced provider-specific embedding values.
func WithEmbeddingProviderOptions(provider ProviderID, values map[string]any) EmbeddingOption {
	return embeddingOptionFromOption(WithProviderOptions(provider, values))
}

// WithEmbeddingProviderOption adds or replaces one advanced provider-specific embedding value.
func WithEmbeddingProviderOption(provider ProviderID, key string, value any) EmbeddingOption {
	return embeddingOptionFromOption(WithProviderOption(provider, key, value))
}

// WithEmbeddingProviderAuthResolver configures a provider-specific embedding credential callback.
func WithEmbeddingProviderAuthResolver(provider ProviderID, resolver AuthResolver) EmbeddingOption {
	return embeddingOptionFromOption(WithProviderAuthResolver(provider, resolver))
}

// Embed calls the registered embedding provider for model.
func (c *Client) Embed(ctx context.Context, model EmbeddingModel, req EmbeddingRequest, opts ...EmbeddingOption) (Embeddings, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if c == nil {
		c = NewClient()
	}
	if err := ValidateModelRef(ModelRef{Provider: model.Provider, ID: model.ID}); err != nil {
		return Embeddings{Model: model.ID, Provider: model.Provider}, err
	}

	registered, ok := c.GetEmbeddingModel(model.Provider, model.ID)
	if !ok {
		return Embeddings{Model: model.ID, Provider: model.Provider}, embeddingModelNotFoundError(model.Provider, model.ID)
	}
	if model.API == "" {
		model = registered
	}

	provider, ok := c.registry.EmbeddingProvider(model.Provider)
	if !ok {
		return Embeddings{Model: model.ID, Provider: model.Provider}, embeddingProviderNotFoundError(model.Provider, model.ID)
	}

	options := c.embeddingRequestOptions(opts)
	if err := validateEmbeddingOptions(model, req, options); err != nil {
		return Embeddings{Model: model.ID, Provider: model.Provider}, err
	}
	if err := ctx.Err(); err != nil {
		return Embeddings{Model: model.ID, Provider: model.Provider}, embeddingAbortedError(err)
	}

	embeddings, err := provider.Embed(ctx, model, req, options)
	embeddings = finalEmbeddings(model, embeddings)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return embeddings, embeddingAbortedError(err)
		}
		return embeddings, fmt.Errorf("embedding provider: %w", err)
	}
	return embeddings, nil
}

// Embed calls the registered embedding provider using the default registry.
func Embed(ctx context.Context, model EmbeddingModel, req EmbeddingRequest, opts ...EmbeddingOption) (Embeddings, error) {
	return defaultClient().Embed(ctx, model, req, opts...)
}

// EmbedBatch embeds req.Inputs with duplicate reuse and retry-aware batch splitting.
func (c *Client) EmbedBatch(ctx context.Context, model EmbeddingModel, req EmbeddingRequest, config EmbeddingBatchConfig, opts ...EmbeddingOption) (EmbeddingBatchResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if c == nil {
		c = NewClient()
	}
	if config.MaxRetries < 0 {
		return EmbeddingBatchResult{Embeddings: Embeddings{Model: model.ID, Provider: model.Provider}}, invalidEmbeddingOptionsError(model, "embedding batch max retries must be non-negative")
	}
	if config.MaxParallelBatches < 0 {
		return EmbeddingBatchResult{Embeddings: Embeddings{Model: model.ID, Provider: model.Provider}}, invalidEmbeddingOptionsError(model, "embedding batch max parallel batches must be non-negative")
	}
	if len(req.Inputs) == 0 {
		embeddings, err := c.Embed(ctx, model, req, opts...)
		return EmbeddingBatchResult{Embeddings: embeddings}, err
	}

	batcher := &embeddingBatcher{
		ctx:    ctx,
		client: c,
		model:  model,
		req:    req,
		config: config,
		opts:   append([]EmbeddingOption(nil), opts...),
	}
	if config.ReuseDuplicateInputs {
		batcher.cache = make(map[string]Embedding)
	}
	if config.MaxParallelBatches > 0 {
		batcher.sem = make(chan struct{}, config.MaxParallelBatches)
	}

	jobs, reused, err := batcher.jobs(req.Inputs)
	if err != nil {
		return EmbeddingBatchResult{
			Embeddings: Embeddings{Model: model.ID, Provider: model.Provider},
			Reused:     reused,
			Summary:    batcher.batchSummary(),
		}, err
	}
	vectors, err := batcher.embedJobs(jobs, 0)
	vectors = orderEmbeddingsByIndex(vectors)
	result := EmbeddingBatchResult{
		Embeddings: Embeddings{
			Model:    model.ID,
			Provider: model.Provider,
			Vectors:  vectors,
		},
		Reused:  reused,
		Summary: batcher.batchSummary(),
	}
	result.Embeddings = finalEmbeddings(model, result.Embeddings)
	if result.Summary.Usage != nil {
		usage := *result.Summary.Usage
		result.Embeddings.Usage = &usage
	}
	if result.Summary.Cost != nil {
		cost := *result.Summary.Cost
		result.Embeddings.Cost = &cost
	}
	result.Embeddings.Attempts = append([]EmbeddingAttempt(nil), result.Summary.Attempts...)
	if err != nil {
		return result, err
	}
	return result, nil
}

// EmbedBatch embeds req.Inputs using the default registry.
func EmbedBatch(ctx context.Context, model EmbeddingModel, req EmbeddingRequest, config EmbeddingBatchConfig, opts ...EmbeddingOption) (EmbeddingBatchResult, error) {
	return defaultClient().EmbedBatch(ctx, model, req, config, opts...)
}

// GetEmbeddingModel returns an embedding model by provider and model id.
func (c *Client) GetEmbeddingModel(provider ProviderID, id ModelID) (EmbeddingModel, bool) {
	if c == nil || c.registry == nil {
		return EmbeddingModel{}, false
	}
	return c.registry.EmbeddingModel(provider, id)
}

// EmbeddingModels returns embedding models from the client registry.
func (c *Client) EmbeddingModels() []EmbeddingModel {
	if c == nil || c.registry == nil {
		return nil
	}
	return c.registry.ListEmbeddingModels()
}

// GetEmbeddingModel returns an embedding model from the default registry.
func GetEmbeddingModel(provider ProviderID, id ModelID) (EmbeddingModel, bool) {
	return defaultClient().GetEmbeddingModel(provider, id)
}

// EmbeddingModels returns embedding models from the default registry.
func EmbeddingModels() []EmbeddingModel {
	return defaultClient().EmbeddingModels()
}

func (b *embeddingBatcher) jobs(inputs []string) ([]embeddingBatchJob, []bool, error) {
	reused := make([]bool, len(inputs))
	if !b.config.ReuseDuplicateInputs {
		jobs := make([]embeddingBatchJob, 0, len(inputs))
		for index, input := range inputs {
			jobs = append(jobs, embeddingBatchJob{text: input, indexes: []int{index}})
		}
		return jobs, reused, nil
	}

	byKey := make(map[string]int)
	jobs := make([]embeddingBatchJob, 0, len(inputs))
	for index, input := range inputs {
		key := b.cacheKey(input)
		if existing, ok := byKey[key]; ok {
			jobs[existing].indexes = append(jobs[existing].indexes, index)
			reused[index] = true
			if err := b.progress(EmbeddingBatchProgress{
				Phase:        EmbeddingBatchPhaseCacheHit,
				InputIndexes: []int{index},
			}); err != nil {
				return nil, reused, err
			}
			continue
		}
		byKey[key] = len(jobs)
		jobs = append(jobs, embeddingBatchJob{text: input, indexes: []int{index}})
	}
	return jobs, reused, nil
}

func (b *embeddingBatcher) embedJobs(jobs []embeddingBatchJob, attempt int) ([]Embedding, error) {
	if len(jobs) == 0 {
		return nil, nil
	}
	if b.config.ReuseDuplicateInputs {
		if cached, ok := b.cachedEmbeddings(jobs); ok {
			return cached, nil
		}
	}

	embeddings, err := b.callProvider(jobs, attempt)
	if err == nil {
		jobVectors, err := embeddingsForJobs(jobs, embeddings)
		if err != nil {
			return nil, err
		}
		b.addResult(embeddings)
		if b.config.ReuseDuplicateInputs {
			b.storeCache(jobs, jobVectors)
		}
		return expandJobEmbeddings(jobs, jobVectors), nil
	}

	b.addError(embeddings)
	if err := b.progress(EmbeddingBatchProgress{
		Phase:        EmbeddingBatchPhaseBatchError,
		Attempt:      attempt,
		BatchSize:    len(jobs),
		InputIndexes: indexesForJobs(jobs),
		Err:          err,
	}); err != nil {
		return nil, err
	}
	if attempt >= b.config.MaxRetries {
		return nil, err
	}
	if len(jobs) > 1 {
		if !ClassifyError(err).RetryHint.Retryable {
			return nil, err
		}
		return b.embedSplitBatches(jobs, attempt+1)
	}
	if !b.config.SplitOversized || ClassifyError(err).Class != ErrorClassContextOverflow {
		return nil, err
	}
	return b.embedOversizedSingleton(jobs[0], attempt+1)
}

func (b *embeddingBatcher) callProvider(jobs []embeddingBatchJob, attempt int) (Embeddings, error) {
	if b.sem != nil {
		select {
		case b.sem <- struct{}{}:
			defer func() { <-b.sem }()
		case <-b.ctx.Done():
			return Embeddings{}, b.ctx.Err()
		}
	}
	req := b.req
	req.Inputs = inputsForJobs(jobs)
	if err := b.progress(EmbeddingBatchProgress{
		Phase:        EmbeddingBatchPhaseBatchStart,
		Attempt:      attempt,
		BatchSize:    len(jobs),
		InputIndexes: indexesForJobs(jobs),
	}); err != nil {
		return Embeddings{}, err
	}
	embeddings, err := b.client.Embed(b.ctx, b.model, req, b.opts...)
	if err != nil {
		return embeddings, err
	}
	if err := b.progress(EmbeddingBatchProgress{
		Phase:        EmbeddingBatchPhaseBatchSuccess,
		Attempt:      attempt,
		BatchSize:    len(jobs),
		InputIndexes: indexesForJobs(jobs),
	}); err != nil {
		return Embeddings{}, err
	}
	return embeddings, nil
}

func (b *embeddingBatcher) embedSplitBatches(jobs []embeddingBatchJob, attempt int) ([]Embedding, error) {
	mid := len(jobs) / 2
	left := jobs[:mid]
	right := jobs[mid:]
	if b.sem == nil {
		leftVectors, err := b.embedJobs(left, attempt)
		if err != nil {
			return nil, err
		}
		rightVectors, err := b.embedJobs(right, attempt)
		if err != nil {
			return nil, err
		}
		return appendEmbeddingSlices(leftVectors, rightVectors), nil
	}

	var leftVectors, rightVectors []Embedding
	var leftErr, rightErr error
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		leftVectors, leftErr = b.embedJobs(left, attempt)
	}()
	go func() {
		defer wg.Done()
		rightVectors, rightErr = b.embedJobs(right, attempt)
	}()
	wg.Wait()
	if leftErr != nil {
		return nil, leftErr
	}
	if rightErr != nil {
		return nil, rightErr
	}
	return appendEmbeddingSlices(leftVectors, rightVectors), nil
}

func (b *embeddingBatcher) embedOversizedSingleton(job embeddingBatchJob, attempt int) ([]Embedding, error) {
	parts := splitEmbeddingInput(job.text)
	if len(parts) < 2 {
		return nil, invalidEmbeddingOptionsError(b.model, "embedding input cannot be split further")
	}
	partJobs := make([]embeddingBatchJob, 0, len(parts))
	for i, part := range parts {
		if err := b.progress(EmbeddingBatchProgress{
			Phase:        EmbeddingBatchPhaseSplit,
			Attempt:      attempt,
			BatchSize:    1,
			InputIndexes: append([]int(nil), job.indexes...),
			SplitPart:    i + 1,
			SplitTotal:   len(parts),
		}); err != nil {
			return nil, err
		}
		partJobs = append(partJobs, embeddingBatchJob{text: part, indexes: []int{i}})
	}
	partVectors, err := b.embedJobs(partJobs, attempt)
	if err != nil {
		return nil, err
	}
	weights := make([]int, 0, len(parts))
	for _, part := range parts {
		weights = append(weights, len([]rune(part)))
	}
	vector := weightedAverageEmbedding(partVectors, weights)
	out := make([]Embedding, 0, len(job.indexes))
	for _, index := range job.indexes {
		out = append(out, Embedding{Index: index, Vector: append([]float32(nil), vector...)})
	}
	return out, nil
}

func (b *embeddingBatcher) progress(progress EmbeddingBatchProgress) error {
	if b.config.Progress == nil {
		return nil
	}
	progress.InputIndexes = append([]int(nil), progress.InputIndexes...)
	return b.config.Progress(progress)
}

func (b *embeddingBatcher) cachedEmbeddings(jobs []embeddingBatchJob) ([]Embedding, bool) {
	out := make([]Embedding, 0, len(jobs))
	for _, job := range jobs {
		embedding, ok := b.cache[b.cacheKey(job.text)]
		if !ok {
			return nil, false
		}
		for _, index := range job.indexes {
			out = append(out, Embedding{Index: index, Vector: append([]float32(nil), embedding.Vector...)})
		}
	}
	return out, true
}

func (b *embeddingBatcher) storeCache(jobs []embeddingBatchJob, embeddings []Embedding) {
	for i, job := range jobs {
		b.cache[b.cacheKey(job.text)] = Embedding{
			Index:  0,
			Vector: append([]float32(nil), embeddings[i].Vector...),
		}
	}
}

func (b *embeddingBatcher) cacheKey(text string) string {
	sum := sha256.Sum256([]byte(text))
	return fmt.Sprintf("%s\x00%s\x00%d\x00%x", b.model.Provider, b.model.ID, b.req.Dimensions, sum)
}

func (b *embeddingBatcher) addResult(embeddings Embeddings) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.summary.RequestCount++
	b.addAttemptsLocked(embeddings.Attempts, 1)
	b.summary.VectorCount += len(embeddings.Vectors)
	if embeddings.Usage != nil {
		if b.summary.Usage == nil {
			b.summary.Usage = &Usage{}
		}
		b.summary.Usage.InputTokens += embeddings.Usage.InputTokens
		b.summary.Usage.OutputTokens += embeddings.Usage.OutputTokens
		b.summary.Usage.TotalTokens += embeddings.Usage.TotalTokens
		b.summary.Usage.ThinkingTokens += embeddings.Usage.ThinkingTokens
		b.summary.Usage.CacheReadInputTokens += embeddings.Usage.CacheReadInputTokens
		b.summary.Usage.CacheWriteInputTokens += embeddings.Usage.CacheWriteInputTokens
	}
	if embeddings.Cost != nil {
		if b.summary.Cost == nil {
			b.summary.Cost = &Cost{Currency: embeddings.Cost.Currency}
		}
		b.summary.Cost.InputCost += embeddings.Cost.InputCost
		b.summary.Cost.OutputCost += embeddings.Cost.OutputCost
		b.summary.Cost.CacheReadInputCost += embeddings.Cost.CacheReadInputCost
		b.summary.Cost.CacheWriteInputCost += embeddings.Cost.CacheWriteInputCost
		b.summary.Cost.TotalCost += embeddings.Cost.TotalCost
		if b.summary.Cost.Currency == "" {
			b.summary.Cost.Currency = embeddings.Cost.Currency
		}
	}
}

func (b *embeddingBatcher) addError(embeddings Embeddings) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.addAttemptsLocked(embeddings.Attempts, 1)
	b.summary.ErrorCount++
}

func (b *embeddingBatcher) addAttemptsLocked(attempts []EmbeddingAttempt, fallbackCount int) {
	if len(attempts) == 0 {
		b.summary.TotalRequestCount += fallbackCount
		return
	}
	b.summary.TotalRequestCount += len(attempts)
	b.summary.Attempts = append(b.summary.Attempts, attempts...)
	for _, attempt := range attempts {
		if attempt.StatusCode != 0 {
			if b.summary.StatusBuckets == nil {
				b.summary.StatusBuckets = make(map[int]int)
			}
			b.summary.StatusBuckets[attempt.StatusCode]++
		}
		if attempt.RequestID != "" {
			b.summary.RequestIDs = append(b.summary.RequestIDs, attempt.RequestID)
		}
	}
}

func (b *embeddingBatcher) batchSummary() EmbeddingBatchSummary {
	b.mu.Lock()
	defer b.mu.Unlock()
	summary := b.summary
	if summary.StatusBuckets != nil {
		summary.StatusBuckets = copyIntIntMap(summary.StatusBuckets)
	}
	summary.RequestIDs = append([]string(nil), summary.RequestIDs...)
	summary.Attempts = append([]EmbeddingAttempt(nil), summary.Attempts...)
	if summary.Usage != nil {
		usage := *summary.Usage
		summary.Usage = &usage
	}
	if summary.Cost != nil {
		cost := *summary.Cost
		summary.Cost = &cost
	}
	return summary
}

func copyIntIntMap(values map[int]int) map[int]int {
	if len(values) == 0 {
		return nil
	}
	copied := make(map[int]int, len(values))
	for key, value := range values {
		copied[key] = value
	}
	return copied
}

func inputsForJobs(jobs []embeddingBatchJob) []string {
	out := make([]string, 0, len(jobs))
	for _, job := range jobs {
		out = append(out, job.text)
	}
	return out
}

func indexesForJobs(jobs []embeddingBatchJob) []int {
	var out []int
	for _, job := range jobs {
		out = append(out, job.indexes...)
	}
	return out
}

func embeddingsForJobs(jobs []embeddingBatchJob, embeddings Embeddings) ([]Embedding, error) {
	if len(embeddings.Vectors) != len(jobs) {
		return nil, fmt.Errorf("embedding provider returned %d vectors for %d inputs", len(embeddings.Vectors), len(jobs))
	}
	ordered := make([]Embedding, len(jobs))
	seen := make([]bool, len(jobs))
	for _, embedding := range embeddings.Vectors {
		if embedding.Index < 0 || embedding.Index >= len(jobs) {
			return nil, fmt.Errorf("embedding provider returned vector index %d for %d inputs", embedding.Index, len(jobs))
		}
		ordered[embedding.Index] = Embedding{
			Index:  embedding.Index,
			Vector: append([]float32(nil), embedding.Vector...),
		}
		seen[embedding.Index] = true
	}
	for i, ok := range seen {
		if !ok {
			return nil, fmt.Errorf("embedding provider did not return vector index %d", i)
		}
	}
	return ordered, nil
}

func expandJobEmbeddings(jobs []embeddingBatchJob, embeddings []Embedding) []Embedding {
	out := make([]Embedding, 0, len(jobs))
	for i, job := range jobs {
		for _, index := range job.indexes {
			out = append(out, Embedding{
				Index:  index,
				Vector: append([]float32(nil), embeddings[i].Vector...),
			})
		}
	}
	return out
}

func appendEmbeddingSlices(left, right []Embedding) []Embedding {
	out := make([]Embedding, 0, len(left)+len(right))
	out = append(out, left...)
	out = append(out, right...)
	return out
}

func orderEmbeddingsByIndex(embeddings []Embedding) []Embedding {
	ordered := append([]Embedding(nil), embeddings...)
	sort.SliceStable(ordered, func(i, j int) bool {
		return ordered[i].Index < ordered[j].Index
	})
	return ordered
}

func splitEmbeddingInput(text string) []string {
	runes := []rune(text)
	if len(runes) < 2 {
		return nil
	}
	mid := len(runes) / 2
	return []string{string(runes[:mid]), string(runes[mid:])}
}

func weightedAverageEmbedding(vectors []Embedding, weights []int) []float32 {
	if len(vectors) == 0 {
		return nil
	}
	out := make([]float32, len(vectors[0].Vector))
	totalWeight := 0
	for i, embedding := range vectors {
		weight := 1
		if i < len(weights) && weights[i] > 0 {
			weight = weights[i]
		}
		totalWeight += weight
		for j := range out {
			if j < len(embedding.Vector) {
				out[j] += embedding.Vector[j] * float32(weight)
			}
		}
	}
	if totalWeight == 0 {
		return out
	}
	for i := range out {
		out[i] /= float32(totalWeight)
	}
	return out
}

func (c *Client) embeddingRequestOptions(opts []EmbeddingOption) Options {
	options := Options{
		HTTPClient: c.httpClient,
		Headers:    copyStringStringMap(c.defaultHeaders),
	}
	options = mergeOptions(options, c.defaultOptions)
	options = applyEmbeddingOptions(options, opts)
	clientResolver := c.authResolver
	if options.AuthResolver != nil {
		clientResolver = options.AuthResolver
	}
	options.AuthResolver = ChainAuthResolver{
		Client:            clientResolver,
		ProviderCallbacks: options.ProviderAuthResolvers,
	}
	return options
}

func applyEmbeddingOptions(options Options, opts []EmbeddingOption) Options {
	applied := cloneOptions(options)
	for _, opt := range opts {
		if opt != nil {
			opt(&applied)
		}
	}
	return applied
}

func embeddingOptionFromOption(opt Option) EmbeddingOption {
	return func(options *Options) {
		if opt != nil {
			opt(options)
		}
	}
}

func validateEmbeddingOptions(model EmbeddingModel, req EmbeddingRequest, options Options) error {
	if len(req.Inputs) == 0 {
		return invalidEmbeddingOptionsError(model, "embedding inputs are required")
	}
	for _, input := range req.Inputs {
		if strings.TrimSpace(input) == "" {
			return invalidEmbeddingOptionsError(model, "embedding inputs must not be empty")
		}
	}
	if req.Dimensions < 0 {
		return invalidEmbeddingOptionsError(model, "embedding dimensions must be non-negative")
	}
	if options.Timeout != nil && *options.Timeout < 0 {
		return invalidEmbeddingOptionsError(model, "timeout must be non-negative")
	}
	if options.MaxRetries != nil && *options.MaxRetries < 0 {
		return invalidEmbeddingOptionsError(model, "max retries must be non-negative")
	}
	if options.MaxRetryDelay != nil && *options.MaxRetryDelay < 0 {
		return invalidEmbeddingOptionsError(model, "max retry delay must be non-negative")
	}
	return nil
}

func invalidEmbeddingOptionsError(model EmbeddingModel, message string) error {
	return &Error{
		Code:     ErrorInvalidOptions,
		Message:  message,
		Provider: model.Provider,
		Model:    model.ID,
	}
}

func embeddingProviderNotFoundError(provider ProviderID, model ModelID) error {
	return &Error{
		Code:     ErrorProviderNotFound,
		Message:  "embedding provider is not registered",
		Provider: provider,
		Model:    model,
	}
}

func embeddingModelNotFoundError(provider ProviderID, model ModelID) error {
	return &Error{
		Code:     ErrorModelNotFound,
		Message:  "embedding model is not registered",
		Provider: provider,
		Model:    model,
	}
}

func embeddingAbortedError(err error) error {
	return &Error{
		Code:    ErrorAborted,
		Message: err.Error(),
		Err:     err,
	}
}

func finalEmbeddings(model EmbeddingModel, embeddings Embeddings) Embeddings {
	if embeddings.Model == "" {
		embeddings.Model = model.ID
	}
	if embeddings.Provider == "" {
		embeddings.Provider = model.Provider
	}
	if embeddings.Usage != nil && embeddings.Cost == nil {
		cost := CostForEmbeddingUsage(model, *embeddings.Usage)
		embeddings.Cost = &cost
	}
	return embeddings
}
