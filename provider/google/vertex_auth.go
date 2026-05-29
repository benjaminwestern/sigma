// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package google

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	"github.com/wintermi/sigma"
	"github.com/wintermi/sigma/internal/redact"
)

// VertexCredentialMode selects the Google Vertex AI authentication path.
type VertexCredentialMode string

const (
	// VertexCredentialAuto resolves a sigma credential first, then falls back to
	// the configured token provider when no API key or token is available.
	VertexCredentialAuto VertexCredentialMode = ""
	// VertexCredentialAPIKey requires an API-key credential.
	VertexCredentialAPIKey VertexCredentialMode = "api-key"
	// VertexCredentialToken requires an OAuth token credential.
	VertexCredentialToken VertexCredentialMode = "token"
)

func (p *VertexProvider) addAuthHeader(ctx context.Context, req *http.Request, model sigma.Model, opts sigma.Options, config vertexRequestConfig) error {
	credential, err := p.vertexCredential(ctx, model, opts, config)
	if err != nil {
		return err
	}
	if credential.Value == "" {
		return p.vertexCredentialUnavailable(model, credential.Source)
	}
	switch credential.Type {
	case sigma.CredentialTypeAPIKey:
		req.Header.Set("X-Goog-Api-Key", credential.Value)
	case sigma.CredentialTypeOAuthToken:
		req.Header.Set("Authorization", "Bearer "+credential.Value)
	default:
		return vertexInvalidOptions(model, fmt.Sprintf("google vertex: unsupported credential type %q", credential.Type), nil)
	}
	return nil
}

func (p *VertexProvider) vertexCredential(ctx context.Context, model sigma.Model, opts sigma.Options, config vertexRequestConfig) (sigma.Credential, error) {
	switch config.CredentialMode {
	case VertexCredentialToken:
		return p.vertexTokenCredential(ctx, model, opts)
	case VertexCredentialAPIKey:
		return p.vertexResolvedCredential(ctx, model, opts, sigma.CredentialTypeAPIKey)
	default:
		credential, err := p.vertexResolvedCredential(ctx, model, opts, "")
		if err == nil {
			return credential, nil
		}
		if !errors.Is(err, sigma.ErrCredentialUnavailable) {
			return sigma.Credential{}, err
		}
		if p.tokenProvider == nil {
			return sigma.Credential{}, err
		}
		return p.vertexTokenCredential(ctx, model, opts)
	}
}

func (p *VertexProvider) vertexResolvedCredential(ctx context.Context, model sigma.Model, opts sigma.Options, want sigma.CredentialType) (sigma.Credential, error) {
	if opts.AuthResolver == nil {
		return sigma.Credential{}, p.vertexCredentialUnavailable(model, "auth-resolver")
	}
	credential, err := opts.AuthResolver.Resolve(ctx, model, opts)
	if err != nil {
		if errors.Is(err, sigma.ErrCredentialUnavailable) {
			return sigma.Credential{}, err
		}
		return sigma.Credential{}, vertexAuthError(model, "google vertex: resolve credential: "+err.Error(), err)
	}
	if credential.Type == "" {
		credential.Type = sigma.CredentialTypeAPIKey
	}
	if want != "" && credential.Type != want {
		return sigma.Credential{}, vertexInvalidOptions(model, fmt.Sprintf("google vertex: credential mode %q requires %q credential, got %q", want, want, credential.Type), nil)
	}
	return credential, nil
}

func (p *VertexProvider) vertexTokenCredential(ctx context.Context, model sigma.Model, opts sigma.Options) (sigma.Credential, error) {
	if p.tokenProvider == nil {
		return p.vertexResolvedCredential(ctx, model, opts, sigma.CredentialTypeOAuthToken)
	}
	credential, err := p.tokenProvider.Token(ctx, model, opts)
	if err != nil {
		if errors.Is(err, sigma.ErrCredentialUnavailable) {
			return sigma.Credential{}, err
		}
		return sigma.Credential{}, vertexAuthError(model, "google vertex: resolve token: "+err.Error(), err)
	}
	if credential.Type == "" {
		credential.Type = sigma.CredentialTypeOAuthToken
	}
	if credential.Type != sigma.CredentialTypeOAuthToken {
		return sigma.Credential{}, vertexInvalidOptions(model, fmt.Sprintf("google vertex: token provider returned %q credential", credential.Type), nil)
	}
	return credential, nil
}

func (p *VertexProvider) vertexCredentialUnavailable(model sigma.Model, sources ...string) error {
	return &sigma.CredentialUnavailableError{
		Provider: model.Provider,
		Model:    model.ID,
		Sources:  sources,
	}
}

func vertexAuthError(model sigma.Model, message string, err error) error {
	return &sigma.Error{
		Code:     sigma.ErrorUnsupported,
		Message:  redact.String(message),
		Provider: model.Provider,
		Model:    model.ID,
		Err:      err,
	}
}
