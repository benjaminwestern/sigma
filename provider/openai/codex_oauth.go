// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/wintermi/sigma"
	"github.com/wintermi/sigma/internal/redact"
)

const (
	codexOAuthClientID                = "app_EMoamEEZ73f0CkXaXp7hrann"
	codexOAuthDeviceVerificationURI   = "https://auth.openai.com/codex/device"
	codexOAuthDeviceRedirectURI       = "https://auth.openai.com/deviceauth/callback"
	codexOAuthDefaultPollInterval     = 5 * time.Second
	codexOAuthMinimumPollInterval     = time.Second
	codexOAuthSlowDownPollIncrement   = 5 * time.Second
	codexOAuthDefaultRefreshBefore    = time.Minute
	codexOAuthCredentialAccountID     = "accountID"
	codexOAuthCredentialChatGPTAcctID = "chatgpt_account_id"
	codexOAuthJWTClaimPath            = "https://api.openai.com/auth"
)

var (
	codexOAuthTokenURL          = "https://auth.openai.com/oauth/token"
	codexOAuthDeviceUserCodeURL = "https://auth.openai.com/api/accounts/deviceauth/usercode"
	codexOAuthDeviceTokenURL    = "https://auth.openai.com/api/accounts/deviceauth/token"
	codexOAuthDeviceTimeout     = 15 * time.Minute
)

// CodexOAuthCredentials carries OpenAI Codex OAuth tokens. Callers own
// persistence; Sigma never stores these credentials.
type CodexOAuthCredentials struct {
	AccessToken  string
	RefreshToken string
	Expiry       time.Time
	AccountID    string
}

// CodexDeviceCodeInfo reports the user code and verification URL that should be
// shown to the caller during OpenAI Codex device-code login.
type CodexDeviceCodeInfo struct {
	UserCode        string
	VerificationURI string
	Interval        time.Duration
	ExpiresIn       time.Duration
}

// CodexDeviceCodeLoginOptions configures OpenAI Codex device-code login.
type CodexDeviceCodeLoginOptions struct {
	HTTPClient   *http.Client
	OnDeviceCode func(CodexDeviceCodeInfo)
}

// CodexOAuthTokenProviderOptions configures the OAuth token provider returned
// by NewCodexOAuthTokenProvider.
type CodexOAuthTokenProviderOptions struct {
	HTTPClient    *http.Client
	Now           func() time.Time
	RefreshBefore time.Duration
	OnRefresh     func(context.Context, CodexOAuthCredentials) error
}

type codexDeviceAuthInfo struct {
	deviceAuthID string
	userCode     string
	interval     time.Duration
}

type codexDeviceTokenSuccess struct {
	authorizationCode string
	codeVerifier      string
}

type codexOAuthToken struct {
	access  string
	refresh string
	expiry  time.Time
}

type codexOAuthTokenProvider struct {
	client        *http.Client
	now           func() time.Time
	refreshBefore time.Duration
	onRefresh     func(context.Context, CodexOAuthCredentials) error

	mu          sync.Mutex
	credentials CodexOAuthCredentials
}

// LoginOpenAICodexDeviceCode runs the OpenAI Codex device-code OAuth flow and
// returns credentials for caller-managed persistence.
func LoginOpenAICodexDeviceCode(ctx context.Context, opts CodexDeviceCodeLoginOptions) (CodexOAuthCredentials, error) {
	device, err := startOpenAICodexDeviceAuth(ctx, opts.HTTPClient)
	if err != nil {
		return CodexOAuthCredentials{}, err
	}
	if opts.OnDeviceCode != nil {
		opts.OnDeviceCode(CodexDeviceCodeInfo{
			UserCode:        device.userCode,
			VerificationURI: codexOAuthDeviceVerificationURI,
			Interval:        device.interval,
			ExpiresIn:       codexOAuthDeviceTimeout,
		})
	}

	code, err := pollOpenAICodexDeviceAuth(ctx, opts.HTTPClient, device)
	if err != nil {
		return CodexOAuthCredentials{}, err
	}
	token, err := exchangeOpenAICodexAuthorizationCode(ctx, opts.HTTPClient, code.authorizationCode, code.codeVerifier, codexOAuthDeviceRedirectURI)
	if err != nil {
		return CodexOAuthCredentials{}, err
	}
	return codexCredentialsFromToken(token)
}

// RefreshOpenAICodexToken refreshes OpenAI Codex OAuth credentials from a
// refresh token.
func RefreshOpenAICodexToken(ctx context.Context, refreshToken string, opts CodexOAuthTokenProviderOptions) (CodexOAuthCredentials, error) {
	token, err := refreshOpenAICodexAccessToken(ctx, opts.HTTPClient, refreshToken)
	if err != nil {
		return CodexOAuthCredentials{}, err
	}
	return codexCredentialsFromToken(token)
}

// NewCodexOAuthTokenProvider adapts caller-managed OpenAI Codex OAuth
// credentials to Sigma's OAuthTokenProvider interface. Refreshed credentials are
// kept in memory and passed to OnRefresh for caller persistence.
func NewCodexOAuthTokenProvider(credentials CodexOAuthCredentials, opts CodexOAuthTokenProviderOptions) sigma.OAuthTokenProvider {
	now := opts.Now
	if now == nil {
		now = time.Now
	}
	refreshBefore := opts.RefreshBefore
	if refreshBefore == 0 {
		refreshBefore = codexOAuthDefaultRefreshBefore
	}
	return &codexOAuthTokenProvider{
		client:        opts.HTTPClient,
		now:           now,
		refreshBefore: refreshBefore,
		onRefresh:     opts.OnRefresh,
		credentials:   credentials,
	}
}

func (p *codexOAuthTokenProvider) Token(ctx context.Context, model sigma.Model, _ sigma.Options) (sigma.Credential, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.credentials.AccessToken == "" {
		return sigma.Credential{}, &sigma.CredentialUnavailableError{
			Provider: model.Provider,
			Model:    model.ID,
			Sources:  []string{"openai-codex-oauth"},
		}
	}

	if err := p.refreshIfNeeded(ctx, model); err != nil {
		return sigma.Credential{}, err
	}

	accountID := p.credentials.AccountID
	if accountID == "" {
		var err error
		accountID, err = codexAccountIDFromToken(p.credentials.AccessToken)
		if err != nil {
			return sigma.Credential{}, err
		}
		p.credentials.AccountID = accountID
	}

	return sigma.Credential{
		Type:   sigma.CredentialTypeOAuthToken,
		Value:  p.credentials.AccessToken,
		Expiry: p.credentials.Expiry,
		Source: "openai-codex-oauth",
		Metadata: map[string]any{
			codexOAuthCredentialAccountID:     accountID,
			codexOAuthCredentialChatGPTAcctID: accountID,
		},
	}, nil
}

func (p *codexOAuthTokenProvider) refreshIfNeeded(ctx context.Context, model sigma.Model) error {
	if !p.shouldRefresh() {
		return nil
	}
	if p.credentials.RefreshToken == "" {
		return &sigma.CredentialUnavailableError{
			Provider: model.Provider,
			Model:    model.ID,
			Sources:  []string{"openai-codex-refresh-token"},
		}
	}
	refreshed, err := RefreshOpenAICodexToken(ctx, p.credentials.RefreshToken, CodexOAuthTokenProviderOptions{
		HTTPClient: p.client,
	})
	if err != nil {
		return err
	}
	p.credentials = refreshed
	if p.onRefresh == nil {
		return nil
	}
	if err := p.onRefresh(ctx, refreshed); err != nil {
		return errors.New("openai codex oauth: refresh callback failed")
	}
	return nil
}

func (p *codexOAuthTokenProvider) shouldRefresh() bool {
	if p.credentials.Expiry.IsZero() {
		return false
	}
	return !p.now().Add(p.refreshBefore).Before(p.credentials.Expiry)
}

func startOpenAICodexDeviceAuth(ctx context.Context, client *http.Client) (codexDeviceAuthInfo, error) {
	body, err := json.Marshal(map[string]string{"client_id": codexOAuthClientID})
	if err != nil {
		return codexDeviceAuthInfo{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, codexOAuthDeviceUserCodeURL, bytes.NewReader(body))
	if err != nil {
		return codexDeviceAuthInfo{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := codexHTTPClient(client).Do(req)
	if err != nil {
		return codexDeviceAuthInfo{}, contextOrError(ctx, fmt.Errorf("openai codex oauth: request device code: %w", err))
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return codexDeviceAuthInfo{}, contextOrError(ctx, fmt.Errorf("openai codex oauth: read device code response: %w", err))
	}
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return codexDeviceAuthInfo{}, fmt.Errorf("openai codex oauth: device code request failed (%d): %s", resp.StatusCode, redact.Preview(string(data), 1024))
	}

	var decoded struct {
		DeviceAuthID string `json:"device_auth_id"`
		UserCode     string `json:"user_code"`
		Interval     any    `json:"interval"`
	}
	if err := json.Unmarshal(data, &decoded); err != nil {
		return codexDeviceAuthInfo{}, fmt.Errorf("openai codex oauth: decode device code response: %w", err)
	}
	interval, err := codexPollInterval(decoded.Interval)
	if err != nil {
		return codexDeviceAuthInfo{}, err
	}
	if decoded.DeviceAuthID == "" || decoded.UserCode == "" {
		return codexDeviceAuthInfo{}, fmt.Errorf("openai codex oauth: device code response missing fields")
	}
	return codexDeviceAuthInfo{
		deviceAuthID: decoded.DeviceAuthID,
		userCode:     decoded.UserCode,
		interval:     interval,
	}, nil
}

func pollOpenAICodexDeviceAuth(ctx context.Context, client *http.Client, device codexDeviceAuthInfo) (codexDeviceTokenSuccess, error) {
	deadline := time.Now().Add(codexOAuthDeviceTimeout)
	interval := device.interval
	if interval <= 0 {
		interval = codexOAuthDefaultPollInterval
	}
	if interval < codexOAuthMinimumPollInterval {
		interval = codexOAuthMinimumPollInterval
	}
	var slowDowns int

	for !time.Now().After(deadline) {
		result, err := pollOpenAICodexDeviceAuthOnce(ctx, client, device)
		if err != nil {
			return codexDeviceTokenSuccess{}, err
		}
		switch result.status {
		case "complete":
			return result.value, nil
		case "slow_down":
			slowDowns++
			interval += codexOAuthSlowDownPollIncrement
		case "pending":
		default:
			return codexDeviceTokenSuccess{}, fmt.Errorf("openai codex oauth: device auth failed")
		}

		remaining := time.Until(deadline)
		if remaining <= 0 {
			break
		}
		if err := sleepContext(ctx, minDuration(interval, remaining)); err != nil {
			return codexDeviceTokenSuccess{}, err
		}
	}
	if slowDowns > 0 {
		return codexDeviceTokenSuccess{}, fmt.Errorf("openai codex oauth: device flow timed out after slow_down responses")
	}
	return codexDeviceTokenSuccess{}, fmt.Errorf("openai codex oauth: device flow timed out")
}

type codexDevicePollResult struct {
	status string
	value  codexDeviceTokenSuccess
}

func pollOpenAICodexDeviceAuthOnce(ctx context.Context, client *http.Client, device codexDeviceAuthInfo) (codexDevicePollResult, error) {
	body, err := json.Marshal(map[string]string{
		"device_auth_id": device.deviceAuthID,
		"user_code":      device.userCode,
	})
	if err != nil {
		return codexDevicePollResult{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, codexOAuthDeviceTokenURL, bytes.NewReader(body))
	if err != nil {
		return codexDevicePollResult{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := codexHTTPClient(client).Do(req)
	if err != nil {
		return codexDevicePollResult{}, contextOrError(ctx, fmt.Errorf("openai codex oauth: poll device auth: %w", err))
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return codexDevicePollResult{}, contextOrError(ctx, fmt.Errorf("openai codex oauth: read device auth response: %w", err))
	}
	if resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
		var decoded struct {
			AuthorizationCode string `json:"authorization_code"`
			CodeVerifier      string `json:"code_verifier"`
		}
		if err := json.Unmarshal(data, &decoded); err != nil {
			return codexDevicePollResult{}, fmt.Errorf("openai codex oauth: decode device auth response: %w", err)
		}
		if decoded.AuthorizationCode == "" || decoded.CodeVerifier == "" {
			return codexDevicePollResult{}, fmt.Errorf("openai codex oauth: device auth response missing fields")
		}
		return codexDevicePollResult{
			status: "complete",
			value: codexDeviceTokenSuccess{
				authorizationCode: decoded.AuthorizationCode,
				codeVerifier:      decoded.CodeVerifier,
			},
		}, nil
	}

	code := codexOAuthErrorCode(data)
	if resp.StatusCode == http.StatusForbidden || resp.StatusCode == http.StatusNotFound || code == "deviceauth_authorization_pending" {
		return codexDevicePollResult{status: "pending"}, nil
	}
	if code == "slow_down" {
		return codexDevicePollResult{status: "slow_down"}, nil
	}
	return codexDevicePollResult{}, fmt.Errorf("openai codex oauth: device auth failed (%d): %s", resp.StatusCode, redact.Preview(string(data), 1024))
}

func exchangeOpenAICodexAuthorizationCode(ctx context.Context, client *http.Client, code string, verifier string, redirectURI string) (codexOAuthToken, error) {
	form := url.Values{}
	form.Set("grant_type", "authorization_code")
	form.Set("client_id", codexOAuthClientID)
	form.Set("code", code)
	form.Set("code_verifier", verifier)
	form.Set("redirect_uri", redirectURI)
	return postOpenAICodexToken(ctx, client, form, "exchange")
}

func refreshOpenAICodexAccessToken(ctx context.Context, client *http.Client, refreshToken string) (codexOAuthToken, error) {
	if refreshToken == "" {
		return codexOAuthToken{}, &sigma.CredentialUnavailableError{
			Sources: []string{"openai-codex-refresh-token"},
		}
	}
	form := url.Values{}
	form.Set("grant_type", "refresh_token")
	form.Set("refresh_token", refreshToken)
	form.Set("client_id", codexOAuthClientID)
	return postOpenAICodexToken(ctx, client, form, "refresh")
}

func postOpenAICodexToken(ctx context.Context, client *http.Client, form url.Values, operation string) (codexOAuthToken, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, codexOAuthTokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return codexOAuthToken{}, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := codexHTTPClient(client).Do(req)
	if err != nil {
		return codexOAuthToken{}, contextOrError(ctx, fmt.Errorf("openai codex oauth: token %s: %w", operation, err))
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return codexOAuthToken{}, contextOrError(ctx, fmt.Errorf("openai codex oauth: read token %s response: %w", operation, err))
	}
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return codexOAuthToken{}, fmt.Errorf("openai codex oauth: token %s failed (%d): %s", operation, resp.StatusCode, redact.Preview(string(data), 1024))
	}

	var decoded struct {
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
		ExpiresIn    int64  `json:"expires_in"`
	}
	if err := json.Unmarshal(data, &decoded); err != nil {
		return codexOAuthToken{}, fmt.Errorf("openai codex oauth: decode token %s response: %w", operation, err)
	}
	if decoded.AccessToken == "" || decoded.RefreshToken == "" || decoded.ExpiresIn <= 0 {
		return codexOAuthToken{}, fmt.Errorf("openai codex oauth: token %s response missing fields", operation)
	}
	return codexOAuthToken{
		access:  decoded.AccessToken,
		refresh: decoded.RefreshToken,
		expiry:  time.Now().Add(time.Duration(decoded.ExpiresIn) * time.Second),
	}, nil
}

func codexCredentialsFromToken(token codexOAuthToken) (CodexOAuthCredentials, error) {
	accountID, err := codexAccountIDFromToken(token.access)
	if err != nil {
		return CodexOAuthCredentials{}, err
	}
	return CodexOAuthCredentials{
		AccessToken:  token.access,
		RefreshToken: token.refresh,
		Expiry:       token.expiry,
		AccountID:    accountID,
	}, nil
}

func codexAccountIDFromToken(token string) (string, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return "", fmt.Errorf("openai codex oauth: failed to extract account id from token")
	}
	payload, err := decodeJWTPayload(parts[1])
	if err != nil {
		return "", fmt.Errorf("openai codex oauth: failed to extract account id from token")
	}
	var decoded map[string]any
	if err := json.Unmarshal(payload, &decoded); err != nil {
		return "", fmt.Errorf("openai codex oauth: failed to extract account id from token")
	}
	claim, ok := decoded[codexOAuthJWTClaimPath].(map[string]any)
	if !ok {
		return "", fmt.Errorf("openai codex oauth: failed to extract account id from token")
	}
	accountID, ok := claim[codexOAuthCredentialChatGPTAcctID].(string)
	if !ok || accountID == "" {
		return "", fmt.Errorf("openai codex oauth: failed to extract account id from token")
	}
	return accountID, nil
}

func decodeJWTPayload(payload string) ([]byte, error) {
	encodings := []*base64.Encoding{
		base64.RawURLEncoding,
		base64.URLEncoding,
		base64.RawStdEncoding,
		base64.StdEncoding,
	}
	for _, encoding := range encodings {
		decoded, err := encoding.DecodeString(payload)
		if err == nil {
			return decoded, nil
		}
	}
	return nil, fmt.Errorf("invalid jwt payload")
}

func codexPollInterval(value any) (time.Duration, error) {
	switch typed := value.(type) {
	case nil:
		return codexOAuthDefaultPollInterval, nil
	case float64:
		if typed < 0 {
			return 0, fmt.Errorf("openai codex oauth: invalid device code interval")
		}
		return time.Duration(typed * float64(time.Second)), nil
	case string:
		seconds, err := strconv.ParseFloat(strings.TrimSpace(typed), 64)
		if err != nil || seconds < 0 {
			return 0, fmt.Errorf("openai codex oauth: invalid device code interval")
		}
		return time.Duration(seconds * float64(time.Second)), nil
	default:
		return 0, fmt.Errorf("openai codex oauth: invalid device code interval")
	}
}

func codexOAuthErrorCode(data []byte) string {
	var decoded struct {
		Error any `json:"error"`
	}
	if err := json.Unmarshal(data, &decoded); err != nil {
		return ""
	}
	switch typed := decoded.Error.(type) {
	case string:
		return typed
	case map[string]any:
		code, _ := typed["code"].(string)
		return code
	default:
		return ""
	}
}

func codexHTTPClient(client *http.Client) *http.Client {
	if client != nil {
		return client
	}
	return http.DefaultClient
}

func sleepContext(ctx context.Context, duration time.Duration) error {
	timer := time.NewTimer(duration)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func contextOrError(ctx context.Context, err error) error {
	if ctxErr := ctx.Err(); ctxErr != nil {
		return ctxErr
	}
	return err
}

func minDuration(a, b time.Duration) time.Duration {
	if a < b {
		return a
	}
	return b
}
