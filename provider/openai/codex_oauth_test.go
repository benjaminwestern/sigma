// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package openai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/wintermi/sigma"
)

func TestLoginOpenAICodexDeviceCodeSuccess(t *testing.T) {
	t.Parallel()

	accessToken := codexTestJWT("acct_login")
	var deviceInfo CodexDeviceCodeInfo
	client := codexOAuthTestClient(t, func(r *http.Request) *http.Response {
		switch r.URL.String() {
		case codexOAuthDeviceUserCodeURL:
			assertCodexOAuthJSONBody(t, r, map[string]any{"client_id": codexOAuthClientID})
			return codexOAuthJSONResponse(http.StatusOK, map[string]any{
				"device_auth_id": "device-auth-id",
				"user_code":      "ABCD-1234",
				"interval":       "5",
			})
		case codexOAuthDeviceTokenURL:
			assertCodexOAuthJSONBody(t, r, map[string]any{
				"device_auth_id": "device-auth-id",
				"user_code":      "ABCD-1234",
			})
			return codexOAuthJSONResponse(http.StatusOK, map[string]any{
				"authorization_code": "oauth-code",
				"code_verifier":      "device-code-verifier",
			})
		case codexOAuthTokenURL:
			assertCodexOAuthFormBody(t, r, map[string]string{
				"grant_type":    "authorization_code",
				"client_id":     codexOAuthClientID,
				"code":          "oauth-code",
				"code_verifier": "device-code-verifier",
				"redirect_uri":  codexOAuthDeviceRedirectURI,
			})
			return codexOAuthJSONResponse(http.StatusOK, map[string]any{
				"access_token":  accessToken,
				"refresh_token": "refresh-token",
				"expires_in":    3600,
			})
		default:
			t.Fatalf("unexpected OAuth URL %q", r.URL.String())
			return nil
		}
	})

	credentials, err := LoginOpenAICodexDeviceCode(context.Background(), CodexDeviceCodeLoginOptions{
		HTTPClient:   client,
		OnDeviceCode: func(info CodexDeviceCodeInfo) { deviceInfo = info },
	})
	if err != nil {
		t.Fatalf("LoginOpenAICodexDeviceCode returned error: %v", err)
	}
	if got, want := credentials.AccessToken, accessToken; got != want {
		t.Fatalf("access token = %q, want %q", got, want)
	}
	if got, want := credentials.RefreshToken, "refresh-token"; got != want {
		t.Fatalf("refresh token = %q, want %q", got, want)
	}
	if got, want := credentials.AccountID, "acct_login"; got != want {
		t.Fatalf("account id = %q, want %q", got, want)
	}
	if credentials.Expiry.IsZero() {
		t.Fatal("expiry was zero")
	}
	if got, want := deviceInfo.UserCode, "ABCD-1234"; got != want {
		t.Fatalf("device user code = %q, want %q", got, want)
	}
	if got, want := deviceInfo.VerificationURI, codexOAuthDeviceVerificationURI; got != want {
		t.Fatalf("device verification uri = %q, want %q", got, want)
	}
	if got, want := deviceInfo.Interval, 5*time.Second; got != want {
		t.Fatalf("device interval = %s, want %s", got, want)
	}
}

func TestOpenAICodexDevicePollPendingAndSlowDown(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		status int
		body   any
		want   string
	}{
		{
			name:   "pending code",
			status: http.StatusForbidden,
			body: map[string]any{
				"error": map[string]any{"code": "deviceauth_authorization_pending"},
			},
			want: "pending",
		},
		{
			name:   "not found pending",
			status: http.StatusNotFound,
			body:   "not ready",
			want:   "pending",
		},
		{
			name:   "slow down",
			status: http.StatusBadRequest,
			body:   map[string]any{"error": "slow_down"},
			want:   "slow_down",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			client := codexOAuthTestClient(t, func(*http.Request) *http.Response {
				return codexOAuthJSONResponse(tt.status, tt.body)
			})

			result, err := pollOpenAICodexDeviceAuthOnce(context.Background(), client, codexDeviceAuthInfo{
				deviceAuthID: "device-auth-id",
				userCode:     "ABCD-1234",
			})
			if err != nil {
				t.Fatalf("pollOpenAICodexDeviceAuthOnce returned error: %v", err)
			}
			if got := result.status; got != tt.want {
				t.Fatalf("poll status = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestOpenAICodexDevicePollCancellation(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	client := codexOAuthTestClient(t, func(r *http.Request) *http.Response {
		return codexOAuthErrorResponse(r.Context().Err())
	})

	_, err := pollOpenAICodexDeviceAuthOnce(ctx, client, codexDeviceAuthInfo{
		deviceAuthID: "device-auth-id",
		userCode:     "ABCD-1234",
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
}

func TestOpenAICodexDevicePollTimeout(t *testing.T) {
	oldTimeout := codexOAuthDeviceTimeout
	codexOAuthDeviceTimeout = time.Millisecond
	t.Cleanup(func() { codexOAuthDeviceTimeout = oldTimeout })

	client := codexOAuthTestClient(t, func(*http.Request) *http.Response {
		return codexOAuthJSONResponse(http.StatusForbidden, map[string]any{
			"error": map[string]any{"code": "deviceauth_authorization_pending"},
		})
	})

	_, err := pollOpenAICodexDeviceAuth(context.Background(), client, codexDeviceAuthInfo{
		deviceAuthID: "device-auth-id",
		userCode:     "ABCD-1234",
		interval:     time.Millisecond,
	})
	if err == nil || !strings.Contains(err.Error(), "timed out") {
		t.Fatalf("error = %v, want timeout", err)
	}
}

func TestLoginOpenAICodexDeviceCodeInvalidResponse(t *testing.T) {
	t.Parallel()

	client := codexOAuthTestClient(t, func(*http.Request) *http.Response {
		return codexOAuthJSONResponse(http.StatusOK, map[string]any{"user_code": "ABCD-1234"})
	})

	_, err := LoginOpenAICodexDeviceCode(context.Background(), CodexDeviceCodeLoginOptions{HTTPClient: client})
	if err == nil || !strings.Contains(err.Error(), "missing fields") {
		t.Fatalf("error = %v, want missing fields", err)
	}
}

func TestLoginOpenAICodexDeviceCodeExchangeFailureRedactsBody(t *testing.T) {
	t.Parallel()

	const token = "secret-access-token"
	client := codexOAuthTestClient(t, func(r *http.Request) *http.Response {
		switch r.URL.String() {
		case codexOAuthDeviceUserCodeURL:
			return codexOAuthJSONResponse(http.StatusOK, map[string]any{
				"device_auth_id": "device-auth-id",
				"user_code":      "ABCD-1234",
				"interval":       0,
			})
		case codexOAuthDeviceTokenURL:
			return codexOAuthJSONResponse(http.StatusOK, map[string]any{
				"authorization_code": "oauth-code",
				"code_verifier":      "device-code-verifier",
			})
		case codexOAuthTokenURL:
			return codexOAuthJSONResponse(http.StatusUnauthorized, map[string]any{
				"error":        "invalid_grant",
				"access_token": token,
			})
		default:
			t.Fatalf("unexpected OAuth URL %q", r.URL.String())
			return nil
		}
	})

	_, err := LoginOpenAICodexDeviceCode(context.Background(), CodexDeviceCodeLoginOptions{HTTPClient: client})
	if err == nil {
		t.Fatal("LoginOpenAICodexDeviceCode returned nil error")
	}
	if strings.Contains(err.Error(), token) {
		t.Fatalf("error leaked token: %v", err)
	}
}

func TestRefreshOpenAICodexTokenSuccess(t *testing.T) {
	t.Parallel()

	accessToken := codexTestJWT("acct_refresh")
	client := codexOAuthTestClient(t, func(r *http.Request) *http.Response {
		assertCodexOAuthFormBody(t, r, map[string]string{
			"grant_type":    "refresh_token",
			"refresh_token": "refresh-token",
			"client_id":     codexOAuthClientID,
		})
		return codexOAuthJSONResponse(http.StatusOK, map[string]any{
			"access_token":  accessToken,
			"refresh_token": "new-refresh-token",
			"expires_in":    3600,
		})
	})

	credentials, err := RefreshOpenAICodexToken(context.Background(), "refresh-token", CodexOAuthTokenProviderOptions{HTTPClient: client})
	if err != nil {
		t.Fatalf("RefreshOpenAICodexToken returned error: %v", err)
	}
	if got, want := credentials.AccessToken, accessToken; got != want {
		t.Fatalf("access token = %q, want %q", got, want)
	}
	if got, want := credentials.RefreshToken, "new-refresh-token"; got != want {
		t.Fatalf("refresh token = %q, want %q", got, want)
	}
	if got, want := credentials.AccountID, "acct_refresh"; got != want {
		t.Fatalf("account id = %q, want %q", got, want)
	}
}

func TestRefreshOpenAICodexTokenFailureRedactsBody(t *testing.T) {
	t.Parallel()

	const refreshToken = "secret-refresh-token"
	client := codexOAuthTestClient(t, func(*http.Request) *http.Response {
		return codexOAuthJSONResponse(http.StatusUnauthorized, map[string]any{
			"error":         "invalid_grant",
			"refresh_token": refreshToken,
		})
	})

	_, err := RefreshOpenAICodexToken(context.Background(), refreshToken, CodexOAuthTokenProviderOptions{HTTPClient: client})
	if err == nil {
		t.Fatal("RefreshOpenAICodexToken returned nil error")
	}
	if strings.Contains(err.Error(), refreshToken) {
		t.Fatalf("error leaked token: %v", err)
	}
}

func TestCodexAccountIDFromToken(t *testing.T) {
	t.Parallel()

	accountID, err := codexAccountIDFromToken(codexTestJWT("acct_jwt"))
	if err != nil {
		t.Fatalf("codexAccountIDFromToken returned error: %v", err)
	}
	if got, want := accountID, "acct_jwt"; got != want {
		t.Fatalf("account id = %q, want %q", got, want)
	}

	_, err = codexAccountIDFromToken("not-a-jwt")
	if err == nil {
		t.Fatal("codexAccountIDFromToken returned nil error for invalid token")
	}
}

func TestCodexOAuthTokenProviderRefreshesAndCallsBack(t *testing.T) {
	t.Parallel()

	now := time.Date(2026, 5, 31, 12, 0, 0, 0, time.UTC)
	accessToken := codexTestJWT("acct_refreshed")
	client := codexOAuthTestClient(t, func(*http.Request) *http.Response {
		return codexOAuthJSONResponse(http.StatusOK, map[string]any{
			"access_token":  accessToken,
			"refresh_token": "new-refresh-token",
			"expires_in":    3600,
		})
	})
	var refreshed CodexOAuthCredentials
	provider := NewCodexOAuthTokenProvider(CodexOAuthCredentials{
		AccessToken:  codexTestJWT("acct_old"),
		RefreshToken: "old-refresh-token",
		Expiry:       now.Add(30 * time.Second),
		AccountID:    "acct_old",
	}, CodexOAuthTokenProviderOptions{
		HTTPClient:    client,
		Now:           func() time.Time { return now },
		RefreshBefore: time.Minute,
		OnRefresh: func(_ context.Context, credentials CodexOAuthCredentials) error {
			refreshed = credentials
			return nil
		},
	})

	credential, err := provider.Token(context.Background(), sigma.Model{ID: "codex", Provider: sigma.ProviderOpenAI}, sigma.Options{})
	if err != nil {
		t.Fatalf("Token returned error: %v", err)
	}
	if got, want := credential.Value, accessToken; got != want {
		t.Fatalf("credential value = %q, want %q", got, want)
	}
	if got, want := credential.Metadata[codexOAuthCredentialAccountID], "acct_refreshed"; got != want {
		t.Fatalf("credential account metadata = %v, want %q", got, want)
	}
	if got, want := refreshed.RefreshToken, "new-refresh-token"; got != want {
		t.Fatalf("refreshed token = %q, want %q", got, want)
	}
}

func TestCodexOAuthTokenProviderCallbackErrorDoesNotLeakTokens(t *testing.T) {
	t.Parallel()

	now := time.Date(2026, 5, 31, 12, 0, 0, 0, time.UTC)
	newToken := codexTestJWT("acct_refreshed")
	client := codexOAuthTestClient(t, func(*http.Request) *http.Response {
		return codexOAuthJSONResponse(http.StatusOK, map[string]any{
			"access_token":  newToken,
			"refresh_token": "new-refresh-token",
			"expires_in":    3600,
		})
	})
	provider := NewCodexOAuthTokenProvider(CodexOAuthCredentials{
		AccessToken:  codexTestJWT("acct_old"),
		RefreshToken: "old-refresh-token",
		Expiry:       now.Add(30 * time.Second),
		AccountID:    "acct_old",
	}, CodexOAuthTokenProviderOptions{
		HTTPClient:    client,
		Now:           func() time.Time { return now },
		RefreshBefore: time.Minute,
		OnRefresh: func(context.Context, CodexOAuthCredentials) error {
			return errors.New("failed to persist " + newToken)
		},
	})

	_, err := provider.Token(context.Background(), sigma.Model{ID: "codex", Provider: sigma.ProviderOpenAI}, sigma.Options{})
	if err == nil {
		t.Fatal("Token returned nil error")
	}
	if strings.Contains(err.Error(), newToken) {
		t.Fatalf("error leaked token: %v", err)
	}
}

func codexOAuthTestClient(t *testing.T, handler func(*http.Request) *http.Response) *http.Client {
	t.Helper()
	return &http.Client{Transport: codexOAuthRoundTripper(handler)}
}

type codexOAuthRoundTripper func(*http.Request) *http.Response

func (rt codexOAuthRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) {
	if err := r.Context().Err(); err != nil {
		return nil, err
	}
	resp := rt(r)
	if resp == nil {
		return nil, errors.New("missing test response")
	}
	return resp, nil
}

func codexOAuthJSONResponse(status int, body any) *http.Response {
	var data []byte
	switch typed := body.(type) {
	case string:
		data = []byte(typed)
	default:
		data, _ = json.Marshal(typed)
	}
	return &http.Response{
		StatusCode: status,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(string(data))),
	}
}

func codexOAuthErrorResponse(err error) *http.Response {
	if err == nil {
		err = errors.New("test error")
	}
	return &http.Response{
		StatusCode: 0,
		Body:       io.NopCloser(strings.NewReader(err.Error())),
	}
}

func assertCodexOAuthJSONBody(t *testing.T, r *http.Request, want map[string]any) {
	t.Helper()
	if got, want := r.Method, http.MethodPost; got != want {
		t.Fatalf("method = %q, want %q", got, want)
	}
	if got, want := r.Header.Get("Content-Type"), "application/json"; got != want {
		t.Fatalf("content type = %q, want %q", got, want)
	}
	var got map[string]any
	if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
		t.Fatalf("Decode request body returned error: %v", err)
	}
	for key, wantValue := range want {
		if gotValue := got[key]; gotValue != wantValue {
			t.Fatalf("request body[%q] = %v, want %v", key, gotValue, wantValue)
		}
	}
}

func assertCodexOAuthFormBody(t *testing.T, r *http.Request, want map[string]string) {
	t.Helper()
	if got, want := r.Method, http.MethodPost; got != want {
		t.Fatalf("method = %q, want %q", got, want)
	}
	if got, want := r.Header.Get("Content-Type"), "application/x-www-form-urlencoded"; got != want {
		t.Fatalf("content type = %q, want %q", got, want)
	}
	data, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("ReadAll request body returned error: %v", err)
	}
	values, err := url.ParseQuery(string(data))
	if err != nil {
		t.Fatalf("ParseQuery request body returned error: %v", err)
	}
	for key, wantValue := range want {
		if gotValue := values.Get(key); gotValue != wantValue {
			t.Fatalf("request form[%q] = %q, want %q", key, gotValue, wantValue)
		}
	}
}

func codexTestJWT(accountID string) string {
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"none"}`))
	payload, _ := json.Marshal(map[string]any{
		codexOAuthJWTClaimPath: map[string]string{
			codexOAuthCredentialChatGPTAcctID: accountID,
		},
	})
	return header + "." + base64.RawURLEncoding.EncodeToString(payload) + ".signature"
}
