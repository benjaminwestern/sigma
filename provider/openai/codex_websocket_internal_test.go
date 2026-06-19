// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package openai

import (
	"net/url"
	"strings"
	"testing"
)

func TestCodexWebSocketProxyURL(t *testing.T) {
	tests := []struct {
		name      string
		target    string
		env       map[string]string
		wantProxy string
		wantErr   string
	}{
		{
			name:   "wss uses https proxy",
			target: "wss://api.openai.com/v1/responses",
			env: map[string]string{
				"HTTPS_PROXY": "http://proxy.example:8080",
			},
			wantProxy: "http://proxy.example:8080",
		},
		{
			name:   "ws uses http proxy",
			target: "ws://api.openai.com/v1/responses",
			env: map[string]string{
				"HTTP_PROXY": "http://proxy.example:8080",
			},
			wantProxy: "http://proxy.example:8080",
		},
		{
			name:   "no proxy suppresses exact host",
			target: "wss://api.openai.com/v1/responses",
			env: map[string]string{
				"HTTPS_PROXY": "http://proxy.example:8080",
				"NO_PROXY":    "api.openai.com",
			},
		},
		{
			name:   "unsupported proxy scheme",
			target: "wss://api.openai.com/v1/responses",
			env: map[string]string{
				"HTTPS_PROXY": "socks5://proxy.example:1080",
			},
			wantErr: "unsupported proxy protocol",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clearCodexProxyEnv(t)
			for key, value := range tt.env {
				t.Setenv(key, value)
			}
			parsed, err := url.Parse(tt.target)
			if err != nil {
				t.Fatalf("Parse returned error: %v", err)
			}

			got, err := codexWebSocketProxyURL(parsed)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("proxy error = %v, want containing %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("codexWebSocketProxyURL returned error: %v", err)
			}
			if tt.wantProxy == "" {
				if got != nil {
					t.Fatalf("proxy URL = %v, want nil", got)
				}
				return
			}
			if got == nil || got.String() != tt.wantProxy {
				t.Fatalf("proxy URL = %v, want %q", got, tt.wantProxy)
			}
		})
	}
}

func clearCodexProxyEnv(t *testing.T) {
	t.Helper()

	for _, key := range []string{
		"HTTP_PROXY",
		"HTTPS_PROXY",
		"NO_PROXY",
		"ALL_PROXY",
		"http_proxy",
		"https_proxy",
		"no_proxy",
		"all_proxy",
	} {
		t.Setenv(key, "")
	}
}
