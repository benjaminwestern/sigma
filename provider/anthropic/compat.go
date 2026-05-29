// Copyright (c) 2026 Matthew Winter
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.

package anthropic

import (
	"net/url"
	"strings"

	"github.com/wintermi/sigma"
)

// MessagesCompat describes Anthropic Messages compatibility differences for
// Anthropic-compatible endpoints. Leave fields at their zero value to use
// provider/base-URL detection, or set them with WithMessagesCompat for tests or
// custom routers.
type MessagesCompat struct {
	EagerToolInputStreaming bool
	LongCacheRetention      bool
	SessionAffinityHeaders  bool
	CacheControlOnTools     bool
	AdaptiveThinking        bool
}

type messagesCompat struct {
	eagerToolInputStreaming bool
	longCacheRetention      bool
	sessionAffinityHeaders  bool
	cacheControlOnTools     bool
	adaptiveThinking        bool
}

func anthropicMessagesCompat(model sigma.Model, baseURL string, override *MessagesCompat) messagesCompat {
	compat := detectedMessagesCompat(model.Provider, baseURL)
	if override == nil {
		return compat
	}
	compat.eagerToolInputStreaming = override.EagerToolInputStreaming
	compat.longCacheRetention = override.LongCacheRetention
	compat.sessionAffinityHeaders = override.SessionAffinityHeaders
	compat.cacheControlOnTools = override.CacheControlOnTools
	compat.adaptiveThinking = override.AdaptiveThinking
	return compat
}

func detectedMessagesCompat(provider sigma.ProviderID, baseURL string) messagesCompat {
	host := baseURLHost(baseURL)
	providerText := strings.ToLower(string(provider))

	switch {
	case provider == sigma.ProviderAnthropic || host == "api.anthropic.com":
		return messagesCompat{
			longCacheRetention:  true,
			cacheControlOnTools: true,
		}
	case provider == sigma.ProviderFireworks || strings.Contains(host, "fireworks.ai"):
		return messagesCompat{
			eagerToolInputStreaming: true,
			sessionAffinityHeaders:  true,
			adaptiveThinking:        true,
		}
	case provider == sigma.ProviderKimi || strings.Contains(providerText, "kimi") || strings.Contains(host, "moonshot") || strings.Contains(host, "kimi"):
		return messagesCompat{
			eagerToolInputStreaming: true,
			sessionAffinityHeaders:  true,
			adaptiveThinking:        true,
		}
	case provider == sigma.ProviderXiaomi || strings.Contains(host, "xiaomi"):
		return messagesCompat{
			eagerToolInputStreaming: true,
			sessionAffinityHeaders:  true,
			adaptiveThinking:        true,
		}
	default:
		return messagesCompat{}
	}
}

func baseURLHost(baseURL string) string {
	parsed, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil {
		return ""
	}
	return strings.ToLower(parsed.Hostname())
}
