/*
Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package sessionprefixcache

import (
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/preadmitter/agentidentity"
)

// defaultSessionHeaders is the fallback identity-header precedence used when a
// deployment does not configure its own list. It mirrors the agent-identity
// preadmitter so a session seeded there and here agree.
var defaultSessionHeaders = []string{
	agentidentity.ClaudeCodeSessionHeader,
	agentidentity.OpenCodeSessionHeader,
	agentidentity.CodexSessionHeader,
	agentidentity.CodexSessionHeaderLegacy,
}

// declaredSessionID extracts a stable session seed from the request body or
// headers. It returns the id and whether one was found. Precedence: the body
// prompt_cache_key (an explicit cache-routing instruction) wins over identity
// headers, which are tried in the given order. When nothing is declared the
// caller falls back to a content-derived root, never a tenant key.
func declaredSessionID(body *fwkrh.InferenceRequestBody, headers map[string]string, priorityHeaders []string) (string, bool) {
	if body == nil {
		return "", false
	}

	// 1. prompt_cache_key from the parsed body wins.
	if m, ok := payloadMap(body); ok {
		if val, present := m["prompt_cache_key"]; present {
			if s, isStr := val.(string); isStr && s != "" {
				return s, true
			}
		}
	}

	// 2. Identity headers, in precedence order.
	if len(priorityHeaders) == 0 {
		priorityHeaders = defaultSessionHeaders
	}
	for _, h := range priorityHeaders {
		if val := headers[h]; val != "" {
			return val, true
		}
	}

	return "", false
}

// payloadMap returns the parsed JSON map of the request body when one is
// available. It tolerates a nil Payload (raw or proto bodies never expose one).
func payloadMap(body *fwkrh.InferenceRequestBody) (fwkrh.PayloadMap, bool) {
	if body.Payload == nil {
		return nil, false
	}
	return body.Payload.AsMap()
}
