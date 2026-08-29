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
	"strconv"

	"github.com/cespare/xxhash/v2"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
)

// declaredIDKeys are the body fields that name a session across turns, in
// priority order. previous_response_id is deliberately absent: its value names
// the previous turn rather than the session, so it changes every turn and can
// only group turns once the response id it refers to is recorded, which the
// ResponseBody hook cannot see.
var declaredIDKeys = []string{"prompt_cache_key", "conversation"}

// maxDeclaredIDLen bounds the alias key a client can impose.
const maxDeclaredIDLen = 256

// declaredID returns the client's own name for this session, or "" if it
// declared none.
//
// The value never enters a chain hash. It is a lookup key for the chain last
// served under this name, which is what lets a session whose history lives
// server-side keep extending one lineage. Two byte-identical prompts carrying
// different session names still hash to the same chain and still match.
//
// The session-header form comes from the agent-identity plugin's request
// attribute rather than from a second header list here, so operators configure
// one place. Enabling that plugin is what turns on header-based aliasing.
func declaredID(req *fwksched.InferenceRequest) string {
	if id, ok := fwksched.ReadRequestAttribute[string](req, agentidentity.AgentIdentityKey); ok {
		if id = clampID(id); id != "" {
			return "agent\x00" + id
		}
	}
	m, ok := req.Body.Payload.(fwkrh.PayloadMap)
	if !ok {
		return ""
	}
	for _, key := range declaredIDKeys {
		s, _ := m[key].(string)
		if s = clampID(s); s != "" {
			return key + "\x00" + s
		}
	}
	return ""
}

func clampID(s string) string {
	if len(s) > maxDeclaredIDLen {
		// Truncating would collapse two sessions whose ids share a prefix onto
		// one lineage. The value is only ever a map key, so hash it instead.
		return strconv.FormatUint(xxhash.Sum64String(s), 16)
	}
	return s
}

// sharedPrefixLen returns how many leading hashes a and b have in common.
func sharedPrefixLen(a, b []uint64) int {
	n := min(len(a), len(b))
	for i := range n {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}
