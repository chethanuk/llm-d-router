/*
Copyright 2026 The llm-d Authors.

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

package steps

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

const (
	keyOrderBlockID  = "block_id"
	keyOrderTokenIDs = "token_ids"
	keyOrderKVParams = "kv_transfer_params"
	keyOrderChoices  = "choices"
	keyOrderMessage  = "message"
	keyOrderContent  = "content"
)

// keyOrderTools has properties in non-alphabetical order (zeta before alpha).
const keyOrderTools = `[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"string"}}}}}]`

const keyOrderClientBody = `{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":` + keyOrderTools + `}`

// TestSteps_PreserveClientNestedKeyOrder asserts that the bytes of a field no
// step modifies reach the upstream unchanged (llm-d-router#2621): re-marshaling
// the parsed map[string]any would sort nested keys alphabetically.
func TestSteps_PreserveClientNestedKeyOrder(t *testing.T) {
	tests := []struct {
		name string
		// run executes one step against upstreamURL with the client body.
		run func(t *testing.T, upstreamURL string, original []byte, parsed map[string]any) error
	}{
		{
			name: "decode",
			run: func(t *testing.T, url string, original []byte, parsed map[string]any) error {
				step, err := NewDecodeStep(gateway.New(config.GatewayConfig{Address: url}), map[string]any{})
				if err != nil {
					t.Fatal(err)
				}
				return step.Execute(context.Background(), &pipeline.RequestContext{
					RequestID: "r", OriginalPath: reqcommon.PathChatCompletions, Model: "m",
					KVTransferParams: map[string]any{keyOrderBlockID: "b"},
					OriginalBody:     original, Body: parsed, ResponseWriter: httptest.NewRecorder(),
				})
			},
		},
		{
			name: "conditional_decode",
			run: func(t *testing.T, url string, original []byte, parsed map[string]any) error {
				step, err := NewConditionalDecodeStep(gateway.New(config.GatewayConfig{Address: url}), nil)
				if err != nil {
					t.Fatal(err)
				}
				return step.Execute(context.Background(), &pipeline.RequestContext{
					RequestID: "r", OriginalPath: reqcommon.PathChatCompletions, Model: "m",
					TokenIDs:     []int{1, 2},
					OriginalBody: original, Body: parsed, ResponseWriter: httptest.NewRecorder(),
				})
			},
		},
		{
			name: "prefill",
			run: func(t *testing.T, url string, original []byte, parsed map[string]any) error {
				step, err := NewPrefillStep(gateway.New(config.GatewayConfig{Address: url}), map[string]any{})
				if err != nil {
					t.Fatal(err)
				}
				return step.Execute(context.Background(), &pipeline.RequestContext{
					RequestID: "r", OriginalPath: reqcommon.PathChatCompletions, Model: "m",
					TokenIDs:         []int{1, 2},
					KVTransferParams: map[string]any{},
					OriginalBody:     original, Body: parsed,
				})
			},
		},
		{
			name: "render",
			run: func(t *testing.T, url string, original []byte, parsed map[string]any) error {
				step, err := NewRenderStep(nil, map[string]any{})
				if err != nil {
					t.Fatal(err)
				}
				step.(*RenderStep).SetServiceAddress(url)
				return step.Execute(context.Background(), &pipeline.RequestContext{
					OriginalPath: reqcommon.PathChatCompletions, Model: "m",
					OriginalBody: original, Body: parsed,
				})
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var got map[string]json.RawMessage
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(body, &got)
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(map[string]any{
					keyOrderTokenIDs: []int{1, 2},
					keyOrderKVParams: map[string]any{keyOrderBlockID: "b"},
					keyOrderChoices:  []map[string]any{{keyOrderMessage: map[string]any{keyOrderContent: "ok"}}},
				})
			}))
			defer server.Close()

			var parsed map[string]any
			if err := json.Unmarshal([]byte(keyOrderClientBody), &parsed); err != nil {
				t.Fatal(err)
			}
			// Steps may legitimately end the pipeline (conditional_decode); only the wire body matters.
			_ = tc.run(t, server.URL, []byte(keyOrderClientBody), parsed)

			if got == nil {
				t.Fatal("upstream received no JSON body")
			}
			if string(got["tools"]) != keyOrderTools {
				t.Errorf("tools bytes reordered:\n got: %s\nwant: %s", got["tools"], keyOrderTools)
			}
		})
	}
}
