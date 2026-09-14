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

package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

// TestHandleEC_Multimedia asserts that video_url, audio_url, and input_audio
// items flow through both EC connectors the same way image_url items do.
// mmTypes in connector_ec_common.go treats video_url / audio_url uniformly
// with image_url (URL-based, dedup-eligible), while input_audio is inline and
// never deduplicates. This table exercises those paths against handleECNIXL
// (threads encoder ec_transfer_params into the prefill body) and
// handleECSharedStorage (primer only — encoder responses are discarded).
//
// Inline audio never deduplicates (see fanoutEncoderPrimerDeduplication note),
// so two input_audio blocks always produce two encoder calls.
func TestHandleEC_Multimedia(t *testing.T) {
	tests := []struct {
		name         string
		handler      func(*Server, http.ResponseWriter, *http.Request, string, []string, reqcommon.APIType)
		items        []map[string]any
		wantECParams bool
		wantECLen    int
		wantEncCalls int32
	}{
		{
			name:    "nixl video",
			handler: (*Server).handleECNIXL,
			items: []map[string]any{
				videoURLItem("https://example.com/v1.mp4"),
				videoURLItem("https://example.com/v2.mp4"),
			},
			wantECParams: true,
			wantECLen:    2,
			wantEncCalls: 2,
		},
		{
			name:    "nixl audio_url",
			handler: (*Server).handleECNIXL,
			items: []map[string]any{
				audioURLItem("https://example.com/a1.mp3"),
				audioURLItem("https://example.com/a2.mp3"),
			},
			wantECParams: true,
			wantECLen:    2,
			wantEncCalls: 2,
		},
		{
			name:    "nixl input_audio",
			handler: (*Server).handleECNIXL,
			items: []map[string]any{
				inlineAudioItem("aaa"),
				inlineAudioItem("bbb"),
			},
			wantECParams: true,
			wantECLen:    2,
			wantEncCalls: 2,
		},
		{
			name:    "shared_storage video",
			handler: (*Server).handleECSharedStorage,
			items: []map[string]any{
				videoURLItem("https://example.com/v1.mp4"),
				videoURLItem("https://example.com/v2.mp4"),
			},
			wantECParams: false,
			wantEncCalls: 2,
		},
		{
			name:    "shared_storage audio_url",
			handler: (*Server).handleECSharedStorage,
			items: []map[string]any{
				audioURLItem("https://example.com/a1.mp3"),
				audioURLItem("https://example.com/a2.mp3"),
			},
			wantECParams: false,
			wantEncCalls: 2,
		},
		{
			name:    "shared_storage input_audio",
			handler: (*Server).handleECSharedStorage,
			items: []map[string]any{
				inlineAudioItem("aaa"),
				inlineAudioItem("bbb"),
			},
			wantECParams: false,
			wantEncCalls: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				seq          atomic.Int32
				encoderCalls atomic.Int32
			)
			encoderBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				encoderCalls.Add(1)
				i := seq.Add(1) - 1
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				// Always include ec_transfer_params. handleECSharedStorage discards
				// encoder responses, so returning it here is a harmless superset that
				// keeps the mock uniform across rows.
				_, _ = fmt.Fprintf(w, `{
					"choices": [{"message": {"content": ""}}],
					"ec_transfer_params": {"hash-%d": {"peer_host": "10.0.0.%d", "peer_port": 5500}}
				}`, i, i)
			}))
			defer encoderBackend.Close()

			encoderURL, err := url.Parse(encoderBackend.URL)
			assert.NoError(t, err)
			srv := NewProxy(Config{Port: "0", DecoderURL: encoderURL})
			srv.logger = log.Log

			var capturedBody []byte
			srv.handlePDConnector = func(_ http.ResponseWriter, r *http.Request, _ string, _ string, _ reqcommon.APIType) {
				buf, err := io.ReadAll(r.Body)
				assert.NoError(t, err)
				capturedBody = buf
			}

			reqBody, _ := json.Marshal(userMessageRequest(tt.items...))
			httpReq := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, io.NopCloser(bytes.NewReader(reqBody)))
			rw := httptest.NewRecorder()

			tt.handler(srv, rw, httpReq, "fake-prefiller:8000", []string{encoderURL.Host}, reqcommon.APITypeChatCompletions)

			assert.Equal(t, tt.wantEncCalls, encoderCalls.Load(), "unexpected encoder call count")
			if !assert.NotNil(t, capturedBody, "handlePDConnector should have been invoked") {
				return
			}
			var parsed map[string]any
			assert.NoError(t, json.Unmarshal(capturedBody, &parsed))

			ec, hasEC := parsed[requestFieldECTransferParams].(map[string]any)
			if tt.wantECParams {
				assert.True(t, hasEC, "prefill body should carry ec_transfer_params")
				assert.Len(t, ec, tt.wantECLen, "one entry per distinct multimodal item")
				for k, v := range ec {
					entry, ok := v.(map[string]any)
					assert.Truef(t, ok, "ec[%q] should be an object", k)
					assert.Containsf(t, entry, "peer_host", "ec[%q] should carry transfer metadata", k)
				}
			} else {
				_, present := parsed[requestFieldECTransferParams]
				assert.False(t, present, "shared_storage primer must not add ec_transfer_params to the prefill body")
			}
		})
	}
}

// TestHandleEC_APIInputs sends each API's request shape through the
// disaggregated route with encoder headers and asserts how many multimodal
// items reach the encoder before the P/D handoff.
func TestHandleEC_APIInputs(t *testing.T) {
	tests := []struct {
		name         string
		apiType      reqcommon.APIType
		path         string
		body         string
		wantEncCalls int32
	}{
		{
			name:         "chat image_url",
			apiType:      reqcommon.APITypeChatCompletions,
			path:         reqcommon.PathChatCompletions,
			body:         `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"describe"},{"type":"image_url","image_url":{"url":"https://example.com/a.jpg"}}]}],"max_tokens":800}`,
			wantEncCalls: 1,
		},
		{
			name:         "responses input_image with url string",
			apiType:      reqcommon.APITypeResponses,
			path:         reqcommon.PathResponses,
			body:         `{"model":"m","input":[{"role":"user","content":[{"type":"input_text","text":"describe"},{"type":"input_image","image_url":"https://example.com/a.jpg"}]}],"max_output_tokens":800}`,
			wantEncCalls: 1,
		},
		{
			name:         "responses input_image with url object",
			apiType:      reqcommon.APITypeResponses,
			path:         reqcommon.PathResponses,
			body:         `{"model":"m","input":[{"role":"user","content":[{"type":"input_image","image_url":{"url":"https://example.com/a.jpg"}}]}],"max_output_tokens":800}`,
			wantEncCalls: 1,
		},
		{
			name:         "responses duplicate input_image urls deduplicated",
			apiType:      reqcommon.APITypeResponses,
			path:         reqcommon.PathResponses,
			body:         `{"model":"m","input":[{"role":"user","content":[{"type":"input_image","image_url":"https://example.com/a.jpg"},{"type":"input_image","image_url":{"url":"https://example.com/a.jpg"}},{"type":"input_image","image_url":"https://example.com/b.jpg"}]}]}`,
			wantEncCalls: 2,
		},
		{
			name:         "responses images across input items",
			apiType:      reqcommon.APITypeResponses,
			path:         reqcommon.PathResponses,
			body:         `{"model":"m","input":["hi",{"role":"user","content":[{"type":"input_image","image_url":"https://example.com/a.jpg"}]},{"type":"function_call_output","call_id":"c","output":"x"},{"role":"user","content":[{"type":"input_image","image_url":"https://example.com/b.jpg"}]}]}`,
			wantEncCalls: 2,
		},
		{
			name:    "responses string input",
			apiType: reqcommon.APITypeResponses,
			path:    reqcommon.PathResponses,
			body:    `{"model":"m","input":"hello"}`,
		},
		{
			name:    "responses text-only input parts",
			apiType: reqcommon.APITypeResponses,
			path:    reqcommon.PathResponses,
			body:    `{"model":"m","input":[{"role":"user","content":[{"type":"input_text","text":"hello"}]}]}`,
		},
		{
			name:    "responses empty input",
			apiType: reqcommon.APITypeResponses,
			path:    reqcommon.PathResponses,
			body:    `{"model":"m","input":[]}`,
		},
		{
			name:    "responses without input",
			apiType: reqcommon.APITypeResponses,
			path:    reqcommon.PathResponses,
			body:    `{"model":"m","previous_response_id":"resp_1"}`,
		},
		{
			name:    "responses object input",
			apiType: reqcommon.APITypeResponses,
			path:    reqcommon.PathResponses,
			body:    `{"model":"m","input":{"type":"input_image","image_url":"https://example.com/a.jpg"}}`,
		},
		{
			name:    "generate token_ids",
			apiType: reqcommon.APITypeGenerate,
			path:    reqcommon.PathGenerate,
			body:    `{"model":"m","token_ids":[1,2],"sampling_params":{"max_tokens":800}}`,
		},
	}

	for _, connector := range []string{ECExampleConnector, ECConnectorNIXL} {
		t.Run(connector, func(t *testing.T) {
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					var encoderCalls atomic.Int32
					encoder := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						n := encoderCalls.Add(1)
						assert.Equal(t, reqcommon.PathChatCompletions, r.URL.Path)
						w.Header().Set("Content-Type", "application/json")
						_, _ = fmt.Fprintf(w, `{"choices":[{"message":{"content":""}}],"ec_transfer_params":{"hash-%d":{"peer_port":5500}}}`, n)
					}))
					defer encoder.Close()

					decodeURL, err := url.Parse("http://decoder:8000")
					require.NoError(t, err)
					srv := NewProxy(Config{Port: "0", DecoderURL: decodeURL, ECConnector: connector})
					srv.logger = log.Log
					srv.allowlistValidator = &AllowlistValidator{}
					var (
						pdCalled   bool
						gotAPIType reqcommon.APIType
					)
					srv.handlePDConnector = func(_ http.ResponseWriter, _ *http.Request, _ string, _ string, apiType reqcommon.APIType) {
						pdCalled = true
						gotAPIType = apiType
					}

					req := httptest.NewRequest(http.MethodPost, tt.path, strings.NewReader(tt.body))
					req.Header.Set(routing.PrefillEndpointHeader, "prefill:8000")
					req.Header.Set(routing.EncoderEndpointsHeader, strings.TrimPrefix(encoder.URL, "http://"))
					recorder := httptest.NewRecorder()
					srv.disaggregatedPrefillHandler(tt.apiType)(recorder, req)

					require.Equal(t, http.StatusOK, recorder.Code, recorder.Body.String())
					assert.Equal(t, tt.wantEncCalls, encoderCalls.Load(), "unexpected encoder call count")
					assert.True(t, pdCalled, "handlePDConnector should have been invoked")
					assert.Equal(t, tt.apiType, gotAPIType)
				})
			}
		})
	}
}
