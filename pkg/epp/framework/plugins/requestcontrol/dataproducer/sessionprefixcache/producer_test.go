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
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/preadmitter/agentidentity"
)

// dataKey mirrors how a scorer binds this producer: the key must be derived from
// the same ProducerType the factory is called with (R2-1).
var dataKey = attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(ProducerType).String()

func testHandle() plugin.Handle {
	return plugin.NewEppHandle(context.Background(), nil, plugin.WithMetricsRecorder(prometheus.NewRegistry()))
}

func newTestProducer(t *testing.T) *sessionPrefixCacheProducer {
	t.Helper()
	p, err := newProducer(context.Background(), ProducerType, defaultParameters, testHandle())
	require.NoError(t, err)
	return p
}

func testEndpoint(name string) fwksched.Endpoint {
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: name}},
		fwkdl.NewMetrics(), fwkdl.NewAttributes(),
	)
}

// chatReq builds a chat-completions request. When id is non-empty it is carried
// as the body prompt_cache_key so it seeds the chain root. Each string becomes a
// distinct user message.
func chatReq(id string, msgs ...string) *fwksched.InferenceRequest {
	messages := make([]fwkrh.Message, len(msgs))
	for i, m := range msgs {
		messages[i] = fwkrh.Message{Role: "user", Content: fwkrh.Content{Raw: m}}
	}
	body := &fwkrh.InferenceRequestBody{
		ChatCompletions: &fwkrh.ChatCompletionsRequest{Messages: messages},
	}
	if id != "" {
		body.Payload = fwkrh.PayloadMap{"prompt_cache_key": id}
	}
	return &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model",
		Body:        body,
	}
}

func schedTo(eps ...fwksched.Endpoint) *fwksched.SchedulingResult {
	return &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: eps},
		},
	}
}

func matchInfo(t *testing.T, ep fwksched.Endpoint) *attrprefix.PrefixCacheMatchInfo {
	t.Helper()
	v, ok := ep.Get(dataKey)
	require.True(t, ok, "endpoint must carry the prefix match attribute")
	return v.(*attrprefix.PrefixCacheMatchInfo)
}

// bigText returns an ASCII string long enough to yield complete 512-byte chunks.
func bigText(n int) string { return strings.Repeat("A", n) }

// TestProduce_AffinityAfterPreRequest is the happy path: once an endpoint has
// served a request, a follow-up with the same declared id and a shared byte
// prefix scores affinity to that endpoint and to no other.
func TestProduce_AffinityAfterPreRequest(t *testing.T) {
	p := newTestProducer(t)
	ep1, ep2 := testEndpoint("pod1"), testEndpoint("pod2")
	eps := []fwksched.Endpoint{ep1, ep2}

	// Turn 1: one message that fills a complete chunk. Cold index -> no match.
	turn1 := chatReq("sess-A", bigText(700))
	require.NoError(t, p.Produce(context.Background(), turn1, eps))
	turn1Total := matchInfo(t, ep1).TotalBlocks()
	require.Greater(t, turn1Total, 0, "turn 1 must yield at least one complete chunk")
	assert.Equal(t, 0, matchInfo(t, ep1).MatchBlocks())
	assert.Equal(t, 0, matchInfo(t, ep2).MatchBlocks())

	p.PreRequest(context.Background(), turn1, schedTo(ep1))
	p.wg.Wait()

	// Turn 2: same id, appended message pushes past the first chunk.
	turn2 := chatReq("sess-A", bigText(700), bigText(700))
	require.NoError(t, p.Produce(context.Background(), turn2, eps))
	assert.Equal(t, turn1Total, matchInfo(t, ep1).MatchBlocks(), "served endpoint keeps the shared prefix")
	assert.Greater(t, matchInfo(t, ep2).TotalBlocks(), turn1Total)
	assert.Equal(t, 0, matchInfo(t, ep2).MatchBlocks(), "unserved endpoint has no affinity")

	// Control: a new declared id reseeds the root, so no chunk matches.
	control := chatReq("sess-B", bigText(700), bigText(700))
	require.NoError(t, p.Produce(context.Background(), control, eps))
	assert.Equal(t, 0, matchInfo(t, ep1).MatchBlocks(), "different declared id must not borrow affinity")
}

// TestChunk_DivergentContentSameDeclaredId asserts a shared declared id does not
// grant a full-chain match when the content diverges: only the shared leading
// chunks match (R1-1).
func TestChunk_DivergentContentSameDeclaredId(t *testing.T) {
	p := newTestProducer(t)
	ep := testEndpoint("pod1")
	eps := []fwksched.Endpoint{ep}

	seed := chatReq("sess", bigText(700), strings.Repeat("B", 700))
	require.NoError(t, p.Produce(context.Background(), seed, eps))
	seedTotal := matchInfo(t, ep).TotalBlocks()
	require.Greater(t, seedTotal, 1, "need at least two chunks to prove partial match")
	p.PreRequest(context.Background(), seed, schedTo(ep))
	p.wg.Wait()

	// Same id and identical first chunk, divergent tail chunk.
	query := chatReq("sess", bigText(700), strings.Repeat("C", 700))
	require.NoError(t, p.Produce(context.Background(), query, eps))
	match := matchInfo(t, ep)
	assert.Equal(t, 1, match.MatchBlocks(), "only the shared leading chunk matches")
	assert.Less(t, match.MatchBlocks(), match.TotalBlocks())
}

// TestChunk_SharedFirstChunkDifferentContinuation asserts that, absent a declared
// id, two sessions sharing an identical >=512B first chunk match exactly one
// chunk (true template sharing), not the divergent tail (R1-1).
func TestChunk_SharedFirstChunkDifferentContinuation(t *testing.T) {
	p := newTestProducer(t)
	ep := testEndpoint("pod1")
	eps := []fwksched.Endpoint{ep}

	seed := chatReq("", bigText(700), strings.Repeat("B", 700))
	require.NoError(t, p.Produce(context.Background(), seed, eps))
	p.PreRequest(context.Background(), seed, schedTo(ep))
	p.wg.Wait()

	query := chatReq("", bigText(700), strings.Repeat("C", 700))
	require.NoError(t, p.Produce(context.Background(), query, eps))
	assert.Equal(t, 1, matchInfo(t, ep).MatchBlocks(), "shared template chunk matches across sessions")
}

// TestChunk_SubChunkGrowthNoSignal asserts sub-chunk content carries no affinity
// signal: growth below the chunk boundary yields no complete chunk, and the first
// complete chunk is stable once content crosses the boundary (R1-2).
func TestChunk_SubChunkGrowthNoSignal(t *testing.T) {
	p := newTestProducer(t)
	ep := testEndpoint("pod1")
	eps := []fwksched.Endpoint{ep}

	// Below one chunk: no complete chunk -> no attribute is published.
	short := chatReq("sess", "hello")
	require.NoError(t, p.Produce(context.Background(), short, eps))
	_, ok := ep.Get(dataKey)
	assert.False(t, ok, "sub-chunk content must not publish a match attribute")

	grown := chatReq("sess", "hello world")
	require.NoError(t, p.Produce(context.Background(), grown, eps))
	_, ok = ep.Get(dataKey)
	assert.False(t, ok, "still below one chunk -> still no attribute")

	// Cross the boundary: first complete chunk is stable across a longer resend.
	seed := chatReq("sess", bigText(700))
	require.NoError(t, p.Produce(context.Background(), seed, eps))
	require.Equal(t, 1, matchInfo(t, ep).TotalBlocks())
	p.PreRequest(context.Background(), seed, schedTo(ep))
	p.wg.Wait()

	resend := chatReq("sess", bigText(700), bigText(700))
	require.NoError(t, p.Produce(context.Background(), resend, eps))
	assert.Equal(t, 1, matchInfo(t, ep).MatchBlocks(), "first complete chunk stays stable")
}

// TestChunk_RoleAndBoundaryFraming asserts message role and boundary framing
// change the byte stream, so structurally different requests with equal plain
// text do not gain false affinity (R1-3).
func TestChunk_RoleAndBoundaryFraming(t *testing.T) {
	p := newTestProducer(t)
	ep := testEndpoint("pod1")
	eps := []fwksched.Endpoint{ep}

	partA, partB := strings.Repeat("a", 300), strings.Repeat("b", 300)

	single := chatReq("sess", partA+partB)
	require.NoError(t, p.Produce(context.Background(), single, eps))
	require.Greater(t, matchInfo(t, ep).TotalBlocks(), 0)
	p.PreRequest(context.Background(), single, schedTo(ep))
	p.wg.Wait()

	// Same concatenated text, but split across a user and an assistant message.
	split := chatReq("sess")
	split.Body.ChatCompletions.Messages = []fwkrh.Message{
		{Role: "user", Content: fwkrh.Content{Raw: partA}},
		{Role: "assistant", Content: fwkrh.Content{Raw: partB}},
	}
	require.NoError(t, p.Produce(context.Background(), split, eps))
	require.Greater(t, matchInfo(t, ep).TotalBlocks(), 0)
	assert.Equal(t, 0, matchInfo(t, ep).MatchBlocks(), "role/boundary framing must break affinity")
}

// TestChunk_AnthropicSystemAndBodies asserts the Anthropic System field feeds the
// chain, and that Responses and Conversations bodies each produce a chain
// (R1-4, R2-2).
func TestChunk_AnthropicSystemAndBodies(t *testing.T) {
	p := newTestProducer(t)

	withSystem := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{Messages: &fwkrh.MessagesRequest{
			System:   fwkrh.AnthropicContent{Raw: bigText(700)},
			Messages: []fwkrh.AnthropicMessage{{Role: "user", Content: fwkrh.AnthropicContent{Raw: "hi"}}},
		}},
	}
	withoutSystem := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{Messages: &fwkrh.MessagesRequest{
			Messages: []fwkrh.AnthropicMessage{{Role: "user", Content: fwkrh.AnthropicContent{Raw: "hi"}}},
		}},
	}
	sysChain := p.buildChain(withSystem)
	require.NotEmpty(t, sysChain, "Anthropic System must contribute to the chain")
	assert.NotEqual(t, sysChain, p.buildChain(withoutSystem), "removing System must change the chain")

	responses := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model",
		Body:        &fwkrh.InferenceRequestBody{Responses: &fwkrh.ResponsesRequest{Input: bigText(700)}},
	}
	assert.NotEmpty(t, p.buildChain(responses), "Responses body must yield a chain")

	conversations := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{Conversations: &fwkrh.ConversationsRequest{
			Items: []fwkrh.ConversationItem{{Type: "message", Role: "user", Content: bigText(700)}},
		}},
	}
	assert.NotEmpty(t, p.buildChain(conversations), "Conversations body must yield a chain")
}

// TestIdentity_Precedence asserts body prompt_cache_key wins over headers, each
// agent-identity header seeds when present, and absence yields no declared id so
// the caller falls back to a content-root seed, never a tenant key (R1-6, R2-7).
func TestIdentity_Precedence(t *testing.T) {
	p := newTestProducer(t)

	cases := []struct {
		name    string
		payload fwkrh.PayloadMap
		headers map[string]string
		wantID  string
		wantOK  bool
	}{
		{
			name:    "body key wins over header",
			payload: fwkrh.PayloadMap{"prompt_cache_key": "body-key"},
			headers: map[string]string{agentidentity.ClaudeCodeSessionHeader: "hdr"},
			wantID:  "body-key",
			wantOK:  true,
		},
		{name: "claude header", headers: map[string]string{agentidentity.ClaudeCodeSessionHeader: "claude"}, wantID: "claude", wantOK: true},
		{name: "opencode header", headers: map[string]string{agentidentity.OpenCodeSessionHeader: "opencode"}, wantID: "opencode", wantOK: true},
		{name: "codex header", headers: map[string]string{agentidentity.CodexSessionHeader: "codex"}, wantID: "codex", wantOK: true},
		{name: "codex legacy header", headers: map[string]string{agentidentity.CodexSessionHeaderLegacy: "codex-legacy"}, wantID: "codex-legacy", wantOK: true},
		{name: "tenant header ignored", headers: map[string]string{"x-tenant-id": "tenant-1"}, wantID: "", wantOK: false},
		{name: "none -> content root seed", wantID: "", wantOK: false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body := &fwkrh.InferenceRequestBody{ChatCompletions: &fwkrh.ChatCompletionsRequest{}}
			if tc.payload != nil {
				body.Payload = tc.payload
			}
			id, ok := declaredSessionID(body, tc.headers, p.sessionHeaders)
			assert.Equal(t, tc.wantOK, ok)
			assert.Equal(t, tc.wantID, id)
		})
	}
}

// TestResponseBody_RefinesDownwardOnLowerUsage asserts the index is provenance
// aware, not grow-only: a response reporting fewer cached tokens than the seeded
// estimate trims the estimated tail so later matches shrink to the confirmed
// prefix (R1-5, R2-3).
func TestResponseBody_RefinesDownwardOnLowerUsage(t *testing.T) {
	p := newTestProducer(t)
	ep := testEndpoint("pod1")
	eps := []fwksched.Endpoint{ep}

	req := chatReq("sess", bigText(1600)) // ~3 complete chunks
	require.NoError(t, p.Produce(context.Background(), req, eps))
	total := matchInfo(t, ep).TotalBlocks()
	require.GreaterOrEqual(t, total, 3)
	p.PreRequest(context.Background(), req, schedTo(ep))
	p.wg.Wait()

	// Before refinement the full estimated chain matches.
	probe := chatReq("sess", bigText(1600))
	require.NoError(t, p.Produce(context.Background(), probe, eps))
	require.Equal(t, total, matchInfo(t, ep).MatchBlocks())

	// The server reports only one chunk's worth of cached prompt.
	resp := &requestcontrol.Response{
		EndOfStream: true,
		Usage:       fwkrh.Usage{PromptTokens: p.blockSizeTokens},
	}
	p.ResponseBody(context.Background(), req, resp, ep.GetMetadata())

	after := chatReq("sess", bigText(1600))
	require.NoError(t, p.Produce(context.Background(), after, eps))
	assert.Equal(t, 1, matchInfo(t, ep).MatchBlocks(), "match refines down to the confirmed prefix")
}

// TestProduce_NoTokenizerDependency asserts the producer declares no consumed
// data (in particular no TokenizedPrompt), guaranteeing it is tokenizer-free,
// while still implementing the request-control hooks it needs (R1-6).
func TestProduce_NoTokenizerDependency(t *testing.T) {
	p := newTestProducer(t)

	_, isConsumer := any(p).(plugin.ConsumerPlugin)
	assert.False(t, isConsumer, "producer must not consume any data (no tokenizer dependency)")

	_, isProducer := any(p).(requestcontrol.DataProducer)
	assert.True(t, isProducer)
	_, isPreRequest := any(p).(requestcontrol.PreRequest)
	assert.True(t, isPreRequest)
	_, isResponseBody := any(p).(requestcontrol.ResponseBodyProcessor)
	assert.True(t, isResponseBody)
}

// TestProduce_NilAndEmptyGuards asserts nil requests, nil bodies, and empty
// message sets neither panic nor publish an attribute (R1-6).
func TestProduce_NilAndEmptyGuards(t *testing.T) {
	p := newTestProducer(t)
	ep := testEndpoint("pod1")
	eps := []fwksched.Endpoint{ep}

	assert.NoError(t, p.Produce(context.Background(), nil, eps))

	nilBody := &fwksched.InferenceRequest{RequestID: uuid.NewString(), TargetModel: "m"}
	assert.NoError(t, p.Produce(context.Background(), nilBody, eps))
	_, ok := ep.Get(dataKey)
	assert.False(t, ok, "nil body must not publish an attribute")

	empty := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "m",
		Body:        &fwkrh.InferenceRequestBody{ChatCompletions: &fwkrh.ChatCompletionsRequest{}},
	}
	assert.NoError(t, p.Produce(context.Background(), empty, eps))
	_, ok = ep.Get(dataKey)
	assert.False(t, ok, "empty messages must not publish an attribute")

	// Nil-guarded lifecycle hooks must not panic either.
	p.PreRequest(context.Background(), nilBody, schedTo(ep))
	p.ResponseBody(context.Background(), nilBody, &requestcontrol.Response{EndOfStream: true}, ep.GetMetadata())
}

// TestFactory_RejectsUnknownField pins the strict-decoding policy shared by the
// data producers.
func TestFactory_RejectsUnknownField(t *testing.T) {
	dec := plugin.StrictDecoder(json.RawMessage(`{"unknownField": "value"}`))
	_, err := Factory(ProducerType, dec, testHandle())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unknownField")
}
