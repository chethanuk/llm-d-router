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
	"slices"
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
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
)

// dataKey mirrors how a scorer binds this producer: the key must be derived from
// the same ProducerType the factory is called with.
var dataKey = attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(ProducerType)

func testHandle() plugin.Handle {
	return plugin.NewEppHandle(context.Background(), nil, plugin.WithMetricsRecorder(prometheus.NewRegistry()))
}

func newTestProducer(t *testing.T) *sessionPrefixCacheProducer {
	t.Helper()
	p, err := newProducer(ProducerType, defaultParameters, testHandle())
	require.NoError(t, err)
	return p
}

func testEndpoint(name string) fwksched.Endpoint {
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: name}},
		fwkdl.NewMetrics(), fwkdl.NewAttributes(),
	)
}

// chatReq builds a chat-completions request. When id is non-empty it is carried
// as the body prompt_cache_key, which must not affect identity. Each string
// becomes a distinct user message.
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

// chainOf returns the chain the producer resolves for a request, dropping the
// alias key it also returns.
func chainOf(p *sessionPrefixCacheProducer, req *fwksched.InferenceRequest) []uint64 {
	chain, _ := p.resolveChain(req)
	return chain
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

	require.NoError(t, p.PreRequest(context.Background(), turn1, schedTo(ep1)))

	// Turn 2: same id, appended message pushes past the first chunk.
	turn2 := chatReq("sess-A", bigText(700), bigText(700))
	require.NoError(t, p.Produce(context.Background(), turn2, eps))
	assert.Equal(t, turn1Total, matchInfo(t, ep1).MatchBlocks(), "served endpoint keeps the shared prefix")
	assert.Greater(t, matchInfo(t, ep2).TotalBlocks(), turn1Total)
	assert.Equal(t, 0, matchInfo(t, ep2).MatchBlocks(), "unserved endpoint has no affinity")

	// Control: content, not the declared id, is the identity. The same bytes under
	// a different declared id match the same prefix.
	control := chatReq("sess-B", bigText(700), bigText(700))
	require.NoError(t, p.Produce(context.Background(), control, eps))
	assert.Equal(t, turn1Total, matchInfo(t, ep1).MatchBlocks(), "identity is content, not the declared id")
}

// TestChunk_DivergentContentMatchesSharedPrefixOnly asserts a chain match stops
// where the bytes diverge: two requests sharing a leading chunk match on that
// chunk alone, never on the whole chain.
func TestChunk_DivergentContentMatchesSharedPrefixOnly(t *testing.T) {
	p := newTestProducer(t)
	ep := testEndpoint("pod1")
	eps := []fwksched.Endpoint{ep}

	seed := chatReq("sess", bigText(700), strings.Repeat("B", 700))
	require.NoError(t, p.Produce(context.Background(), seed, eps))
	seedTotal := matchInfo(t, ep).TotalBlocks()
	require.Greater(t, seedTotal, 1, "need at least two chunks to prove partial match")
	require.NoError(t, p.PreRequest(context.Background(), seed, schedTo(ep)))

	// Identical first chunk, divergent tail chunk.
	query := chatReq("sess", bigText(700), strings.Repeat("C", 700))
	require.NoError(t, p.Produce(context.Background(), query, eps))
	match := matchInfo(t, ep)
	assert.Equal(t, 1, match.MatchBlocks(), "only the shared leading chunk matches")
	assert.Less(t, match.MatchBlocks(), match.TotalBlocks())
}

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
	require.NoError(t, p.PreRequest(context.Background(), seed, schedTo(ep)))

	resend := chatReq("sess", bigText(700), bigText(700))
	require.NoError(t, p.Produce(context.Background(), resend, eps))
	assert.Equal(t, 1, matchInfo(t, ep).MatchBlocks(), "first complete chunk stays stable")
}

// TestChunk_RoleAndBoundaryFraming asserts message role and boundary framing
// change the byte stream, so structurally different requests with equal plain
// text do not gain false affinity.
func TestChunk_RoleAndBoundaryFraming(t *testing.T) {
	p := newTestProducer(t)
	ep := testEndpoint("pod1")
	eps := []fwksched.Endpoint{ep}

	partA, partB := strings.Repeat("a", 300), strings.Repeat("b", 300)

	single := chatReq("sess", partA+partB)
	require.NoError(t, p.Produce(context.Background(), single, eps))
	require.Greater(t, matchInfo(t, ep).TotalBlocks(), 0)
	require.NoError(t, p.PreRequest(context.Background(), single, schedTo(ep)))

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
// chain, and that Responses and Conversations bodies each produce a chain.
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
	sysChain := chainOf(p, withSystem)
	require.NotEmpty(t, sysChain, "Anthropic System must contribute to the chain")
	plainChain := chainOf(p, withoutSystem)
	assert.NotEqual(t, sysChain, plainChain, "removing System must change the chain")

	responses := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model",
		Body:        &fwkrh.InferenceRequestBody{Responses: &fwkrh.ResponsesRequest{Input: bigText(700)}},
	}
	responsesChain := chainOf(p, responses)
	assert.NotEmpty(t, responsesChain, "Responses body must yield a chain")

	conversations := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{Conversations: &fwkrh.ConversationsRequest{
			Items: []fwkrh.ConversationItem{{Type: "message", Role: "user", Content: bigText(700)}},
		}},
	}
	chain := chainOf(p, conversations)
	assert.NotEmpty(t, chain, "Conversations body must yield a chain")
}

// TestIdentity_DeclaredIdsDoNotAffectIdentity asserts the central invariant:
// hash-prefix equality holds exactly when content prefixes are byte-identical.
// Declared client ids alias a lineage, they do not name it, so byte-identical
// content must resolve to one chain no matter who sent it or how they labelled
// it. Folding a declared id into the chain root would break template sharing,
// the traffic class this producer exists to serve.
func TestIdentity_DeclaredIdsDoNotAffectIdentity(t *testing.T) {
	p := newTestProducer(t)
	const shared = 700

	cases := []struct {
		name    string
		payload fwkrh.PayloadMap
		agentID string
	}{
		{name: "no declaration"},
		{name: "body prompt_cache_key", payload: fwkrh.PayloadMap{"prompt_cache_key": "thread-1"}},
		{name: "different prompt_cache_key", payload: fwkrh.PayloadMap{"prompt_cache_key": "thread-2"}},
		{name: "conversation id", payload: fwkrh.PayloadMap{"conversation": "conv-1"}},
		{name: "agent identity", agentID: "session-a"},
		{name: "different agent identity", agentID: "session-b"},
	}

	want := chainOf(p, chatReq("", bigText(shared)))
	require.NotEmpty(t, want)

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := chatReq("", bigText(shared))
			if tc.agentID != "" {
				req.PutAttribute(agentidentity.AgentIdentityKey, tc.agentID)
			}
			if tc.payload != nil {
				req.Body.Payload = tc.payload
			}
			got := chainOf(p, req)
			assert.Equal(t, want, got, "declared ids must not change the chain")
		})
	}
}

// TestIdentity_ModelAndSaltIsolate asserts the two inputs that legitimately seed
// the chain root do isolate it. The model because its KV is not portable, the
// cache salt because it is the client's explicit request for isolation. Both
// mirror how the tokenized producer and vLLM scope a prefix cache.
func TestIdentity_ModelAndSaltIsolate(t *testing.T) {
	p := newTestProducer(t)

	base := chatReq("", bigText(700))
	want := chainOf(p, base)
	require.NotEmpty(t, want)

	cases := []struct {
		name     string
		mutate   func(*fwksched.InferenceRequest)
		wantSame bool
	}{
		{name: "same request", mutate: func(*fwksched.InferenceRequest) {}, wantSame: true},
		{
			name:   "different model",
			mutate: func(r *fwksched.InferenceRequest) { r.TargetModel = "other-model" },
		},
		{
			name:   "different cache salt",
			mutate: func(r *fwksched.InferenceRequest) { r.Body.ChatCompletions.CacheSalt = "tenant-salt" },
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := chatReq("", bigText(700))
			tc.mutate(req)
			got := chainOf(p, req)
			if tc.wantSame {
				assert.Equal(t, want, got)
				return
			}
			assert.NotEqual(t, want, got, "root seed must isolate the chain")
		})
	}
}

// TestResponseBody_CalibratesBytesPerToken asserts the bytes-per-token ratio
// tracks engine-reported usage, so the block size reported to PrefixCacheMatchInfo
// consumers reflects the traffic instead of an assumption about English prose.
// The index itself is untouched: a pod that served a request holds the KV for all
// of it, so there is no over-estimated tail to walk back.
func TestResponseBody_CalibratesBytesPerToken(t *testing.T) {
	req := chatReq("", bigText(4096))
	stream, textOnly := contentStream(req.Body)
	require.True(t, textOnly)

	// atDefault is the token count that reproduces the starting ratio exactly.
	// Deriving it from the framed stream rather than from the text length keeps
	// the no-movement case honest: a count that merely lands near the ratio would
	// hold still because the EWMA truncates a small delta to zero, which would
	// prove nothing about calibration.
	atDefault := len(stream) * bytesPerTokenScale / initialBytesPerTokenQ

	cases := []struct {
		name   string
		tokens func(atDefault int) int
		// wantDirection is the expected move of blockSizeTokens away from the
		// starting point.
		wantDirection int
	}{
		// Dense tokenization (CJK in UTF-8) means a chunk of bytes is worth more
		// tokens than the starting ratio assumes.
		{name: "denser than default", tokens: func(d int) int { return d * 2 }, wantDirection: +1},
		{name: "at default", tokens: func(d int) int { return d }, wantDirection: 0},
		// Sparse tokenization (repetitive text the tokenizer collapses) means a
		// chunk of bytes is worth fewer tokens.
		{name: "sparser than default", tokens: func(d int) int { return d / 4 }, wantDirection: -1},
		{name: "usage absent", tokens: func(int) int { return 0 }, wantDirection: 0},
		{name: "usage negative", tokens: func(int) int { return -5 }, wantDirection: 0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := newTestProducer(t)
			ep := testEndpoint("pod1")
			before := p.blockSizeTokens()

			resp := &requestcontrol.Response{
				EndOfStream: true,
				Usage:       fwkrh.Usage{PromptTokens: tc.tokens(atDefault)},
			}
			// Repeat so the EWMA converges far enough to be visible.
			for range 40 {
				p.ResponseBody(context.Background(), req, resp, ep.GetMetadata())
			}

			after := p.blockSizeTokens()
			switch {
			case tc.wantDirection > 0:
				assert.Greater(t, after, before)
			case tc.wantDirection < 0:
				assert.Less(t, after, before)
			default:
				assert.Equal(t, before, after)
			}
			assert.Positive(t, after, "block size in tokens must stay positive")
		})
	}
}

// TestResponseBody_LeavesIndexIntact asserts a completed response neither adds to
// nor removes from the index: PreRequest owns seeding, and the served pod holds
// the whole prompt it just processed.
func TestResponseBody_LeavesIndexIntact(t *testing.T) {
	p := newTestProducer(t)
	ep := testEndpoint("pod1")
	eps := []fwksched.Endpoint{ep}

	req := chatReq("", bigText(1600))
	require.NoError(t, p.Produce(context.Background(), req, eps))
	total := matchInfo(t, ep).TotalBlocks()
	require.GreaterOrEqual(t, total, 3)
	require.NoError(t, p.PreRequest(context.Background(), req, schedTo(ep)))

	// A response reporting far fewer prompt tokens than the byte estimate implies
	// must not shrink the match: the pod prefilled and now holds every chunk.
	resp := &requestcontrol.Response{
		EndOfStream: true,
		Usage:       fwkrh.Usage{PromptTokens: 1},
	}
	p.ResponseBody(context.Background(), req, resp, ep.GetMetadata())

	probe := chatReq("", bigText(1600))
	require.NoError(t, p.Produce(context.Background(), probe, eps))
	assert.Equal(t, total, matchInfo(t, ep).MatchBlocks(), "reported usage must not evict served chunks")
}

// TestProduce_NoTokenizerDependency asserts the producer declares no consumed
// data (in particular no TokenizedPrompt), guaranteeing it is tokenizer-free,
// while still implementing the request-control hooks it needs.
func TestProduce_NoTokenizerDependency(t *testing.T) {
	p := newTestProducer(t)

	// The hook set itself is pinned by the compile-time assertions in producer.go.
	// What cannot be checked there is the absence of a dependency: consuming any
	// data key would put a token producer in front of this one.
	_, isConsumer := any(p).(plugin.ConsumerPlugin)
	assert.False(t, isConsumer, "producer must not consume any data (no tokenizer dependency)")
}

// TestProduce_NilAndEmptyGuards asserts nil requests, nil bodies, and empty
// message sets neither panic nor publish an attribute.
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
	require.NoError(t, p.PreRequest(context.Background(), nilBody, schedTo(ep)))
	p.ResponseBody(context.Background(), nilBody, &requestcontrol.Response{EndOfStream: true}, ep.GetMetadata())
}

// TestFactory_ParameterValidation pins the strict-decoding policy shared by the
// data producers and the bounds each parameter carries.
func TestFactory_ParameterValidation(t *testing.T) {
	cases := []struct {
		name    string
		params  string
		wantErr string
	}{
		{name: "defaults", params: `{}`},
		{name: "unknown field", params: `{"unknownField": "value"}`, wantErr: "unknownField"},
		{name: "zero maxChunks", params: `{"maxChunks": 0}`, wantErr: "maxChunks"},
		{name: "negative maxChunks", params: `{"maxChunks": -1}`, wantErr: "maxChunks"},
		{name: "zero chunk size", params: `{"chunkSizeBytes": 0}`, wantErr: "chunkSizeBytes"},
		{name: "zero entries per pod", params: `{"maxEntriesPerPod": 0}`, wantErr: "maxEntriesPerPod"},
		{name: "zero aliased sessions", params: `{"maxAliasedSessions": 0}`, wantErr: "maxAliasedSessions"},
		{name: "chain longer than a pod's LRU", params: `{"maxChunks": 200, "maxEntriesPerPod": 100}`, wantErr: "maxEntriesPerPod"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Factory(ProducerType, plugin.StrictDecoder(json.RawMessage(tc.params)), testHandle())
			if tc.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

// TestChunk_MaxChunksCapsTheChain asserts the default covers a whole prompt this
// size and that a lowered cap truncates it. A truncated chain reports its match
// ratio against the cap rather than against the prompt, which is why the default
// sits past any real context window.
func TestChunk_MaxChunksCapsTheChain(t *testing.T) {
	p := newTestProducer(t)

	capped := defaultParameters
	capped.MaxChunks = 2
	q, err := newProducer(ProducerType, capped, testHandle())
	require.NoError(t, err)

	req := chatReq("", bigText(512*10))
	assert.Len(t, chainOf(p, req), 10)
	assert.Len(t, chainOf(q, req), 2, "a cap truncates the chain")
}

// responsesReq builds a /v1/responses request carrying only the newest turn,
// which is how a client that lets the engine hold the conversation sends it.
func responsesReq(session, input string) *fwksched.InferenceRequest {
	body := &fwkrh.InferenceRequestBody{Responses: &fwkrh.ResponsesRequest{Input: input}}
	if session != "" {
		body.Payload = fwkrh.PayloadMap{"conversation": session}
	}
	return &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model",
		Body:        body,
	}
}

// TestAlias_ServerSideHistoryContinuesOneLineage covers the traffic the declared
// id exists for. On the Responses surface the engine holds the conversation, so
// each turn arrives carrying only its own input and shares no bytes with the
// previous one. Without the declared id every turn would start a fresh lineage
// and score no affinity at all.
func TestAlias_ServerSideHistoryContinuesOneLineage(t *testing.T) {
	p := newTestProducer(t)
	ep1, ep2 := testEndpoint("pod1"), testEndpoint("pod2")
	eps := []fwksched.Endpoint{ep1, ep2}

	turn1 := responsesReq("conv-1", bigText(700))
	require.NoError(t, p.Produce(context.Background(), turn1, eps))
	require.Equal(t, 1, matchInfo(t, ep1).TotalBlocks())
	assert.Equal(t, 0, matchInfo(t, ep1).MatchBlocks(), "cold index has nothing to match")
	require.NoError(t, p.PreRequest(context.Background(), turn1, schedTo(ep1)))

	turn2 := responsesReq("conv-1", strings.Repeat("B", 700))
	require.NoError(t, p.Produce(context.Background(), turn2, eps))
	assert.Equal(t, 1, matchInfo(t, ep1).MatchBlocks(), "turn 2 continues the lineage on the pod that served turn 1")
	assert.Equal(t, 2, matchInfo(t, ep1).TotalBlocks())
	assert.Equal(t, 0, matchInfo(t, ep2).MatchBlocks(), "an unserved endpoint gains nothing")
	require.NoError(t, p.PreRequest(context.Background(), turn2, schedTo(ep1)))

	turn3 := responsesReq("conv-1", strings.Repeat("C", 700))
	require.NoError(t, p.Produce(context.Background(), turn3, eps))
	assert.Equal(t, 2, matchInfo(t, ep1).MatchBlocks(), "the lineage keeps growing across turns")

	// A turn too small to fill a chunk still belongs to the session it declared.
	short := responsesReq("conv-1", "ok")
	require.NoError(t, p.Produce(context.Background(), short, eps))
	assert.Equal(t, 2, matchInfo(t, ep1).MatchBlocks(), "a sub-chunk turn still routes to its session")

	// The id keys the lineage; it does not seed the hash. The same first turn
	// under a different session name matches on content alone.
	other := responsesReq("conv-2", bigText(700))
	require.NoError(t, p.Produce(context.Background(), other, eps))
	assert.Equal(t, 1, matchInfo(t, ep1).MatchBlocks(), "identity stays content-derived")
}

// TestAlias_WithoutDeclaredIdEachTurnStartsFresh is the control for the test
// above: it is the declared id, not the surface, that carries the lineage.
func TestAlias_WithoutDeclaredIdEachTurnStartsFresh(t *testing.T) {
	p := newTestProducer(t)
	ep := testEndpoint("pod1")
	eps := []fwksched.Endpoint{ep}

	turn1 := responsesReq("", bigText(700))
	require.NoError(t, p.Produce(context.Background(), turn1, eps))
	require.NoError(t, p.PreRequest(context.Background(), turn1, schedTo(ep)))

	turn2 := responsesReq("", strings.Repeat("B", 700))
	require.NoError(t, p.Produce(context.Background(), turn2, eps))
	assert.Equal(t, 0, matchInfo(t, ep).MatchBlocks(), "an undeclared turn shares no bytes with the last one")
	assert.Equal(t, 1, matchInfo(t, ep).TotalBlocks())
}

// TestAlias_ScopedByModelAndCacheSalt asserts a session name reused under a
// different model or cache salt cannot inherit the other lineage. The salt is
// the client's request for cache isolation; an alias that ignored it would route
// across the boundary it draws.
func TestAlias_ScopedByModelAndCacheSalt(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*fwksched.InferenceRequest)
	}{
		{name: "same scope", mutate: func(*fwksched.InferenceRequest) {}},
		{name: "different model", mutate: func(r *fwksched.InferenceRequest) { r.TargetModel = "other-model" }},
		{name: "different cache salt", mutate: func(r *fwksched.InferenceRequest) { r.Body.Responses.CacheSalt = "tenant-b" }},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := newTestProducer(t)
			ep := testEndpoint("pod1")
			eps := []fwksched.Endpoint{ep}

			seed := responsesReq("shared-name", bigText(700))
			require.NoError(t, p.Produce(context.Background(), seed, eps))
			require.NoError(t, p.PreRequest(context.Background(), seed, schedTo(ep)))

			next := responsesReq("shared-name", strings.Repeat("B", 700))
			tc.mutate(next)
			require.NoError(t, p.Produce(context.Background(), next, eps))

			if tc.name == "same scope" {
				assert.Equal(t, 2, matchInfo(t, ep).TotalBlocks(), "the lineage continues within one scope")
				return
			}
			assert.Equal(t, 1, matchInfo(t, ep).TotalBlocks(), "a new scope starts its own lineage")
			assert.Equal(t, 0, matchInfo(t, ep).MatchBlocks())
		})
	}
}

// TestAlias_ResentHistoryIgnoresTheAlias asserts the alias stays out of the way
// on the surfaces where the client sends the whole prompt every turn. There the
// content chain is what was actually sent, so a turn that edits earlier history
// must lose the affinity it no longer shares rather than inherit it from the id.
func TestAlias_ResentHistoryIgnoresTheAlias(t *testing.T) {
	p := newTestProducer(t)
	ep := testEndpoint("pod1")
	eps := []fwksched.Endpoint{ep}

	seed := chatReq("sess-A", bigText(700), strings.Repeat("B", 700))
	require.NoError(t, p.Produce(context.Background(), seed, eps))
	require.NoError(t, p.PreRequest(context.Background(), seed, schedTo(ep)))

	edited := chatReq("sess-A", strings.Repeat("Z", 700), strings.Repeat("B", 700))
	require.NoError(t, p.Produce(context.Background(), edited, eps))
	assert.Equal(t, 0, matchInfo(t, ep).MatchBlocks(), "edited history must not inherit affinity from the declared id")
	assert.Equal(t, 2, matchInfo(t, ep).TotalBlocks(), "the chain covers the prompt that was sent, nothing more")
}

// TestChunk_OpaqueContentBreaksFalseAffinity covers the agentic case. Tool
// definitions, the tool-call trace, and attached media are all prompt the engine
// tokenizes. Two sessions sharing a system prompt but diverging in any of them
// must not collapse onto one chain: that would report a full match on a pod
// whose KV diverged thousands of tokens earlier.
func TestChunk_OpaqueContentBreaksFalseAffinity(t *testing.T) {
	base := func() *fwkrh.MessagesRequest {
		return &fwkrh.MessagesRequest{
			System: fwkrh.AnthropicContent{Raw: bigText(700)},
			Messages: []fwkrh.AnthropicMessage{{
				Role: "assistant",
				Content: fwkrh.AnthropicContent{Structured: []fwkrh.AnthropicContentBlock{
					{Type: "thinking", Thinking: strings.Repeat("t", 600)},
					{Type: "tool_use", Name: "read_file", Input: json.RawMessage(`{"path":"` + strings.Repeat("p", 600) + `"}`)},
					{Type: "tool_result", ToolUseID: "tu-1", Content: fwkrh.AnthropicContent{Raw: strings.Repeat("r", 600)}},
					{Type: "image", Source: &fwkrh.AnthropicImageSource{Type: "base64", MediaType: "image/png", Data: strings.Repeat("d", 600)}},
					// Trailing text so every block above is followed by enough
					// bytes to complete its chunk. A change confined to the
					// trailing partial chunk carries no signal by design.
					{Type: "text", Text: strings.Repeat("x", 600)},
				}},
			}},
		}
	}

	cases := []struct {
		name     string
		mutate   func(*fwkrh.MessagesRequest)
		wantSame bool
	}{
		{name: "identical bodies", mutate: func(*fwkrh.MessagesRequest) {}, wantSame: true},
		{
			name:   "tool definitions added",
			mutate: func(m *fwkrh.MessagesRequest) { m.Tools = []fwkrh.AnthropicTool{{Name: "bash"}} },
		},
		{
			name:   "different tool called",
			mutate: func(m *fwkrh.MessagesRequest) { m.Messages[0].Content.Structured[1].Name = "write_file" },
		},
		{
			name: "different tool arguments",
			mutate: func(m *fwkrh.MessagesRequest) {
				m.Messages[0].Content.Structured[1].Input = json.RawMessage(`{"path":"` + strings.Repeat("q", 600) + `"}`)
			},
		},
		{
			name: "different tool result",
			mutate: func(m *fwkrh.MessagesRequest) {
				m.Messages[0].Content.Structured[2].Content = fwkrh.AnthropicContent{Raw: strings.Repeat("s", 600)}
			},
		},
		{
			name: "different reasoning replay",
			mutate: func(m *fwkrh.MessagesRequest) {
				m.Messages[0].Content.Structured[0].Thinking = strings.Repeat("u", 600)
			},
		},
		{
			name: "different image",
			mutate: func(m *fwkrh.MessagesRequest) {
				m.Messages[0].Content.Structured[3].Source.Data = strings.Repeat("e", 600)
			},
		},
	}

	p := newTestProducer(t)
	req := func(m *fwkrh.MessagesRequest) *fwksched.InferenceRequest {
		return &fwksched.InferenceRequest{
			RequestID:   uuid.NewString(),
			TargetModel: "test-model",
			Body:        &fwkrh.InferenceRequestBody{Messages: m},
		}
	}
	want := chainOf(p, req(base()))
	require.Greater(t, len(want), 1)

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := base()
			tc.mutate(m)
			got := chainOf(p, req(m))
			if tc.wantSame {
				assert.Equal(t, want, got)
				return
			}
			assert.NotEqual(t, want, got, "divergent prompt content must not share a chain")
		})
	}
}

// TestResponseBody_SkipsUnmeasurableSamples asserts calibration only counts a
// reported token count that measures the same content the router hashed.
func TestResponseBody_SkipsUnmeasurableSamples(t *testing.T) {
	cases := []struct {
		name string
		body *fwkrh.InferenceRequestBody
	}{
		{
			// Anthropic reports input_tokens net of cached and cache-creation
			// blocks, so on a cache hit it counts a fraction of the prompt and
			// would drag the ratio to its floor.
			name: "anthropic reports tokens net of cache",
			body: &fwkrh.InferenceRequestBody{Messages: &fwkrh.MessagesRequest{
				System: fwkrh.AnthropicContent{Raw: bigText(4096)},
			}},
		},
		{
			// The engine reports tokens for the conversation it holds, while the
			// router framed only the turn the client sent.
			name: "server-side history counts turns the router never framed",
			body: &fwkrh.InferenceRequestBody{Responses: &fwkrh.ResponsesRequest{Input: bigText(4096)}},
		},
		{
			// An image is named by digest, not carried, so the stream length is
			// not the number of bytes the engine tokenized.
			name: "multimodal content is named, not carried",
			body: &fwkrh.InferenceRequestBody{ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{{Role: "user", Content: fwkrh.Content{Structured: []fwkrh.ContentBlock{
					{Type: "text", Text: bigText(4096)},
					{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: "https://example.test/a.png"}},
				}}}},
			}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := newTestProducer(t)
			ep := testEndpoint("pod1")
			before := p.blockSizeTokens()

			req := &fwksched.InferenceRequest{
				RequestID:   uuid.NewString(),
				TargetModel: "test-model",
				Body:        tc.body,
			}
			// A count this far below the byte length would move the ratio hard.
			resp := &requestcontrol.Response{
				EndOfStream: true,
				Usage:       fwkrh.Usage{PromptTokens: 8},
			}
			for range 40 {
				p.ResponseBody(context.Background(), req, resp, ep.GetMetadata())
			}
			assert.Equal(t, before, p.blockSizeTokens(), "an unmeasurable sample must not move the ratio")
		})
	}
}

// TestChunk_BoundariesSurviveAppend pins the invariant the whole design rests
// on: appending to a stream never moves an earlier chunk boundary, so a hash
// match at position i really does prove the byte prefix through chunk i is
// identical. Raw tool arguments reach the stream as wire bytes that need not be
// valid UTF-8, and a boundary decision allowed to look at end-of-stream would
// split those one way alone and another way with the next turn appended.
func TestChunk_BoundariesSurviveAppend(t *testing.T) {
	const seed, chunkSize, maxChunks = 42, 512, 8

	// 512 bytes ending mid-rune: the bytes that decide the boundary are not all
	// present, so no chunk is emitted yet.
	head := append([]byte(strings.Repeat("a", chunkSize-1)), 0xC3)
	assert.Empty(t, chunkChain(head, seed, chunkSize, maxChunks),
		"a stream without the boundary reserve yields no chunk")

	grown := append(slices.Clone(head), 0x80, 0xA9, 'z')
	first := chunkChain(grown, seed, chunkSize, maxChunks)
	require.Len(t, first, 1)

	longer := chunkChain(append(slices.Clone(grown), []byte(strings.Repeat("z", 4096))...), seed, chunkSize, maxChunks)
	require.Greater(t, len(longer), len(first))
	assert.Equal(t, first, longer[:len(first)], "an earlier boundary must not move as the stream grows")
}

// TestAlias_RepeatedTurnDoesNotResetTheLineage asserts a turn whose content
// hashes to a prefix of its own session extends that session rather than
// replacing it with the shorter chain. A user repeating an earlier question is
// not the client resending history, and treating it as one would discard every
// turn the session had accumulated.
func TestAlias_RepeatedTurnDoesNotResetTheLineage(t *testing.T) {
	p := newTestProducer(t)
	ep := testEndpoint("pod1")
	eps := []fwksched.Endpoint{ep}

	turn1 := responsesReq("conv-1", bigText(700))
	require.NoError(t, p.Produce(context.Background(), turn1, eps))
	require.NoError(t, p.PreRequest(context.Background(), turn1, schedTo(ep)))

	turn2 := responsesReq("conv-1", strings.Repeat("B", 700))
	require.NoError(t, p.Produce(context.Background(), turn2, eps))
	require.Equal(t, 2, matchInfo(t, ep).TotalBlocks())
	require.NoError(t, p.PreRequest(context.Background(), turn2, schedTo(ep)))

	repeat := responsesReq("conv-1", bigText(700))
	require.NoError(t, p.Produce(context.Background(), repeat, eps))
	assert.Equal(t, 3, matchInfo(t, ep).TotalBlocks(), "a repeated turn extends the lineage")
	assert.Equal(t, 2, matchInfo(t, ep).MatchBlocks(), "and keeps the affinity the session accumulated")
}

// TestAlias_ChangedPreambleStartsANewLineage asserts a declared session whose
// instructions change does not inherit the affinity earned by the turns behind
// them. Agent clients rewrite instructions on every turn with the working
// directory, the tool list and the time; once that preamble moves, the engine
// holds no KV for the conversation as the client is now sending it, so claiming
// the remembered lineage would report a full prefix match against a pod that
// can serve none of it.
func TestAlias_ChangedPreambleStartsANewLineage(t *testing.T) {
	p := newTestProducer(t)
	ep := testEndpoint("pod1")
	eps := []fwksched.Endpoint{ep}

	turn := func(instructions, input string) *fwksched.InferenceRequest {
		return &fwksched.InferenceRequest{
			RequestID:   uuid.NewString(),
			TargetModel: "test-model",
			Body: &fwkrh.InferenceRequestBody{
				Responses: &fwkrh.ResponsesRequest{Instructions: instructions, Input: input},
				Payload:   fwkrh.PayloadMap{"conversation": "conv-1"},
			},
		}
	}
	serve := func(req *fwksched.InferenceRequest) {
		t.Helper()
		require.NoError(t, p.Produce(context.Background(), req, eps))
		require.NoError(t, p.PreRequest(context.Background(), req, schedTo(ep)))
	}

	stable := bigText(700)
	serve(turn(stable, strings.Repeat("B", 700)))

	// Same preamble: the second turn continues the lineage and lands on the pod
	// that served the first.
	second := turn(stable, strings.Repeat("C", 700))
	require.NoError(t, p.Produce(context.Background(), second, eps))
	assert.Positive(t, matchInfo(t, ep).MatchBlocks(), "a stable preamble keeps the lineage")
	require.NoError(t, p.PreRequest(context.Background(), second, schedTo(ep)))

	// The preamble moves: nothing recorded behind it is reusable.
	third := turn(strings.Repeat("Z", 700), strings.Repeat("D", 700))
	require.NoError(t, p.Produce(context.Background(), third, eps))
	assert.Zero(t, matchInfo(t, ep).MatchBlocks(), "a changed preamble must not inherit the lineage")
}

// TestChunk_SeparatorInContentCannotForgeAFrame asserts a client cannot
// synthesize a message boundary by embedding framing bytes in its own text.
// Were boundaries delimited rather than length-prefixed, a caller could craft
// one turn that framed itself as several and claim the affinity of a session it
// never sent, or split its own text so it stopped matching itself.
func TestChunk_SeparatorInContentCannotForgeAFrame(t *testing.T) {
	p := newTestProducer(t)

	a := strings.Repeat("a", 300)
	b := strings.Repeat("b", 300)

	// Two turns, framed by the producer.
	twoTurns := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{ChatCompletions: &fwkrh.ChatCompletionsRequest{
			Messages: []fwkrh.Message{
				{Role: "user", Content: fwkrh.Content{Raw: a}},
				{Role: "assistant", Content: fwkrh.Content{Raw: b}},
			},
		}},
	}

	// One turn whose text spells out the frame the producer would have written.
	forged := &fwksched.InferenceRequest{
		RequestID:   uuid.NewString(),
		TargetModel: "test-model",
		Body: &fwkrh.InferenceRequestBody{ChatCompletions: &fwkrh.ChatCompletionsRequest{
			Messages: []fwkrh.Message{
				{Role: "user", Content: fwkrh.Content{
					Raw: a + "\x1fchat\x1fassistant\x1f" + b,
				}},
			},
		}},
	}

	assert.NotEqual(t, chainOf(p, twoTurns), chainOf(p, forged),
		"content must not be able to impersonate a frame boundary")
}

// TestChunk_ToolUseIDIsFramed asserts two transcripts differing only in the id
// of a tool call resolve to different chains. Templates render that id, so a
// shared chain would claim a match the engine's KV does not hold.
func TestChunk_ToolUseIDIsFramed(t *testing.T) {
	p := newTestProducer(t)

	req := func(id string) *fwksched.InferenceRequest {
		return &fwksched.InferenceRequest{
			RequestID:   uuid.NewString(),
			TargetModel: "test-model",
			Body: &fwkrh.InferenceRequestBody{Messages: &fwkrh.MessagesRequest{
				Messages: []fwkrh.AnthropicMessage{
					{Role: "user", Content: fwkrh.AnthropicContent{Raw: bigText(600)}},
					{Role: "assistant", Content: fwkrh.AnthropicContent{Structured: []fwkrh.AnthropicContentBlock{
						{Type: "tool_use", ID: id, Name: "get_weather", Input: []byte(`{"city":"SF"}`)},
					}}},
					{Role: "user", Content: fwkrh.AnthropicContent{Raw: bigText(600)}},
				},
			}},
		}
	}

	assert.NotEqual(t, chainOf(p, req("toolu_A")), chainOf(p, req("toolu_B")),
		"a differing tool_use id must break the chain")
}

// TestAlias_AgentIdentityContinuesOneLineage asserts the agent-identity
// attribute works as an alias source, not just the body keys. It is the form an
// operator gets by enabling the agent-identity plugin, and the only one that
// covers clients which name their session in a header.
func TestAlias_AgentIdentityContinuesOneLineage(t *testing.T) {
	p := newTestProducer(t)
	ep := testEndpoint("pod1")
	eps := []fwksched.Endpoint{ep}

	turn := func(id, input string) *fwksched.InferenceRequest {
		req := &fwksched.InferenceRequest{
			RequestID:   uuid.NewString(),
			TargetModel: "test-model",
			Body: &fwkrh.InferenceRequestBody{
				Responses: &fwkrh.ResponsesRequest{Input: input},
			},
		}
		req.PutAttribute(agentidentity.AgentIdentityKey, id)
		return req
	}

	first := turn("session-a", bigText(700))
	require.NoError(t, p.Produce(context.Background(), first, eps))
	require.NoError(t, p.PreRequest(context.Background(), first, schedTo(ep)))

	// The next turn of the same session carries none of the earlier bytes, so
	// only the alias can connect it to the pod that served them.
	second := turn("session-a", strings.Repeat("B", 700))
	require.NoError(t, p.Produce(context.Background(), second, eps))
	assert.Positive(t, matchInfo(t, ep).MatchBlocks(), "the agent identity must continue the lineage")

	// A different session must not inherit it.
	other := turn("session-b", strings.Repeat("C", 700))
	require.NoError(t, p.Produce(context.Background(), other, eps))
	assert.Zero(t, matchInfo(t, ep).MatchBlocks(), "a different agent identity starts fresh")
}

// TestChunk_ToolSchemaChangeBreaksAffinity asserts a tool whose schema or
// description changes under a stable name resolves to a different chain.
// Templates render the whole definition, so matching on the name alone would
// claim a hit against a pod whose prompt diverges in the tool block.
func TestChunk_ToolSchemaChangeBreaksAffinity(t *testing.T) {
	p := newTestProducer(t)

	req := func(desc string) *fwksched.InferenceRequest {
		return &fwksched.InferenceRequest{
			RequestID:   uuid.NewString(),
			TargetModel: "test-model",
			Body: &fwkrh.InferenceRequestBody{Messages: &fwkrh.MessagesRequest{
				Tools: []fwkrh.AnthropicTool{{
					Name:        "get_weather",
					Description: desc,
					InputSchema: []byte(`{"type":"object"}`),
				}},
				Messages: []fwkrh.AnthropicMessage{
					{Role: "user", Content: fwkrh.AnthropicContent{Raw: bigText(900)}},
				},
			}},
		}
	}

	assert.NotEqual(t, chainOf(p, req(bigText(600))), chainOf(p, req(strings.Repeat("Z", 600))),
		"a changed tool description must break the chain")
}
