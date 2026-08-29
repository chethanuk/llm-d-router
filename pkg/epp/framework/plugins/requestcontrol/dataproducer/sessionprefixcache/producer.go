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

// Package sessionprefixcache provides a tokenizer-free DataProducer that derives
// prefix-cache affinity from request content at session granularity. It hashes
// the request's framed text into a chain of fixed-size byte chunks, keeps a
// per-pod LRU of chains served to each endpoint, and publishes a longest-prefix
// match as PrefixCacheMatchInfo so affinity-aware scorers can prefer the pod that
// most recently served the shared prefix. It consumes no tokenized prompt and
// runs against a stock inference engine.
package sessionprefixcache

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"sync/atomic"
	"time"

	lru "github.com/hashicorp/golang-lru/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// ProducerType is the plugin type registered with the framework.
const ProducerType = "session-prefix-cache-producer"

const (
	defaultChunkSizeBytes = 512

	// defaultMaxChunks caps a single request's chain. At the default chunk size
	// it covers a megabyte of content, past the context window of any current
	// model, so a real session is never truncated: a chain cut mid-session would
	// report its match ratio against the cap rather than against the prompt and
	// read as higher coverage than the pod actually holds. The cap is required
	// because an aliased session appends to its chain every turn, and a chain
	// past maxEntriesPerPod would evict its own head as it indexes its tail.
	defaultMaxChunks = 2048

	// defaultMaxEntriesPerPod bounds one pod's LRU of chain hashes. At the
	// default 512-byte chunk it tracks the leading 50MB of distinct content per
	// pod, which covers a few thousand concurrent agentic sessions carrying the
	// 12k-token shared roots these workloads typically have.
	defaultMaxEntriesPerPod = 100000

	// defaultMaxAliasedSessions bounds how many declared session ids keep a
	// remembered chain. Each entry costs eight bytes per chunk, so the default
	// pairs with defaultMaxChunks for a ceiling in the tens of megabytes.
	defaultMaxAliasedSessions = 4096

	// initialBytesPerTokenQ is the starting bytes-per-token ratio, in units of
	// bytesPerTokenScale. English prose tokenizes near four bytes per token;
	// code runs leaner and CJK measured in UTF-8 bytes runs far richer, so the
	// ratio is a starting point that ResponseBody calibrates from engine-reported
	// usage rather than a constant to be trusted.
	initialBytesPerTokenQ = 4 * bytesPerTokenScale

	// bytesPerTokenScale is the fixed-point denominator for the calibrated
	// bytes-per-token ratio, which is otherwise too coarse as an integer.
	bytesPerTokenScale = 256

	// minBytesPerTokenQ and maxBytesPerTokenQ bound a single observation to a
	// band no real tokenizer leaves, so a misreported usage block moves the
	// ratio by a bounded amount instead of an arbitrary one.
	minBytesPerTokenQ = 1 * bytesPerTokenScale
	maxBytesPerTokenQ = 64 * bytesPerTokenScale

	// bytesPerTokenAlphaShift sets the calibration EWMA weight to 1/8, damping
	// per-request noise while still tracking a fleet whose traffic mix shifts.
	bytesPerTokenAlphaShift = 3

	// podActiveCheckInterval is how often the janitor prunes pods no longer in
	// the pool from the index.
	podActiveCheckInterval = 2 * time.Minute
)

var (
	_ requestcontrol.DataProducer          = &sessionPrefixCacheProducer{}
	_ requestcontrol.PreRequest            = &sessionPrefixCacheProducer{}
	_ requestcontrol.ResponseBodyProcessor = &sessionPrefixCacheProducer{}
)

// Parameters is the user-facing plugin configuration block.
type Parameters struct {
	// ChunkSizeBytes is the minimum size of a complete content chunk.
	ChunkSizeBytes int `json:"chunkSizeBytes"`
	// MaxChunks caps how many leading chunks a single request contributes.
	MaxChunks int `json:"maxChunks"`
	// MaxEntriesPerPod bounds the per-pod LRU of chain hashes.
	MaxEntriesPerPod int `json:"maxEntriesPerPod"`
	// MaxAliasedSessions bounds how many declared session ids keep a remembered
	// chain.
	MaxAliasedSessions int `json:"maxAliasedSessions"`
}

// defaultParameters seeds Factory and the tests with production defaults.
var defaultParameters = Parameters{
	ChunkSizeBytes:     defaultChunkSizeBytes,
	MaxChunks:          defaultMaxChunks,
	MaxEntriesPerPod:   defaultMaxEntriesPerPod,
	MaxAliasedSessions: defaultMaxAliasedSessions,
}

// sessionPrefixCacheProducer derives content-based prefix-cache affinity at
// session granularity.
type sessionPrefixCacheProducer struct {
	typedName      fwkplugin.TypedName
	dk             fwkplugin.DataKey
	index          *index
	chunkSizeBytes int
	maxChunks      int

	// alias remembers the chain last served under each declared session id, so a
	// session whose history the client does not resend still extends one lineage.
	alias *lru.Cache[string, []uint64]

	// pluginState carries the resolved chain from Produce to PreRequest.
	pluginState *fwkplugin.PluginState

	// bytesPerTokenQ is the calibrated bytes-per-token ratio scaled by
	// bytesPerTokenScale. It converts a chunk size in bytes into the block size
	// in tokens that PrefixCacheMatchInfo consumers use to price a match.
	bytesPerTokenQ atomic.Int64
}

// Factory builds the producer from raw plugin parameters, defaulting every field
// and rejecting unknown ones via the strict decoder.
func Factory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := defaultParameters
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' producer: %w", ProducerType, err)
		}
	}

	return newProducer(name, params, handle)
}

func newProducer(name string, params Parameters, handle fwkplugin.Handle) (*sessionPrefixCacheProducer, error) {
	if handle == nil {
		return nil, errors.New("plugin handle is required")
	}
	if params.ChunkSizeBytes <= 0 {
		return nil, fmt.Errorf("invalid configuration: chunkSizeBytes must be > 0 (current value: %d)", params.ChunkSizeBytes)
	}
	if params.MaxChunks <= 0 {
		return nil, fmt.Errorf("invalid configuration: maxChunks must be > 0 (current value: %d)", params.MaxChunks)
	}
	if params.MaxEntriesPerPod <= 0 {
		return nil, fmt.Errorf("invalid configuration: maxEntriesPerPod must be > 0 (current value: %d)", params.MaxEntriesPerPod)
	}
	if params.MaxAliasedSessions <= 0 {
		return nil, fmt.Errorf("invalid configuration: maxAliasedSessions must be > 0 (current value: %d)", params.MaxAliasedSessions)
	}
	// A chain longer than one pod's LRU evicts its own head as it indexes its
	// tail, leaving the pod that just served the request matching nothing.
	if params.MaxChunks > params.MaxEntriesPerPod {
		return nil, fmt.Errorf("invalid configuration: maxChunks (%d) must be <= maxEntriesPerPod (%d)",
			params.MaxChunks, params.MaxEntriesPerPod)
	}

	alias, err := lru.New[string, []uint64](params.MaxAliasedSessions)
	if err != nil {
		return nil, fmt.Errorf("failed to create the session alias cache: %w", err)
	}

	p := &sessionPrefixCacheProducer{
		typedName:      fwkplugin.TypedName{Type: ProducerType, Name: name},
		dk:             attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(name),
		index:          newIndex(params.MaxEntriesPerPod),
		chunkSizeBytes: params.ChunkSizeBytes,
		maxChunks:      params.MaxChunks,
		alias:          alias,
		pluginState:    fwkplugin.NewPluginState(handle.Context()),
	}
	p.bytesPerTokenQ.Store(initialBytesPerTokenQ)

	go p.cleanUpInactivePods(handle.Context(), handle)
	return p, nil
}

// chainState carries the chain Produce resolved through to PreRequest, so that
// the chain a request is scored against is the one it is indexed under. Resolving
// twice would not give that: the alias moves as concurrent turns of the same
// session land, so the second resolve could return a different lineage than the
// one Produce published to the scorer.
type chainState struct {
	chain    []uint64
	aliasKey string
}

func (s *chainState) Clone() fwkplugin.StateData {
	return &chainState{chain: slices.Clone(s.chain), aliasKey: s.aliasKey}
}

// stateKey names this producer's slot in the per-request plugin state.
func (p *sessionPrefixCacheProducer) stateKey() fwkplugin.StateKey {
	return fwkplugin.StateKey(p.typedName.Name)
}

// TypedName returns the type and name of the plugin.
func (p *sessionPrefixCacheProducer) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Produces declares the PrefixCacheMatchInfo attribute written by this producer.
func (p *sessionPrefixCacheProducer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{p.dk: attrprefix.PrefixCacheMatchInfo{}}
}

// Produce hashes the request content into a chunk chain and publishes each
// candidate pod's longest cached prefix as PrefixCacheMatchInfo. Content that
// fills no complete chunk and belongs to no known session carries no affinity
// signal and publishes nothing, which leaves consumers to score it neutrally
// rather than to read a zero-length chain as zero coverage.
func (p *sessionPrefixCacheProducer) Produce(_ context.Context, req *fwksched.InferenceRequest, pods []fwksched.Endpoint) error {
	if req == nil || req.Body == nil {
		return nil
	}

	chain, aliasKey := p.resolveChain(req)
	if len(chain) == 0 {
		return nil
	}
	// Without an id every concurrent request shares one plugin-state key, so one
	// request would seed another's chain against its endpoint.
	if req.RequestID != "" {
		p.pluginState.Write(req.RequestID, p.stateKey(), &chainState{chain: chain, aliasKey: aliasKey})
	}

	candidates := make([]serverID, 0, len(pods))
	for _, pod := range pods {
		candidates = append(candidates, serverID(pod.GetMetadata().ID))
	}
	matches := p.index.LongestPrefixes(chain, candidates)

	total := len(chain)
	blockSizeTokens := p.blockSizeTokens()
	for _, pod := range pods {
		match := matches[serverID(pod.GetMetadata().ID)]
		pod.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(match, total, blockSizeTokens))
	}
	return nil
}

// PreRequest seeds the index with the chain served to the selected endpoint of
// every scheduling profile so subsequent requests can match it, and records the
// chain against the request's declared session id so the next turn of that
// session continues it.
//
// The index entry records where a turn was routed. The engine has not confirmed
// it holds anything, and the entry carries no eviction signal. Seeding at this
// point is what lets concurrent turns of one session find each other before any
// of them completes.
func (p *sessionPrefixCacheProducer) PreRequest(ctx context.Context, req *fwksched.InferenceRequest, result *fwksched.SchedulingResult) error {
	if req == nil || req.Body == nil || result == nil {
		return nil
	}
	if req.RequestID == "" {
		return nil
	}
	defer p.pluginState.Delete(req.RequestID)

	state, err := fwkplugin.ReadPluginStateKey[*chainState](p.pluginState, req.RequestID, p.stateKey())
	if err != nil {
		log.FromContext(ctx).V(logutil.DEBUG).Info("no resolved chain for request, skipping index seed",
			"requestID", req.RequestID, "error", err)
		return nil
	}
	chain := state.chain
	if len(chain) == 0 {
		return nil
	}
	if state.aliasKey != "" {
		p.alias.Add(state.aliasKey, chain)
	}

	// Seed the served endpoint of every scheduling profile, not just the primary,
	// so P/D-disaggregated prefill nodes also gain affinity. A pod may front
	// multiple profiles; dedupe to avoid redundant index writes.
	seen := make(map[serverID]struct{}, len(result.ProfileResults))
	for _, pr := range result.ProfileResults {
		if pr == nil || len(pr.TargetEndpoints) == 0 {
			continue
		}
		srv := serverID(pr.TargetEndpoints[0].GetMetadata().ID)
		if _, ok := seen[srv]; ok {
			continue
		}
		seen[srv] = struct{}{}
		p.index.Add(chain, srv)
	}
	return nil
}

// ResponseBody calibrates the bytes-per-token ratio from the prompt-token count
// the engine reports for content the router already measured in bytes.
//
// The index needs no correction here. The chain covers exactly the content sent,
// and a pod that has served a request holds the KV for all of it, so there is no
// over-estimated tail to walk back. What the router cannot derive on its own is
// how many tokens a chunk of bytes became, which the reported usage supplies.
// Consumers of PrefixCacheMatchInfo price a match in tokens, so the ratio has to
// track the traffic rather than an assumed constant for English prose.
//
// A sample only counts when the reported count measures the same content the
// router hashed. Anthropic reports input_tokens net of cached and cache-creation
// blocks, and a request carrying images or audio spends tokens on content the
// stream names by digest rather than carries, so neither is a usable denominator.
func (p *sessionPrefixCacheProducer) ResponseBody(_ context.Context, req *fwksched.InferenceRequest, response *requestcontrol.Response, _ *fwkdl.EndpointMetadata) {
	if req == nil || req.Body == nil || response == nil || !response.EndOfStream {
		return
	}
	if req.Body.Messages != nil {
		return
	}
	// These surfaces report tokens for the conversation the engine holds, while
	// the router only ever framed the turn the client sent.
	if serverSideHistory(req.Body) {
		return
	}
	promptTokens := response.Usage.PromptTokens
	if promptTokens <= 0 {
		return
	}

	stream, textOnly := contentStream(req.Body)
	if !textOnly || len(stream) == 0 {
		return
	}
	p.observeBytesPerToken(len(stream), promptTokens)
}

// observeBytesPerToken folds one observation into the calibrated ratio.
func (p *sessionPrefixCacheProducer) observeBytesPerToken(streamBytes, promptTokens int) {
	sample := int64(streamBytes) * bytesPerTokenScale / int64(promptTokens)
	sample = min(max(sample, minBytesPerTokenQ), maxBytesPerTokenQ)
	for {
		cur := p.bytesPerTokenQ.Load()
		next := max(cur+(sample-cur)/(1<<bytesPerTokenAlphaShift), minBytesPerTokenQ)
		if p.bytesPerTokenQ.CompareAndSwap(cur, next) {
			return
		}
	}
}

// blockSizeTokens converts the chunk size in bytes into the token count that
// PrefixCacheMatchInfo consumers use to turn a block match into a token match.
func (p *sessionPrefixCacheProducer) blockSizeTokens() int {
	n := int(int64(p.chunkSizeBytes) * bytesPerTokenScale / p.bytesPerTokenQ.Load())
	return max(n, 1)
}

// cleanUpInactivePods periodically removes pods no longer in the pool.
func (p *sessionPrefixCacheProducer) cleanUpInactivePods(ctx context.Context, handle fwkplugin.Handle) {
	ticker := time.NewTicker(podActiveCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			active := make(map[serverID]struct{})
			for _, nsn := range handle.PodList() {
				active[serverID(nsn)] = struct{}{}
			}
			for _, srv := range p.index.Pods() {
				if _, ok := active[srv]; !ok {
					p.index.RemovePod(srv)
					log.FromContext(ctx).V(logutil.VERBOSE).Info("Removed pod not in active set", "pod", srv)
				}
			}
		}
	}
}

// resolveChain hashes the request's framed content into a chunk chain, extending
// the chain remembered for the request's declared session id when the client did
// not resend the session's history. It returns the chain and that id, if any.
//
// A client that resends history needs no alias: the content chain already covers
// every earlier turn. A client on the Responses or Conversations surfaces sends
// only the newest turn, because the engine holds the rest; there the content
// chain starts a fresh lineage every turn, and appending it to the remembered one
// is what keeps the session on the pod that has been serving it.
//
// The declared id is never hashed. Two byte-identical prompts under different
// session names still produce the same chain, because neither has a remembered
// one on first contact.
//
// It reads the alias cache, so its result depends on what other turns of the
// same session have already been served. Produce resolves it once and hands the
// result to PreRequest through the plugin state rather than letting PreRequest
// resolve it again, which would let a concurrent turn move the alias between the
// two hooks and index a lineage no Produce ever scored.
func (p *sessionPrefixCacheProducer) resolveChain(req *fwksched.InferenceRequest) (chain []uint64, id string) {
	b := req.Body
	stream, _ := contentStream(b)
	seed := rootSeed(req.TargetModel, cacheSalt(b))
	content := chunkChain(stream, seed, p.chunkSizeBytes, p.maxChunks)

	// Only the surfaces that let the engine hold turns the client does not resend
	// consult the alias. Where the client sends the whole prompt every turn, the
	// content chain is what was sent, and extending it past a point the client
	// edited would claim a match the pod cannot honour.
	declared := declaredID(req)
	if declared == "" || !serverSideHistory(b) {
		return content, ""
	}
	// The alias key carries the same root the chain does, so a session name
	// reused under another model or cache salt cannot inherit a lineage across
	// the isolation boundary the salt exists to draw. It also carries the
	// preamble: agent clients rewrite instructions every turn with the working
	// directory and the time, and that invalidates every turn behind it, so a
	// lineage built on one preamble must not be extended under another.
	id = strconv.FormatUint(seed, 16) + "\x00" +
		strconv.FormatUint(preambleDigest(b), 16) + "\x00" + declared

	prior, ok := p.alias.Get(id)
	if !ok || len(prior) == 0 {
		return content, id
	}
	// A turn too small to fill a chunk carries no content signal of its own, but
	// it still belongs to the lineage the client named.
	if len(content) == 0 {
		return prior, id
	}
	// The content chain covering the whole remembered one means the client sent
	// this session's history after all, so it is the truth and supersedes what was
	// remembered. A content chain that merely repeats some leading part of the
	// lineage is not a resend: a turn that happens to hash to a prefix of its own
	// session, such as one repeating an earlier question, must extend the lineage
	// rather than replace it with a shorter one.
	if sharedPrefixLen(prior, content) == len(prior) {
		return content, id
	}

	remaining := p.maxChunks - len(prior)
	if remaining <= 0 {
		return prior, id
	}
	tail := chunkChain(continuationStream(b), prior[len(prior)-1], p.chunkSizeBytes, remaining)
	return append(slices.Clone(prior), tail...), id
}

// serverSideHistory reports whether the request's API surface lets the engine
// hold earlier turns that the client does not resend.
func serverSideHistory(b *fwkrh.InferenceRequestBody) bool {
	return b.Responses != nil || b.Conversations != nil
}

// cacheSalt returns the request's cache-isolation salt, if any.
func cacheSalt(b *fwkrh.InferenceRequestBody) string {
	switch {
	case b.Completions != nil:
		return b.Completions.CacheSalt
	case b.ChatCompletions != nil:
		return b.ChatCompletions.CacheSalt
	case b.Messages != nil:
		return b.Messages.CacheSalt
	case b.Responses != nil:
		return b.Responses.CacheSalt
	case b.Conversations != nil:
		return b.Conversations.CacheSalt
	}
	return ""
}
