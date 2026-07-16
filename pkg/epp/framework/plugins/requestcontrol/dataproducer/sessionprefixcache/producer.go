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
	"fmt"
	"strings"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	sessionprefixcacheconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sessionprefixcache/constants"
)

// ProducerType is the plugin type registered with the framework.
const ProducerType = sessionprefixcacheconstants.ProducerType

const (
	defaultChunkSizeBytes   = 512
	defaultMaxChunks        = 256
	defaultMaxEntriesPerPod = 100000

	// bytesPerToken maps chunk bytes to an approximate token count, used both for
	// the scorer's optional absolute-length term and to reconcile a response's
	// reported prompt-token usage against whole chunks.
	bytesPerToken = 4

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
	// SessionHeaders is the ordered list of request headers consulted for a
	// session id when the body carries no prompt_cache_key.
	SessionHeaders []string `json:"sessionHeaders"`
}

// defaultParameters seeds Factory and the tests with production defaults.
var defaultParameters = Parameters{
	ChunkSizeBytes:   defaultChunkSizeBytes,
	MaxChunks:        defaultMaxChunks,
	MaxEntriesPerPod: defaultMaxEntriesPerPod,
	SessionHeaders:   defaultSessionHeaders,
}

// sessionPrefixCacheProducer derives content-based prefix-cache affinity at
// session granularity.
type sessionPrefixCacheProducer struct {
	typedName       fwkplugin.TypedName
	dk              fwkplugin.DataKey
	index           *index
	state           *fwkplugin.PluginState
	stateKey        fwkplugin.StateKey
	chunkSizeBytes  int
	maxChunks       int
	blockSizeTokens int
	sessionHeaders  []string
}

// chainState is the per-request chain saved by Produce for PreRequest to seed
// and ResponseBody to confirm.
type chainState struct {
	chain []uint64
}

func (s *chainState) Clone() fwkplugin.StateData {
	return &chainState{chain: append([]uint64(nil), s.chain...)}
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

	ctx := context.Background()
	if handle != nil {
		ctx = handle.Context()
	}
	return newProducer(ctx, name, params, handle)
}

func newProducer(ctx context.Context, name string, params Parameters, handle fwkplugin.Handle) (*sessionPrefixCacheProducer, error) {
	if params.ChunkSizeBytes <= 0 {
		return nil, fmt.Errorf("invalid configuration: chunkSizeBytes must be > 0 (current value: %d)", params.ChunkSizeBytes)
	}
	if params.MaxChunks <= 0 {
		return nil, fmt.Errorf("invalid configuration: maxChunks must be > 0 (current value: %d)", params.MaxChunks)
	}
	if params.MaxEntriesPerPod <= 0 {
		return nil, fmt.Errorf("invalid configuration: maxEntriesPerPod must be > 0 (current value: %d)", params.MaxEntriesPerPod)
	}

	headers := make([]string, 0, len(params.SessionHeaders))
	for _, h := range params.SessionHeaders {
		if lowered := strings.ToLower(strings.TrimSpace(h)); lowered != "" {
			headers = append(headers, lowered)
		}
	}

	blockSizeTokens := params.ChunkSizeBytes / bytesPerToken
	if blockSizeTokens < 1 {
		blockSizeTokens = 1
	}

	p := &sessionPrefixCacheProducer{
		typedName:       fwkplugin.TypedName{Type: ProducerType, Name: name},
		dk:              attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(name),
		index:           newIndex(params.MaxEntriesPerPod),
		state:           fwkplugin.NewPluginState(ctx),
		stateKey:        fwkplugin.StateKey(name),
		chunkSizeBytes:  params.ChunkSizeBytes,
		maxChunks:       params.MaxChunks,
		blockSizeTokens: blockSizeTokens,
		sessionHeaders:  headers,
	}

	if handle != nil {
		go p.cleanUpInactivePods(ctx, handle)
	}
	return p, nil
}

// TypedName returns the type and name of the plugin.
func (p *sessionPrefixCacheProducer) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Produces declares the PrefixCacheMatchInfo attribute written by this producer.
func (p *sessionPrefixCacheProducer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{p.dk: attrprefix.PrefixCacheMatchInfo{}}
}

// Produce hashes the request content into a chunk chain, saves it for the
// pre-request seeding step, and publishes each candidate pod's longest cached
// prefix as PrefixCacheMatchInfo. Content that fills no complete chunk carries no
// affinity signal and publishes nothing.
func (p *sessionPrefixCacheProducer) Produce(_ context.Context, req *fwksched.InferenceRequest, pods []fwksched.Endpoint) error {
	if req == nil || req.Body == nil {
		return nil
	}

	chain := p.buildChain(req)
	if len(chain) == 0 {
		return nil
	}
	p.state.Write(req.RequestID, p.stateKey, &chainState{chain: chain})

	total := len(chain)
	key := p.dk.String()
	for _, pod := range pods {
		match := p.index.LongestPrefix(chain, ServerID(pod.GetMetadata().NamespacedName))
		pod.Put(key, attrprefix.NewPrefixCacheMatchInfo(match, total, p.blockSizeTokens))
	}
	return nil
}

// PreRequest seeds the index with the chain served to the selected endpoint of
// every scheduling profile so subsequent requests can match it.
func (p *sessionPrefixCacheProducer) PreRequest(ctx context.Context, req *fwksched.InferenceRequest, result *fwksched.SchedulingResult) {
	if req == nil || result == nil {
		return
	}
	defer p.state.Delete(req.RequestID)

	st, err := fwkplugin.ReadPluginStateKey[*chainState](p.state, req.RequestID, p.stateKey)
	if err != nil || st == nil || len(st.chain) == 0 {
		return
	}

	// Seed the served endpoint of every scheduling profile, not just the primary,
	// so P/D-disaggregated prefill nodes also gain affinity (R2-6). A pod may
	// front multiple profiles; dedupe to avoid redundant index writes.
	//
	// Seeding is synchronous: ResponseBody may refine the same chain downward on
	// the served endpoint, and an async Add(estimated) racing after that trim
	// would re-insert the discarded tail.
	seen := make(map[ServerID]struct{})
	for _, pr := range result.ProfileResults {
		if pr == nil || len(pr.TargetEndpoints) == 0 {
			continue
		}
		srv := ServerID(pr.TargetEndpoints[0].GetMetadata().NamespacedName)
		if _, ok := seen[srv]; ok {
			continue
		}
		seen[srv] = struct{}{}
		p.index.Add(st.chain, srv, estimated)
	}
}

// ResponseBody confirms the prefix the served pod actually cached (from reported
// prompt-token usage) and trims any over-estimated tail.
func (p *sessionPrefixCacheProducer) ResponseBody(_ context.Context, req *fwksched.InferenceRequest, response *requestcontrol.Response, targetEndpoint *fwkdl.EndpointMetadata) {
	if req == nil || req.Body == nil || response == nil || !response.EndOfStream || targetEndpoint == nil {
		return
	}
	promptTokens := response.Usage.PromptTokens
	if promptTokens <= 0 {
		return
	}

	// PreRequest already deleted the per-request state, so recompute the chain
	// from the (still-present) request body; buildChain is pure and reproduces
	// the exact seeding used at Produce time.
	chain := p.buildChain(req)
	if len(chain) == 0 {
		return
	}

	// Reported prompt tokens map back to whole chunks (round to nearest). The
	// confirmed prefix is capped by the chain: a match can never exceed total.
	confirmedChunks := (promptTokens + p.blockSizeTokens/2) / p.blockSizeTokens
	if confirmedChunks > len(chain) {
		confirmedChunks = len(chain)
	}

	srv := ServerID(targetEndpoint.NamespacedName)
	p.index.Add(chain[:confirmedChunks], srv, confirmed)
	p.index.TrimEstimatedTail(chain[confirmedChunks:], srv)
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
			active := make(map[ServerID]struct{})
			for _, nsn := range handle.PodList() {
				active[ServerID(nsn)] = struct{}{}
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

// buildChain hashes the request's framed content into a chunk chain seeded by the
// model, cache salt, and declared session id.
func (p *sessionPrefixCacheProducer) buildChain(req *fwksched.InferenceRequest) []uint64 {
	b := req.Body
	declaredID, _ := declaredSessionID(b, req.Headers, p.sessionHeaders)
	return chunkChain(contentStream(b), req.TargetModel, cacheSalt(b), declaredID, p.chunkSizeBytes, p.maxChunks)
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
