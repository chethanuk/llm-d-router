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

package multimodal

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmm "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/multimodal"
)

func TestLRUCapacityFromCacheSizeMB(t *testing.T) {
	assert.Equal(t, 2, lruCapacityFromCacheSizeMB(4))
	assert.Equal(t, 1024, lruCapacityFromCacheSizeMB(2048))
	assert.Equal(t, 2048, lruCapacityFromCacheSizeMB(0))
}

func TestFactory(t *testing.T) {
	raw, err := json.Marshal(map[string]any{"cacheSizeInMBPerServer": 4})
	require.NoError(t, err)

	created, err := Factory("mm-producer", plugin.StrictDecoder(raw), &testHandle{ctx: context.Background()})
	require.NoError(t, err)
	require.NotNil(t, created)
	assert.Equal(t, "mm-producer", created.TypedName().Name)
	assert.Equal(t, ProducerType, created.TypedName().Type)

	_, err = Factory("bad", plugin.StrictDecoder(json.RawMessage(`{"cacheSizeInMBPerServer":"bad"}`)), &testHandle{ctx: context.Background()})
	require.Error(t, err)
}

func TestExtractMMItemsFromTokenizedPrompt(t *testing.T) {
	items := ExtractMMItems(&scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{
				MultiModalFeatures: []fwkrh.MultiModalFeature{
					{Modality: fwkrh.ModalityImage, Hash: "image-a", Length: 576},
					{Modality: fwkrh.ModalityImage, Hash: "image-b", Length: 0},
					{Modality: fwkrh.ModalityImage, Hash: "image-a", Length: 144},
				},
			},
		},
	})

	assert.ElementsMatch(t, []attrmm.MatchItem{
		{Hash: "image-a", Size: 1, Modality: string(fwkrh.ModalityImage)},
		{Hash: "image-b", Size: 1, Modality: string(fwkrh.ModalityImage)},
	}, items)
}

func TestExtractMMItemsNilTokenizedPromptReturnsNil(t *testing.T) {
	items := ExtractMMItems(&scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{},
	})
	assert.Nil(t, items)
}

func TestExtractMMItemsEmptyMultiModalFeaturesReturnsNil(t *testing.T) {
	items := ExtractMMItems(&scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{},
		},
	})
	assert.Nil(t, items)
}

func TestExtractMMItemsIgnoresProtocolStructs(t *testing.T) {
	// Protocol structs carry multimodal content but are never read; only the
	// tokenized prompt's features count.
	items := ExtractMMItems(&scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{{
					Role: "user",
					Content: fwkrh.Content{Structured: []fwkrh.ContentBlock{
						{Type: "image_url", ImageURL: fwkrh.ImageBlock{URL: "https://example.com/cat.png"}},
					}},
				}},
			},
		},
	})

	assert.Nil(t, items)
}

func TestProduceMatchesMultiplePodsAndPreRequestUpdatesPlacement(t *testing.T) {
	producer := newTestProducer(t, nil, nil)
	podA := k8stypes.NamespacedName{Namespace: "default", Name: "pod-a"}
	podB := k8stypes.NamespacedName{Namespace: "default", Name: "pod-b"}
	podC := k8stypes.NamespacedName{Namespace: "default", Name: "pod-c"}
	producer.putCacheEntry("hash-a", podA, podB)

	endpointA := newEndpoint(podA)
	endpointB := newEndpoint(podB)
	endpointC := newEndpoint(podC)
	request := requestWithHashes("req-1", map[string]int{"hash-a": 80, "hash-c": 20})

	require.NoError(t, producer.Produce(context.Background(), request, []scheduling.Endpoint{endpointA, endpointB, endpointC}))

	img := string(fwkrh.ModalityImage)
	assertMatchInfo(t, producer, endpointA,
		[]attrmm.MatchItem{{Hash: "hash-a", Size: 1, Modality: img}},
		[]attrmm.MatchItem{{Hash: "hash-a", Size: 1, Modality: img}, {Hash: "hash-c", Size: 1, Modality: img}})
	assertMatchInfo(t, producer, endpointB,
		[]attrmm.MatchItem{{Hash: "hash-a", Size: 1, Modality: img}},
		[]attrmm.MatchItem{{Hash: "hash-a", Size: 1, Modality: img}, {Hash: "hash-c", Size: 1, Modality: img}})
	assertMatchInfo(t, producer, endpointC,
		nil,
		[]attrmm.MatchItem{{Hash: "hash-a", Size: 1, Modality: img}, {Hash: "hash-c", Size: 1, Modality: img}})

	producer.PreRequest(context.Background(), request, schedulingResult(endpointC))
	producer.wg.Wait()

	cache := producer.cacheSnapshot()
	assert.Contains(t, cache["hash-a"], podA.String())
	assert.Contains(t, cache["hash-a"], podB.String())
	assert.Contains(t, cache["hash-a"], podC.String())
	assert.Contains(t, cache["hash-c"], podC.String())
}

func TestLRUEviction(t *testing.T) {
	producer := newTestProducer(t, &Parameters{CacheSizeInMBPerServer: 4}, nil)
	endpoint := newEndpoint(k8stypes.NamespacedName{Namespace: "default", Name: "pod-a"})

	for _, hash := range []string{"hash-1", "hash-2", "hash-3"} {
		request := requestWithHashes(hash, map[string]int{hash: 1})
		require.NoError(t, producer.Produce(context.Background(), request, []scheduling.Endpoint{endpoint}))
		producer.PreRequest(context.Background(), request, schedulingResult(endpoint))
		producer.wg.Wait()
	}

	cache := producer.cacheSnapshot()
	assert.NotContains(t, cache, "hash-1")
	assert.Contains(t, cache, "hash-2")
	assert.Contains(t, cache, "hash-3")
}

func TestStalePodCleanup(t *testing.T) {
	podA := k8stypes.NamespacedName{Namespace: "default", Name: "pod-a"}
	podB := k8stypes.NamespacedName{Namespace: "default", Name: "pod-b"}
	producer := newTestProducer(t, nil, func() []k8stypes.NamespacedName { return []k8stypes.NamespacedName{podA} })
	producer.putCacheEntry("hash-a", podA, podB)

	// Simulate the periodic cleanup loop firing.
	producer.removeStalePods()

	assert.NotContains(t, producer.cacheSnapshot()["hash-a"], podB.String())
	assert.Contains(t, producer.cacheSnapshot()["hash-a"], podA.String())

	endpointA := newEndpoint(podA)
	endpointB := newEndpoint(podB)
	require.NoError(t, producer.Produce(context.Background(), requestWithHashes("req", map[string]int{"hash-a": 1}), []scheduling.Endpoint{endpointA, endpointB}))

	img := string(fwkrh.ModalityImage)
	assertMatchInfo(t, producer, endpointA,
		[]attrmm.MatchItem{{Hash: "hash-a", Size: 1, Modality: img}},
		[]attrmm.MatchItem{{Hash: "hash-a", Size: 1, Modality: img}})
	assertMatchInfo(t, producer, endpointB,
		nil,
		[]attrmm.MatchItem{{Hash: "hash-a", Size: 1, Modality: img}})
}

func TestProducerEndpointExtractorInterfaceContract(t *testing.T) {
	producer := newTestProducer(t, nil, nil)
	var _ fwkdl.EndpointExtractor = producer
	assert.True(t, reflect.TypeOf(producer).Implements(reflect.TypeFor[fwkdl.EndpointExtractor]()))
}

func TestExtractEndpointRemovesDeletedPod(t *testing.T) {
	podA := k8stypes.NamespacedName{Namespace: "default", Name: "pod-a"}
	podB := k8stypes.NamespacedName{Namespace: "default", Name: "pod-b"}
	producer := newTestProducer(t, nil, nil)
	producer.putCacheEntry("hash-a", podA, podB)

	err := producer.Extract(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventDelete,
		Endpoint: fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: podB}, nil),
	})

	require.NoError(t, err)
	cache := producer.cacheSnapshot()
	assert.Contains(t, cache["hash-a"], podA.String())
	assert.NotContains(t, cache["hash-a"], podB.String())
}

type testHandle struct {
	ctx     context.Context
	podList func() []k8stypes.NamespacedName
}

func (h *testHandle) Context() context.Context {
	return h.ctx
}

func (h *testHandle) Plugin(string) plugin.Plugin {
	return nil
}

func (h *testHandle) AddPlugin(string, plugin.Plugin) {}

func (h *testHandle) GetAllPlugins() []plugin.Plugin {
	return nil
}

func (h *testHandle) GetAllPluginsWithNames() map[string]plugin.Plugin {
	return nil
}

func (h *testHandle) Metrics() plugin.MetricsRecorder {
	return nil
}

func (h *testHandle) PodList() []k8stypes.NamespacedName {
	if h.podList == nil {
		return nil
	}
	return h.podList()
}

const testName = "test-mm-embeddings-cache-producer"

func newTestProducer(t *testing.T, params *Parameters, podList func() []k8stypes.NamespacedName) *Producer {
	t.Helper()
	producer, err := New(context.Background(), testName, params, podList)
	require.NoError(t, err)
	return producer
}

func newEndpoint(name k8stypes.NamespacedName) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: name},
		&fwkdl.Metrics{},
		nil,
	)
}

func requestWithHashes(requestID string, hashToWeight map[string]int) *scheduling.InferenceRequest {
	features := make([]fwkrh.MultiModalFeature, 0, len(hashToWeight))
	for hash, weight := range hashToWeight {
		features = append(features, fwkrh.MultiModalFeature{Modality: fwkrh.ModalityImage, Hash: hash, Length: weight})
	}
	return &scheduling.InferenceRequest{
		RequestID: requestID,
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{MultiModalFeatures: features},
		},
	}
}

func schedulingResult(target scheduling.Endpoint) *scheduling.SchedulingResult {
	return &scheduling.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"default": {TargetEndpoints: []scheduling.Endpoint{target}},
		},
	}
}

func assertMatchInfo(t *testing.T, p *Producer, endpoint scheduling.Endpoint, matchedItems, requestItems []attrmm.MatchItem) {
	t.Helper()
	raw, ok := endpoint.Get(p.dk.String())
	require.True(t, ok)
	info, ok := raw.(*attrmm.EncoderCacheMatchInfo)
	require.True(t, ok)
	assert.ElementsMatch(t, matchedItems, info.MatchedItems())
	assert.ElementsMatch(t, requestItems, info.RequestItems())
}

// newDumpTestProducer builds a Producer with per-pod LRUs seeded with `n` content
// hashes each, so DumpState can be exercised without the cleanup goroutine.
func newDumpTestProducer(t *testing.T, capacity int, pods map[string]int) *Producer {
	t.Helper()
	p := &Producer{
		caches:    make(map[string]*lru.Cache[string, struct{}]),
		cacheSize: capacity,
	}
	for pod, n := range pods {
		c, err := lru.New[string, struct{}](capacity)
		require.NoError(t, err)
		for i := 0; i < n; i++ {
			c.Add(fmt.Sprintf("%s-hash-%d", pod, i), struct{}{})
		}
		p.caches[pod] = c
	}
	return p
}

func TestProducer_DumpState(t *testing.T) {
	tests := []struct {
		name     string
		capacity int
		pods     map[string]int
		want     encoderCacheState
	}{
		{
			name:     "empty",
			capacity: 2048,
			pods:     map[string]int{},
			want: encoderCacheState{
				Endpoints:      []endpointEncoderCacheState{},
				TotalEndpoints: 0,
				MaxEndpoints:   maxDebugDumpEndpoints,
				CacheCapacity:  2048,
			},
		},
		{
			name:     "populated sorted by cached items desc",
			capacity: 2048,
			pods:     map[string]int{"pod-a": 1, "pod-b": 3},
			want: encoderCacheState{
				Endpoints: []endpointEncoderCacheState{
					{Endpoint: "pod-b", CachedItems: 3},
					{Endpoint: "pod-a", CachedItems: 1},
				},
				TotalEndpoints: 2,
				MaxEndpoints:   maxDebugDumpEndpoints,
				CacheCapacity:  2048,
			},
		},
		{
			name:     "tie broken by endpoint name",
			capacity: 2048,
			pods:     map[string]int{"pod-y": 2, "pod-x": 2},
			want: encoderCacheState{
				Endpoints: []endpointEncoderCacheState{
					{Endpoint: "pod-x", CachedItems: 2},
					{Endpoint: "pod-y", CachedItems: 2},
				},
				TotalEndpoints: 2,
				MaxEndpoints:   maxDebugDumpEndpoints,
				CacheCapacity:  2048,
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := newDumpTestProducer(t, tc.capacity, tc.pods)
			payload, err := p.DumpState()
			require.NoError(t, err)
			require.True(t, json.Valid(payload))
			var got encoderCacheState
			require.NoError(t, json.Unmarshal(payload, &got))
			require.Equal(t, tc.want, got)
		})
	}
}

func TestProducer_DumpStateCapsEndpoints(t *testing.T) {
	pods := map[string]int{}
	for i := 0; i < maxDebugDumpEndpoints+5; i++ {
		pods[fmt.Sprintf("pod-%03d", i)] = i + 1
	}
	p := newDumpTestProducer(t, 2048, pods)

	payload, err := p.DumpState()
	require.NoError(t, err)
	var got encoderCacheState
	require.NoError(t, json.Unmarshal(payload, &got))
	require.True(t, got.Truncated)
	require.Equal(t, maxDebugDumpEndpoints+5, got.TotalEndpoints)
	require.Equal(t, maxDebugDumpEndpoints, got.MaxEndpoints)
	require.Len(t, got.Endpoints, maxDebugDumpEndpoints)
	// Busiest endpoint first: pod with the most cached items heads the list.
	require.Equal(t, fmt.Sprintf("pod-%03d", maxDebugDumpEndpoints+4), got.Endpoints[0].Endpoint)
}

// TestProducer_DumpStateNoHashLeak proves the content hashes stored in the per-pod
// LRUs are never emitted: only pod identities and counts appear in the payload.
func TestProducer_DumpStateNoHashLeak(t *testing.T) {
	p := newDumpTestProducer(t, 2048, map[string]int{"pod-a": 3, "pod-b": 2})
	payload, err := p.DumpState()
	require.NoError(t, err)

	// Seeded hash keys ("pod-a-hash-0", ...) must not surface anywhere.
	for _, pod := range []string{"pod-a", "pod-b"} {
		for i := 0; i < 3; i++ {
			require.NotContains(t, string(payload), fmt.Sprintf("%s-hash-%d", pod, i),
				"content hash leaked into DumpState payload")
		}
	}
	// Only the four documented keys are present.
	var raw map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(payload, &raw))
	for k := range raw {
		require.Truef(t, map[string]bool{
			"endpoints": true, "totalEndpoints": true, "maxEndpoints": true,
			"cacheCapacity": true, "truncated": true,
		}[k], "unexpected key %q in payload", k)
	}
	// Sanity: pod identity (not the hashes) is what appears.
	require.True(t, strings.Contains(string(payload), "pod-a"))
}
