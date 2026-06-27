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

package preciseprefixcache

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/jellydator/ttlcache/v3"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

func addSubscriber(t *testing.T, p *Producer, name, addr string) {
	t.Helper()
	require.NoError(t, p.Extract(discardCtx(t), fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: newEndpoint(name, addr),
	}))
}

// extractorProducerWithCleanup builds a discovery-enabled producer and registers
// its subscriber-manager shutdown so subscriber goroutines are torn down.
func extractorProducerWithCleanup(t *testing.T, discoverPods bool) *Producer {
	t.Helper()
	p := newExtractorProducer(discoverPods)
	t.Cleanup(func() { p.subscribersManager.Shutdown(discardCtx(t)) })
	return p
}

func TestProducer_DumpState(t *testing.T) {
	tests := []struct {
		name   string
		build  func(t *testing.T) *Producer
		assert func(t *testing.T, got preciseState)
	}{
		{
			name:  "empty",
			build: func(t *testing.T) *Producer { return extractorProducerWithCleanup(t, true) },
			assert: func(t *testing.T, got preciseState) {
				require.Equal(t, preciseState{
					BlockSizeTokens:      0,
					PodDiscoveryEnabled:  true,
					Subscribers:          []string{},
					TotalSubscribers:     0,
					MaxSubscribers:       maxDebugDumpSubscribers,
					SubscribersTruncated: false,
					SpeculativeEnabled:   false,
					SpeculativeEntries:   0,
				}, got)
			},
		},
		{
			name: "populated subscribers sorted",
			build: func(t *testing.T) *Producer {
				p := extractorProducerWithCleanup(t, true)
				// Add out of order; the dump must sort.
				addSubscriber(t, p, "pod-b", "10.0.0.2")
				addSubscriber(t, p, "pod-a", "10.0.0.1")
				return p
			},
			assert: func(t *testing.T, got preciseState) {
				require.Equal(t, []string{"ns/pod-a", "ns/pod-b"}, got.Subscribers)
				require.Equal(t, 2, got.TotalSubscribers)
				require.False(t, got.SubscribersTruncated)
			},
		},
		{
			name: "subscribers capped",
			build: func(t *testing.T) *Producer {
				p := extractorProducerWithCleanup(t, true)
				for i := 0; i < maxDebugDumpSubscribers+5; i++ {
					addSubscriber(t, p, fmt.Sprintf("pod-%03d", i), fmt.Sprintf("10.1.%d.%d", i/256, i%256))
				}
				return p
			},
			assert: func(t *testing.T, got preciseState) {
				require.True(t, got.SubscribersTruncated)
				require.Equal(t, maxDebugDumpSubscribers+5, got.TotalSubscribers)
				require.Equal(t, maxDebugDumpSubscribers, got.MaxSubscribers)
				require.Len(t, got.Subscribers, maxDebugDumpSubscribers)
				// Sorted-and-capped head: lexicographically smallest names retained.
				require.Equal(t, "ns/pod-000", got.Subscribers[0])
			},
		},
		{
			name: "speculative enabled",
			build: func(t *testing.T) *Producer {
				cache := ttlcache.New[string, *speculativeEntries](
					ttlcache.WithTTL[string, *speculativeEntries](time.Minute),
				)
				cache.Set("req-1", &speculativeEntries{}, ttlcache.DefaultTTL)
				cache.Set("req-2", &speculativeEntries{}, ttlcache.DefaultTTL)
				return &Producer{
					typedName:          plugin.TypedName{Type: PluginType, Name: PluginType},
					speculativeEnabled: true,
					speculativeCache:   cache,
				}
			},
			assert: func(t *testing.T, got preciseState) {
				require.True(t, got.SpeculativeEnabled)
				require.Equal(t, 2, got.SpeculativeEntries)
				require.Empty(t, got.Subscribers)
			},
		},
		{
			name: "pod discovery disabled",
			build: func(t *testing.T) *Producer {
				p := extractorProducerWithCleanup(t, false)
				// Extract is a no-op when discovery is off.
				addSubscriber(t, p, "pod-a", "10.0.0.1")
				return p
			},
			assert: func(t *testing.T, got preciseState) {
				require.False(t, got.PodDiscoveryEnabled)
				require.Equal(t, 0, got.TotalSubscribers)
				require.Empty(t, got.Subscribers)
			},
		},
		{
			name:  "nil managers no panic",
			build: func(t *testing.T) *Producer { return &Producer{} },
			assert: func(t *testing.T, got preciseState) {
				require.Equal(t, preciseState{
					Subscribers:    []string{},
					MaxSubscribers: maxDebugDumpSubscribers,
				}, got)
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := tc.build(t)
			payload, err := p.DumpState()
			require.NoError(t, err)
			require.True(t, json.Valid(payload))
			var got preciseState
			require.NoError(t, json.Unmarshal(payload, &got))
			tc.assert(t, got)
		})
	}
}
