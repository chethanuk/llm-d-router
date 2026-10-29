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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// forwardMapSize reports how many hashes the forward map still tracks. A hash
// the per-pod LRU has dropped must not survive here: the forward map is what
// LongestPrefixes reads, so a stale entry both leaks memory for the lifetime of
// the process and reports a match against a pod the LRU no longer claims.
func forwardMapSize(i *index) int {
	i.mu.RLock()
	defer i.mu.RUnlock()
	return len(i.hashToPods)
}

func TestIndex_EvictionKeepsForwardMapConsistent(t *testing.T) {
	const lruSize = 4
	podA, podB := serverID{Namespace: "ns", Name: "a"}, serverID{Namespace: "ns", Name: "b"}

	cases := []struct {
		name        string
		chains      [][]uint64
		srv         []serverID
		wantForward int
	}{
		{
			// Under the cap nothing is evicted.
			name:        "within capacity",
			chains:      [][]uint64{{1, 2, 3}},
			srv:         []serverID{podA},
			wantForward: 3,
		},
		{
			// The second chain pushes the first past the cap. The evicted hashes
			// must leave the forward map with it.
			name:        "overflow evicts the oldest hashes",
			chains:      [][]uint64{{1, 2, 3, 4}, {5, 6, 7, 8}},
			srv:         []serverID{podA, podA},
			wantForward: lruSize,
		},
		{
			// Two pods each get their own LRU, so neither evicts the other's
			// entries and a shared hash keeps both holders.
			name:        "per-pod capacity is independent",
			chains:      [][]uint64{{1, 2, 3, 4}, {1, 2, 3, 4}},
			srv:         []serverID{podA, podB},
			wantForward: 4,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			idx := newIndex(lruSize)
			for n, chain := range tc.chains {
				idx.Add(chain, tc.srv[n])
			}
			assert.Equal(t, tc.wantForward, forwardMapSize(idx))
		})
	}
}

func TestIndex_EvictedHashStopsMatching(t *testing.T) {
	podA := serverID{Namespace: "ns", Name: "a"}
	idx := newIndex(2)

	idx.Add([]uint64{1, 2}, podA)
	require.Equal(t, 2, idx.LongestPrefixes([]uint64{1, 2}, []serverID{podA})[podA])

	// Serving a different chain evicts the first one.
	idx.Add([]uint64{3, 4}, podA)
	assert.Equal(t, 0, idx.LongestPrefixes([]uint64{1, 2}, []serverID{podA})[podA],
		"an evicted chain must not keep reporting a match")
	assert.Equal(t, 2, idx.LongestPrefixes([]uint64{3, 4}, []serverID{podA})[podA])
	assert.Equal(t, 2, forwardMapSize(idx))
}

func TestIndex_LongestPrefixesStopsAtDivergence(t *testing.T) {
	podA, podB := serverID{Namespace: "ns", Name: "a"}, serverID{Namespace: "ns", Name: "b"}
	idx := newIndex(16)
	// Two sessions sharing a first chunk and diverging after it.
	idx.Add([]uint64{1, 2, 3, 4}, podA)
	idx.Add([]uint64{1, 5}, podB)

	got := idx.LongestPrefixes([]uint64{1, 2, 3}, []serverID{podA, podB})
	assert.Equal(t, 3, got[podA], "podA holds the whole queried prefix")
	assert.Equal(t, 1, got[podB], "podB diverges after the shared first chunk")

	// A pod holding nothing of the chain reports no match rather than being absent
	// in a way a caller could read as a full match.
	podC := serverID{Namespace: "ns", Name: "c"}
	assert.Equal(t, 0, idx.LongestPrefixes([]uint64{1, 2, 3}, []serverID{podC})[podC])
}

func TestIndex_RemovePodClearsItsHashes(t *testing.T) {
	podA, podB := serverID{Namespace: "ns", Name: "a"}, serverID{Namespace: "ns", Name: "b"}
	idx := newIndex(16)
	idx.Add([]uint64{1, 2, 3}, podA)
	idx.Add([]uint64{1, 2, 3}, podB)

	idx.RemovePod(podA)

	assert.ElementsMatch(t, []serverID{podB}, idx.Pods())
	assert.Equal(t, 0, idx.LongestPrefixes([]uint64{1, 2, 3}, []serverID{podA})[podA])
	assert.Equal(t, 3, idx.LongestPrefixes([]uint64{1, 2, 3}, []serverID{podB})[podB],
		"removing one pod must not disturb another holding the same chain")

	// The last holder leaving drops the hashes entirely.
	idx.RemovePod(podB)
	assert.Empty(t, idx.Pods())
	assert.Equal(t, 0, forwardMapSize(idx))
}
