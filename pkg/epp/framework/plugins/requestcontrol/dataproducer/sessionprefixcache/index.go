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
	"sync"

	lru "github.com/hashicorp/golang-lru/v2"
	k8stypes "k8s.io/apimachinery/pkg/types"
)

// serverID identifies a model-serving pod by its NamespacedName.
type serverID k8stypes.NamespacedName

func (s serverID) String() string {
	return k8stypes.NamespacedName(s).String()
}

// podSet holds the pods believed to hold a given chain hash.
type podSet map[serverID]struct{}

// index maps each chain hash to the pods that have served it, backed by a
// per-pod LRU so total memory scales with pod count rather than with unique
// sessions. It mirrors the approximate prefix-cache indexer: the forward map
// answers a lookup in one pass over the chain instead of one pass per candidate
// pod, and each pod's LRU eviction callback keeps the forward map consistent.
type index struct {
	mu             sync.RWMutex
	hashToPods     map[uint64]podSet
	podToLRU       map[serverID]*lru.Cache[uint64, struct{}]
	defaultLRUSize int
}

func newIndex(defaultLRUSize int) *index {
	return &index{
		hashToPods:     make(map[uint64]podSet),
		podToLRU:       make(map[serverID]*lru.Cache[uint64, struct{}]),
		defaultLRUSize: defaultLRUSize,
	}
}

// Add records that srv has served every chunk in chain. Re-adding a hash also
// refreshes its LRU recency, so a chain that keeps being served stays resident
// for as long as the traffic lasts.
func (i *index) Add(chain []uint64, srv serverID) {
	if len(chain) == 0 {
		return
	}
	i.mu.Lock()
	defer i.mu.Unlock()

	c, ok := i.podToLRU[srv]
	if !ok {
		// NewWithEvict only errors on a non-positive size, rejected at construction.
		c, _ = lru.NewWithEvict(i.defaultLRUSize, i.makeEvictionFn(srv))
		i.podToLRU[srv] = c
	}

	for _, h := range chain {
		c.Add(h, struct{}{})
		pods := i.hashToPods[h]
		if pods == nil {
			pods = make(podSet)
			i.hashToPods[h] = pods
		}
		pods[srv] = struct{}{}
	}
}

// makeEvictionFn drops the pod from the forward map when its LRU evicts a hash.
// It runs inside an LRU mutation, so the caller already holds i.mu.
func (i *index) makeEvictionFn(srv serverID) func(uint64, struct{}) {
	return func(h uint64, _ struct{}) {
		pods, ok := i.hashToPods[h]
		if !ok {
			return
		}
		delete(pods, srv)
		if len(pods) == 0 {
			delete(i.hashToPods, h)
		}
	}
}

// LongestPrefixes returns, for each candidate, how many leading chain hashes it
// currently holds. Because each hash chains in every prior chunk, a run that
// breaks at position i means the byte prefix diverged at chunk i; that
// candidate's walk stops there.
//
// The chain is walked once, dropping candidates as they diverge, so the cost is
// bounded by the chain length rather than by chain length times pod count.
func (i *index) LongestPrefixes(chain []uint64, candidates []serverID) map[serverID]int {
	matches := make(map[serverID]int, len(candidates))
	if len(chain) == 0 || len(candidates) == 0 {
		return matches
	}

	i.mu.RLock()
	defer i.mu.RUnlock()

	alive := make(map[serverID]struct{}, len(candidates))
	for _, srv := range candidates {
		alive[srv] = struct{}{}
	}

	for _, h := range chain {
		if len(alive) == 0 {
			break
		}
		holders := i.hashToPods[h]
		for srv := range alive {
			if _, ok := holders[srv]; ok {
				matches[srv]++
				continue
			}
			delete(alive, srv)
		}
	}
	return matches
}

// RemovePod drops all state for a pod that has left the pool.
func (i *index) RemovePod(srv serverID) {
	i.mu.Lock()
	defer i.mu.Unlock()

	if c, ok := i.podToLRU[srv]; ok {
		c.Purge() // fires the eviction callback for every hash, clearing hashToPods
		delete(i.podToLRU, srv)
	}
}

// Pods returns the set of pods currently tracked.
func (i *index) Pods() []serverID {
	i.mu.RLock()
	defer i.mu.RUnlock()

	pods := make([]serverID, 0, len(i.podToLRU))
	for srv := range i.podToLRU {
		pods = append(pods, srv)
	}
	return pods
}
