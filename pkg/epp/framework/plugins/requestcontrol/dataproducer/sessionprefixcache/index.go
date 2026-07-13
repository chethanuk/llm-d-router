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

// ServerID identifies a model-serving pod by its NamespacedName.
type ServerID k8stypes.NamespacedName

func (s ServerID) String() string {
	return k8stypes.NamespacedName(s).String()
}

// provenance records how a chain hash entered the index. Estimated entries come
// from the pre-request seeding (a chunk count derived from content length);
// Confirmed entries are backed by the served response's reported prompt-token
// usage. Confirmed outranks Estimated: it is never overwritten by an estimate,
// and only Estimated entries are trimmed when a response refines the count down.
type provenance uint8

const (
	estimated provenance = iota
	confirmed
)

// index is a per-pod LRU of chain hashes with longest-prefix lookup. Each pod's
// cache is bounded independently so total memory scales with pod count, not with
// unique sessions. It mirrors the approx prefix-cache indexer's per-pod-LRU
// design, adding provenance so an over-estimated tail can be walked back.
type index struct {
	mu             sync.RWMutex
	podToLRU       map[ServerID]*lru.Cache[uint64, provenance]
	defaultLRUSize int
}

func newIndex(defaultLRUSize int) *index {
	return &index{
		podToLRU:       make(map[ServerID]*lru.Cache[uint64, provenance]),
		defaultLRUSize: defaultLRUSize,
	}
}

// cacheFor returns the pod's LRU, creating it on first use.
func (i *index) cacheFor(srv ServerID) *lru.Cache[uint64, provenance] {
	if c, ok := i.podToLRU[srv]; ok {
		return c
	}
	// NewWithEvict only errors on a non-positive size, guarded at construction.
	c, _ := lru.New[uint64, provenance](i.defaultLRUSize)
	i.podToLRU[srv] = c
	return c
}

// Add records the chain hashes for srv at the given provenance. A Confirmed
// entry is never downgraded to Estimated, so repeated pre-request seeding cannot
// erase what a response has already confirmed.
func (i *index) Add(chain []uint64, srv ServerID, prov provenance) {
	if len(chain) == 0 {
		return
	}
	i.mu.Lock()
	defer i.mu.Unlock()

	c := i.cacheFor(srv)
	for _, h := range chain {
		if prov == estimated {
			if cur, ok := c.Peek(h); ok && cur == confirmed {
				continue
			}
		}
		c.Add(h, prov)
	}
}

// TrimEstimatedTail removes the given hashes from srv's cache only when they are
// still Estimated. It refines an over-estimate downward without disturbing any
// prefix a response has confirmed.
func (i *index) TrimEstimatedTail(chain []uint64, srv ServerID) {
	if len(chain) == 0 {
		return
	}
	i.mu.Lock()
	defer i.mu.Unlock()

	c, ok := i.podToLRU[srv]
	if !ok {
		return
	}
	for _, h := range chain {
		if cur, ok := c.Peek(h); ok && cur == estimated {
			c.Remove(h)
		}
	}
}

// LongestPrefix returns the number of leading chain hashes srv currently caches.
// Because each hash chains in every prior chunk, a run that breaks at position i
// means the byte prefix diverged at chunk i; the walk stops there.
func (i *index) LongestPrefix(chain []uint64, srv ServerID) int {
	i.mu.RLock()
	defer i.mu.RUnlock()

	c, ok := i.podToLRU[srv]
	if !ok {
		return 0
	}
	n := 0
	for _, h := range chain {
		if !c.Contains(h) {
			break
		}
		n++
	}
	return n
}

// RemovePod drops all state for a pod that has left the pool.
func (i *index) RemovePod(srv ServerID) {
	i.mu.Lock()
	defer i.mu.Unlock()
	delete(i.podToLRU, srv)
}

// Pods returns the set of pods currently tracked.
func (i *index) Pods() []ServerID {
	i.mu.RLock()
	defer i.mu.RUnlock()

	pods := make([]ServerID, 0, len(i.podToLRU))
	for srv := range i.podToLRU {
		pods = append(pods, srv)
	}
	return pods
}
