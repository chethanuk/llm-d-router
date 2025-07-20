//go:build !ignore

// Package kvevents contains the implementation of the KV-events processing
// system. It provides a pool for handling events coming from a distributed
// KV-cache pool, allowing for cache-tracking and real-time updates
// to the KV-cache index. The package is designed to work with the
// kvcache.Indexer to maintain an up-to-date state of the KV-cache
// and to facilitate the scoring of pods based on the KV-cache index state.
package kvevents
