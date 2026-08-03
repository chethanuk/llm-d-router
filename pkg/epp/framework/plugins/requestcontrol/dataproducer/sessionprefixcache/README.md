# Session Prefix Cache Producer

`session-prefix-cache-producer` produces `PrefixCacheMatchInfo` for prefix-cache
affinity at session granularity, without a tokenizer and against a stock
inference engine. It complements `approx-prefix-cache-producer` (token-block
hashing) and `precise-prefix-cache-producer` (engine KV-events): those need a
tokenizer or engine cooperation, this one needs only the request content already
parsed by the router.

## How it works

For each request the producer builds a **content chain**:

1. **Frame** the textual content into a single byte stream. Every segment is
   framed as `US + apiSurface + US + role + US + text` (`US` = `0x1f`), so a
   `[user:"ab"]` request and a `[user:"a", assistant:"b"]` request hash
   differently even though their concatenated text is equal. Chat, Completions,
   Anthropic Messages (including the top-level `System`), Responses, and
   Conversations bodies all contribute; non-text (image/audio) content is
   ignored.
2. **Chunk** the stream into complete, rune-safe chunks of `chunkSizeBytes`
   (default 512). The trailing partial chunk is dropped: a chunk carries reuse
   signal only once it is full, mirroring how a model server's KV block becomes
   reusable only when filled.
3. **Chain-hash** the chunks with `xxhash`. The root chunk is seeded with the
   target model, cache salt, and a declared session id; each later chunk folds
   in the previous hash. A hash match at position `i` therefore proves the whole
   byte prefix through chunk `i` is identical. The declared id only seeds the
   root — byte equality of every chunk is what grants a match, so two sessions
   that declare the same id but diverge in content match only their shared
   leading chunks.

The **declared session id** is resolved with this precedence: the body
`prompt_cache_key`, then the configured session headers (defaulting to the
`agent-identity` header set). When none is present the chain is seeded from
content alone; absent-session requests are never collapsed onto a tenant or
account key.

A per-pod LRU index records which chains were served to which endpoint.
`Produce` scores each candidate pod by its longest cached prefix; `PreRequest`
synchronously seeds the served endpoint of every scheduling profile (so
P/D-disaggregated prefill nodes gain affinity too); `ResponseBody` reads the
served response's reported prompt-token usage to **confirm** the prefix the
engine actually cached and **trim** any over-estimated tail, so the index
refines downward rather than only growing.

## Configuration

| Field | Default | Description |
|---|---|---|
| `chunkSizeBytes` | `512` | Minimum size of a complete content chunk. |
| `maxChunks` | `256` | Cap on leading chunks a single request contributes. |
| `maxEntriesPerPod` | `100000` | Bound on the per-pod LRU of chain hashes. |
| `sessionHeaders` | agent-identity set | Ordered request headers consulted for a session id when the body carries no `prompt_cache_key`. |

Bind it to a prefix-cache scorer via `prefixMatchInfoProducerName` set to this
producer's instance name; it coexists with the approximate and precise
producers under distinct named keys.
