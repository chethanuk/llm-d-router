# Session Prefix Cache Producer

`session-prefix-cache-producer` produces `PrefixCacheMatchInfo` for prefix-cache
affinity at session granularity, without a tokenizer and against a stock
inference engine. It complements `approx-prefix-cache-producer` (token-block
hashing) and `precise-prefix-cache-producer` (engine KV events): those need a
tokenizer or engine cooperation, this one needs only the request content already
parsed by the router.

## How it works

For each request the producer builds a **content chain**:

1. **Frame** everything the engine will tokenize into a single byte stream.
   Every segment contributes its API surface, role, and text as length-prefixed
   fields, so a `[user:"ab"]` request and a `[user:"a", assistant:"b"]` request
   hash differently even though their concatenated text is equal. Lengths rather
   than a delimiter keep the boundary out of the client's reach: any byte a
   delimiter reserved could otherwise be sent as content to forge a frame.
   Chat, Completions, Anthropic Messages (including the top-level `System`),
   Responses, and Conversations bodies all contribute, as do tool definitions and
   the tool-call trace: on an agentic surface those are the bulk of the prompt,
   and leaving them out would let two sessions sharing a system prompt collapse
   onto one chain while their engine KV had diverged thousands of tokens earlier.
   Images and audio are framed by a digest of their source rather than by their
   bytes, so they stay distinguishable while the byte count is marked as not
   measuring them.
2. **Chunk** the stream into complete, rune-safe chunks of `chunkSizeBytes`
   (default 512). The trailing partial chunk is dropped: a chunk carries reuse
   signal only once it is full, mirroring how a model server's KV block becomes
   reusable only when filled.
3. **Chain-hash** the chunks with `xxhash`, seeded by the target model and the
   request's cache salt, each chunk folding in the previous hash. A hash match at
   position `i` therefore proves the framed prefix through chunk `i` is identical,
   which is the condition under which the engine can reuse KV. How closely the
   frame tracks the engine's rendered prompt is bounded by what the router
   parsed; see Limits.

Nothing outside the content itself enters the hash, with two exceptions that
mirror how the tokenized producer and vLLM scope a prefix cache: the **target
model**, because its KV is not portable, and **`cache_salt`**, because it is the
client's explicit request for cache isolation.

### Declared session ids

A declared client id keys the chain; it never seeds it. Two byte-identical
prompts carrying different session names still produce the same chain and still
match, which is the template-sharing traffic this producer exists to serve.

The id matters on the surfaces where the engine holds the conversation. A client
on `/v1/responses` or `/v1/conversations` sends only the newest turn, so its
content shares no bytes with the previous turn and the content chain alone would
start a fresh lineage every time. There the producer looks up the chain last
served under the declared id and appends the new turn to it, which keeps the
session on the pod that has been serving it. If the client resends history
instead, the content chain already covers every earlier turn and the alias stays
out of the way.

On Chat Completions, Completions, and Anthropic Messages the client sends the
whole prompt every turn, so the content chain is what was actually sent and the
alias is never consulted. A turn that edits earlier history loses the affinity it
no longer shares rather than inheriting it from the id.

Ids are read from `prompt_cache_key` and `conversation` in the body, and from the
`agent-identity` plugin's request attribute; enabling that plugin is what turns
on aliasing by session header. The alias key carries the same model and cache
salt the chain does, so a session name reused under another model or salt cannot
inherit a lineage across the isolation boundary the salt draws.

`previous_response_id` names the previous turn rather than the session, so it
changes every turn. Grouping turns by it needs the response id recorded against
the chain that produced it, which `requestcontrol.Response` does not carry.

### Index and hooks

A per-pod LRU index records which chains were served to which endpoint, with a
forward `hash -> pods` map so a request is scored in one pass over the chain
rather than one pass per candidate pod. `Produce` scores each candidate pod by
its longest cached prefix and hands the resolved chain to `PreRequest` through
the shared plugin state, so concurrent turns of one session cannot score against
one lineage and index another. `PreRequest` seeds the served endpoint of every
scheduling profile, so P/D-disaggregated prefill nodes gain affinity too.
`ResponseBody` calibrates the bytes-per-token ratio from the prompt-token count
the engine reports, which is what converts a chunk size in bytes into the block
size in tokens that `PrefixCacheMatchInfo` consumers use to price a match.

Calibration only counts a reported count that measures the same content the
router hashed. Anthropic reports `input_tokens` net of cached and cache-creation
blocks, and a request carrying images spends tokens on content the stream names
by digest, so neither is a usable denominator and both are skipped.

## Configuration

| Field | Default | Description |
|---|---|---|
| `chunkSizeBytes` | `512` | Minimum size of a complete content chunk. |
| `maxChunks` | `2048` | Cap on chunks a single chain carries, a megabyte of content at the default chunk size. |
| `maxEntriesPerPod` | `100000` | Bound on the per-pod LRU of chain hashes. |
| `maxAliasedSessions` | `4096` | Bound on how many declared session ids keep a remembered chain. |

The producer is Alpha, so the EPP must be started with
`--allow-experimental-plugins`. Bind it to a prefix-cache scorer via
`prefixMatchInfoProducerName` set to this producer's instance name; it coexists
with the approximate and precise producers under distinct named keys.

```yaml
- type: prefix-cache-scorer
  parameters:
    prefixMatchInfoProducerName: session-prefix-cache-producer
```

## Limits

The index records estimates. A `PreRequest` seed says where a turn was routed
rather than what the engine confirms it holds. Prefill covers the whole prompt,
so the seed is accurate once the request completes and decays from there as the
engine evicts under pressure, which means a request can be routed warm and
prefill cold. The `cached_tokens` an engine reports measures the state before it
served, which serving supersedes, so it cannot correct the entry it arrives
with. Turning it into a retention signal, and folding in engine-labeled KV
events for truthful extents, are tracked separately in the umbrella issue.

Identity assumes a turn extends the previous turn's bytes. That holds for
straightforward multi-turn chat, and does not hold where the engine's rendered
prompt is not append-only, most commonly when a model's chat template drops
earlier reasoning at a turn boundary. There the chain keeps extending while the
engine's real prefix has diverged, and the result is a partial hit rather than
the full one the score predicted. The request still completes; it prefills where
it could have hit cache.

An aliased session appends to its chain every turn and is never told the
conversation was compacted, so a session running past `maxChunks` freezes its
chain at the cap and keeps routing to the pods that served it.

The bytes-per-token ratio is a single value across the pool. An EPP fronting
models with markedly different tokenizers prices every request at their blend.

Content that fills no complete chunk and belongs to no known session carries no
signal, so short prompts score zero everywhere and fall through to the other
scorers.

Structured fields reach the stream re-serialized, not as the bytes the client
sent: the router parses the body before this producer sees it, so key order and
number form are already normalized on the chat and Responses surfaces. Two
requests whose tool definitions differ only in key order therefore hash alike,
while the engine's template renders them differently. Anthropic tool schemas and
tool inputs keep their wire bytes and are exact.

A seed is written when a turn is routed, not when it succeeds. A request that
never reaches the engine still leaves its chain against the pod it was sent to,
and the next turn of that session is steered back to it. The entry ages out
under LRU pressure rather than being withdrawn.

Declared session ids share one namespace per model and salt. Two clients that
send the same `prompt_cache_key` are treated as one session, so a deployment
serving mutually untrusted tenants from one pool should set `cache_salt`, which
scopes the chain root and the alias with it.

The chain covers the conversational surfaces. Embeddings, pre-tokenized
`generate`, and image-generation bodies contribute nothing, so a request on one
of them scores zero on every endpoint rather than being placed on a wrong one.
