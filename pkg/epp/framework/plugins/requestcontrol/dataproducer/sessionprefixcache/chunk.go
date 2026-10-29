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
	"bytes"
	"encoding/binary"
	"encoding/json"
	"strconv"
	"unicode/utf8"

	"github.com/cespare/xxhash/v2"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// Each content segment is framed as a sequence of length-prefixed fields, so a
// [user:"ab"] request and a [user:"a", assistant:"b"] request hash differently
// even though their concatenated text is equal. The length prefix is what makes
// the boundary unforgeable: a delimiter byte would let a client that embeds it
// in its own text mint a frame and claim another session's affinity.

// streamWriter accumulates the framed content stream and tracks whether every
// part of it is carried verbatim.
type streamWriter struct {
	buf bytes.Buffer
	// textOnly is false once a segment names binary content by digest instead
	// of carrying its bytes, which makes the stream length an unusable
	// denominator for the bytes-per-token calibration.
	textOnly bool
}

// seg appends one framed segment. Empty text still emits its frame, so a message
// carrying no content is distinguishable from an absent message.
func (w *streamWriter) seg(surface, role, text string) {
	w.field(surface)
	w.field(role)
	w.field(text)
}

// field writes s prefixed by its length. No field content can then be mistaken
// for a field boundary, whatever bytes the client sends.
func (w *streamWriter) field(s string) {
	var n [binary.MaxVarintLen64]byte
	w.buf.Write(n[:binary.PutUvarint(n[:], uint64(len(s)))])
	w.buf.WriteString(s)
}

// segJSON frames a structured field by its canonical JSON encoding, skipping
// values that carry nothing.
func (w *streamWriter) segJSON(surface, role string, v any) {
	switch text := anyText(v); text {
	case "", "null", "[]", "{}":
	default:
		w.seg(surface, role, text)
	}
}

// openAIContent frames one chat message's content, reporting whether any block
// produced a segment.
func (w *streamWriter) openAIContent(surface, role string, c fwkrh.Content) bool {
	if c.Raw != "" {
		w.seg(surface, role, c.Raw)
		return true
	}
	for _, blk := range c.Structured {
		switch blk.Type {
		case "image_url":
			w.textOnly = false
			w.seg(surface, role+"/image", mediaDigest(blk.ImageURL.URL))
		case "input_audio":
			w.textOnly = false
			w.seg(surface, role+"/audio", mediaDigest(blk.InputAudio.Format, blk.InputAudio.Data))
		case "video_url":
			w.textOnly = false
			w.seg(surface, role+"/video", mediaDigest(blk.VideoURL.URL))
		default:
			w.seg(surface, role+"/"+blk.Type, blk.Text)
		}
	}
	return len(c.Structured) > 0
}

// anthropicContent frames one Anthropic content value, reporting whether any
// block produced a segment. The tool-call trace and extended-thinking replay are
// prompt content on this surface and are framed alongside the text.
func (w *streamWriter) anthropicContent(role string, c fwkrh.AnthropicContent) bool {
	if c.Raw != "" {
		w.seg("anthropic", role, c.Raw)
		return true
	}
	for _, blk := range c.Structured {
		switch blk.Type {
		case "thinking":
			w.seg("anthropic", role+"/thinking", blk.Thinking)
		case "tool_use":
			w.seg("anthropic", role+"/tool_use", blk.ID)
			w.seg("anthropic", role+"/tool_name", blk.Name)
			w.seg("anthropic", role+"/tool_input", string(blk.Input))
		case "tool_result":
			w.seg("anthropic", role+"/tool_result", blk.ToolUseID)
			w.anthropicContent(role+"/tool_result", blk.Content)
		case "image":
			w.textOnly = false
			if s := blk.Source; s != nil {
				w.seg("anthropic", role+"/image", mediaDigest(s.Type, s.MediaType, s.URL, s.Data))
			} else {
				w.seg("anthropic", role+"/image", "")
			}
		default:
			w.seg("anthropic", role+"/"+blk.Type, blk.Text)
		}
	}
	return len(c.Structured) > 0
}

// contentStream concatenates everything the engine will tokenize into a single
// framed byte stream, and reports whether that content is carried verbatim.
//
// Tool definitions and the tool-call trace are part of the prompt an agent
// sends, so they are framed alongside the messages. Leaving them out would let
// two sessions that share a system prompt and a first question collapse onto one
// chain while their engine KV diverged thousands of tokens earlier, and the
// scorer would then report a full match against a pod that cannot honour it.
//
// Binary content (images, audio, video) is framed by a digest of its source
// rather than by its bytes. The digest keeps two otherwise identical prompts
// apart but does not measure the content, which is what the returned flag
// reports.
func contentStream(b *fwkrh.InferenceRequestBody) (stream []byte, textOnly bool) {
	if b == nil {
		return nil, false
	}
	w := streamWriter{textOnly: true}

	if c := b.Completions; c != nil {
		switch {
		case c.Prompt.Raw != "":
			w.seg("completions", "", c.Prompt.Raw)
		case len(c.Prompt.Strings) > 0:
			// A batch prompt is framed per element so the boundary between two
			// prompts is visible in the bytes rather than joined away.
			for _, s := range c.Prompt.Strings {
				w.seg("completions", "", s)
			}
		case len(c.Prompt.TokenIDs) > 0:
			w.segJSON("completions", "tokens", c.Prompt.TokenIDs)
		}
	}
	if cc := b.ChatCompletions; cc != nil {
		// Tools and template settings render ahead of the messages, so they are
		// framed first to keep the stream in the order the engine sees.
		w.segJSON("chat", "tools", cc.Tools)
		w.segJSON("chat", "documents", cc.Documents)
		if cc.ChatTemplate != "" {
			w.seg("chat", "template", cc.ChatTemplate)
		}
		w.segJSON("chat", "template_kwargs", cc.ChatTemplateKWArgs)
		for _, m := range cc.Messages {
			if !w.openAIContent("chat", m.Role, m.Content) {
				w.seg("chat", m.Role, "")
			}
			w.segJSON("chat", m.Role+"/tool_calls", m.ToolCalls)
		}
	}
	if ms := b.Messages; ms != nil {
		w.segJSON("anthropic", "tools", ms.Tools)
		// Anthropic carries the system prompt as a top-level field, not a
		// role:"system" message; it must contribute to the chain.
		w.anthropicContent("system", ms.System)
		for _, m := range ms.Messages {
			if !w.anthropicContent(m.Role, m.Content) {
				w.seg("anthropic", m.Role, "")
			}
		}
	}
	if r := b.Responses; r != nil {
		w.segJSON("responses", "instructions", r.Instructions)
		w.segJSON("responses", "tools", r.Tools)
		w.segJSON("responses", "input", r.Input)
	}
	if c := b.Conversations; c != nil {
		for _, it := range c.Items {
			w.seg("conversations", it.Type+"/"+it.Role, anyText(it.Content))
		}
	}
	return w.buf.Bytes(), w.textOnly
}

// continuationStream frames a server-side-history request without its preamble,
// so that appending it to a remembered lineage does not count the instructions
// and tool definitions once per turn. It frames whatever the client sent as the
// turn body: for a client that sends only its newest item that is one turn, and
// for a client that resends the conversation it is all of them.
func continuationStream(b *fwkrh.InferenceRequestBody) []byte {
	var w streamWriter
	if r := b.Responses; r != nil {
		w.segJSON("responses", "input", r.Input)
	}
	if c := b.Conversations; c != nil {
		for _, it := range c.Items {
			w.seg("conversations", it.Type+"/"+it.Role, anyText(it.Content))
		}
	}
	return w.buf.Bytes()
}

// preambleDigest hashes the part of a server-side-history request that renders
// ahead of every turn: the instructions and tool definitions. The engine holds
// the earlier turns behind that preamble, so a session that changes it cannot
// reuse any of their KV, and the digest is what keeps a remembered lineage from
// outliving the preamble it was built on.
func preambleDigest(b *fwkrh.InferenceRequestBody) uint64 {
	var w streamWriter
	if r := b.Responses; r != nil {
		w.segJSON("responses", "instructions", r.Instructions)
		w.segJSON("responses", "tools", r.Tools)
	}
	d := xxhash.New()
	_, _ = d.Write(w.buf.Bytes())
	return d.Sum64()
}

// mediaDigest names binary content by a hash of its source fields.
func mediaDigest(parts ...string) string {
	d := xxhash.New()
	for _, p := range parts {
		writeSeeded(d, p)
	}
	return strconv.FormatUint(d.Sum64(), 16)
}

// anyText renders a free-form JSON value as text: a string verbatim, anything
// else as its canonical JSON encoding. Go encodes map keys in sorted order and
// struct fields in declaration order, so the encoding is stable across requests.
func anyText(v any) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	encoded, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	return string(encoded)
}

// rootSeed derives the chain's starting hash. Only the target model and the
// request's cache salt seed it, matching how the tokenized producer and vLLM
// itself scope a prefix cache: the model because its KV is not portable, the
// salt because it is the client's explicit request for cache isolation.
func rootSeed(model, salt string) uint64 {
	d := xxhash.New()
	writeSeeded(d, model)
	writeSeeded(d, salt)
	return d.Sum64()
}

// chunkChain splits the content stream into complete, rune-safe chunks of at
// least chunkSize bytes and returns the running chain hash of each chunk,
// continuing from seed. The trailing partial chunk is dropped: sub-chunk growth
// carries no reuse signal, matching how a model server's KV block only becomes
// reusable once a full block is filled.
//
// Every hash folds in the previous one, so a matching hash at position i proves
// the entire byte prefix through chunk i is identical. Nothing outside the
// content and the seed enters the hash, so byte-identical prefixes match whoever
// sent them. Passing a previous chain's last hash as seed continues that chain,
// which is how a session whose history the client does not resend keeps growing
// one lineage.
//
// A chunk is emitted only when the bytes that decide its boundary are all
// present, so appending to a stream never moves an earlier boundary. Without
// that reserve a stream ending mid-rune would chunk differently from the same
// stream with the next turn appended, and raw tool arguments reach the stream as
// wire bytes that need not be valid UTF-8.
func chunkChain(stream []byte, seed uint64, chunkSize, maxChunks int) []uint64 {
	if chunkSize <= 0 || maxChunks <= 0 {
		return nil
	}
	// utf8.UTFMax-1 is the most continuation bytes a boundary can skip.
	reserve := utf8.UTFMax - 1
	if len(stream) < chunkSize+reserve {
		return nil
	}

	chain := make([]uint64, 0, min(len(stream)/chunkSize, maxChunks))

	d := xxhash.New()
	prev := seed
	var le [8]byte
	for i := 0; i+chunkSize+reserve <= len(stream) && len(chain) < maxChunks; {
		// Extend to the next rune boundary. UTF-8 continuation bytes match
		// 10xxxxxx, so skipping them lands on the next leading byte.
		end := i + chunkSize
		for end < i+chunkSize+reserve && stream[end]&0xC0 == 0x80 {
			end++
		}

		d.Reset()
		binary.LittleEndian.PutUint64(le[:], prev)
		_, _ = d.Write(le[:])
		_, _ = d.Write(stream[i:end])

		prev = d.Sum64()
		chain = append(chain, prev)
		i = end
	}
	return chain
}

// writeSeeded writes s followed by a NUL delimiter so that adjacent fields
// cannot run together (e.g. model "ab"+salt "c" must differ from "a"+"bc").
func writeSeeded(d *xxhash.Digest, s string) {
	_, _ = d.WriteString(s)
	_, _ = d.Write([]byte{0})
}
