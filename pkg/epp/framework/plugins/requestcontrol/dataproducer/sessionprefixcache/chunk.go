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
	"unicode/utf8"

	"github.com/cespare/xxhash/v2"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// fieldSep frames each content segment so that role, API surface, and message
// boundary all contribute to the byte stream. Two requests whose plaintext
// concatenation is equal but whose framing differs (e.g. [user:"ab"] vs
// [user:"a", assistant:"b"]) therefore hash to different chains and do not
// falsely share affinity.
const fieldSep = 0x1f // ASCII unit separator

// contentStream concatenates the request's textual content into a single framed
// byte stream. Each segment is framed as
// sep + apiSurface + sep + role + sep + plainText so structurally distinct
// requests never collapse to the same bytes. Non-text content (images, audio)
// is ignored: only the text carried by PlainText contributes.
func contentStream(b *fwkrh.InferenceRequestBody) []byte {
	if b == nil {
		return nil
	}

	var buf bytes.Buffer
	seg := func(surface, role, text string) {
		if text == "" {
			return
		}
		buf.WriteByte(fieldSep)
		buf.WriteString(surface)
		buf.WriteByte(fieldSep)
		buf.WriteString(role)
		buf.WriteByte(fieldSep)
		buf.WriteString(text)
	}

	if b.Completions != nil {
		seg("completions", "", b.Completions.Prompt.PlainText())
	}
	if b.ChatCompletions != nil {
		for _, m := range b.ChatCompletions.Messages {
			seg("chat", m.Role, m.Content.PlainText())
		}
	}
	if b.Messages != nil {
		// Anthropic carries the system prompt as a top-level field, not a
		// role:"system" message; it must contribute to the chain.
		seg("anthropic", "system", anthropicText(b.Messages.System))
		for _, m := range b.Messages.Messages {
			seg("anthropic", m.Role, anthropicText(m.Content))
		}
	}
	if b.Responses != nil {
		seg("responses", "instructions", anyText(b.Responses.Instructions))
		seg("responses", "input", anyText(b.Responses.Input))
	}
	if b.Conversations != nil {
		for _, it := range b.Conversations.Items {
			seg("conversations", it.Role, anyText(it.Content))
		}
	}
	return buf.Bytes()
}

// anthropicText renders an AnthropicContent as plain text, mirroring the
// text-only extraction the OpenAI Content.PlainText performs.
func anthropicText(c fwkrh.AnthropicContent) string {
	if c.Raw != "" {
		return c.Raw
	}
	var buf bytes.Buffer
	for _, blk := range c.Structured {
		if blk.Type == "text" {
			buf.WriteString(blk.Text)
			buf.WriteByte(' ')
		}
	}
	return buf.String()
}

// anyText renders a free-form JSON value (Responses input/instructions or a
// Conversations item body) as text: a string verbatim, anything else as its
// canonical JSON encoding so structured input still yields a stable chain.
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

// chunkChain splits the content stream into complete, rune-safe chunks of at
// least chunkSize bytes and returns the running chain hash of each chunk. The
// trailing partial chunk (fewer than chunkSize bytes) is dropped: sub-chunk
// growth carries no reuse signal, matching how a model server's KV block only
// becomes reusable once a full block is filled.
//
// The chain is seeded by (model, salt, declaredID, chunk0) and each subsequent
// hash folds in the previous hash, so a matching hash at position i proves the
// entire byte prefix through chunk i is identical. declaredID only seeds the
// root; byte equality of every chunk is what grants a match.
func chunkChain(stream []byte, model, salt, declaredID string, chunkSize, maxChunks int) []uint64 {
	if chunkSize <= 0 || len(stream) < chunkSize {
		return nil
	}

	var chain []uint64
	var prev uint64
	for i := 0; i < len(stream) && len(chain) < maxChunks; {
		end := i
		for end < len(stream) && end-i < chunkSize {
			_, size := utf8.DecodeRune(stream[end:])
			end += size
		}
		if end-i < chunkSize {
			break // trailing partial chunk: dropped
		}
		chunk := stream[i:end]

		var h uint64
		if len(chain) == 0 {
			h = hashRoot(model, salt, declaredID, chunk)
		} else {
			h = hashNext(prev, chunk)
		}
		chain = append(chain, h)
		prev = h
		i = end
	}
	return chain
}

// hashRoot hashes the first chunk together with the routing identity. The
// declared id is folded in here only, so it seeds the chain root without ever
// substituting for byte equality of the content itself.
func hashRoot(model, salt, declaredID string, chunk []byte) uint64 {
	d := xxhash.New()
	writeSeeded(d, model)
	writeSeeded(d, salt)
	writeSeeded(d, declaredID)
	_, _ = d.Write(chunk)
	return d.Sum64()
}

// hashNext chains a chunk onto the previous hash.
func hashNext(prev uint64, chunk []byte) uint64 {
	var le [8]byte
	binary.LittleEndian.PutUint64(le[:], prev)
	d := xxhash.New()
	_, _ = d.Write(le[:])
	_, _ = d.Write(chunk)
	return d.Sum64()
}

// writeSeeded writes s followed by a NUL delimiter so that adjacent seed fields
// cannot run together (e.g. model "ab"+salt "c" must differ from "a"+"bc").
func writeSeeded(d *xxhash.Digest, s string) {
	_, _ = d.WriteString(s)
	_, _ = d.Write([]byte{0})
}
