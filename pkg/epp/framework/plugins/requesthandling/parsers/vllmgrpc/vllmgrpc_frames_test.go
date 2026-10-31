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

package vllmgrpc

import (
	"context"
	"slices"
	"testing"

	pb "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vllmgrpc/api/gen"
)

// A streamed response can deliver several gRPC frames in one body, so usage must
// not depend on how many frames share it or on which of them are readable.
func TestParseResponse_ReadsEveryFrameInABody(t *testing.T) {
	streamChunk := func(prompt, completion, cached uint32) []byte {
		return createGrpcPayload(t, &pb.GenerateResponse{Response: &pb.GenerateResponse_Chunk{
			Chunk: &pb.GenerateStreamChunk{TokenIds: []uint32{1, 2, 3}, PromptTokens: prompt, CompletionTokens: completion, CachedTokens: cached},
		}})
	}
	complete := func(prompt, completion, cached uint32) []byte {
		return createGrpcPayload(t, &pb.GenerateResponse{Response: &pb.GenerateResponse_Complete{
			Complete: &pb.GenerateComplete{FinishReason: "stop", PromptTokens: prompt, CompletionTokens: completion, CachedTokens: cached},
		}})
	}
	noUsage := streamChunk(0, 0, 0)
	done := complete(7, 10, 0)

	tests := []struct {
		name           string
		body           []byte
		wantUsage      bool
		wantPrompt     int
		wantCompletion int
		wantTotal      int
		wantErr        bool
	}{
		{
			name: "empty body reports no usage",
		},
		{
			name:           "usage in the second of two frames is read",
			body:           slices.Concat(noUsage, done),
			wantUsage:      true,
			wantPrompt:     7,
			wantCompletion: 10,
			wantTotal:      17,
		},
		{
			name:           "usage in the third of three frames is read",
			body:           slices.Concat(noUsage, noUsage, done),
			wantUsage:      true,
			wantPrompt:     7,
			wantCompletion: 10,
			wantTotal:      17,
		},
		{
			name:           "last usage wins over earlier usage",
			body:           slices.Concat(streamChunk(7, 1, 0), done),
			wantUsage:      true,
			wantPrompt:     7,
			wantCompletion: 10,
			wantTotal:      17,
		},
		{
			name:           "later frame without usage keeps earlier usage",
			body:           slices.Concat(done, noUsage),
			wantUsage:      true,
			wantPrompt:     7,
			wantCompletion: 10,
			wantTotal:      17,
		},
		{
			name:           "later frame with zero prompt keeps the prompt count",
			body:           slices.Concat(streamChunk(7, 1, 3), streamChunk(0, 1, 0)),
			wantUsage:      true,
			wantPrompt:     7,
			wantCompletion: 1,
			wantTotal:      8,
		},
		{
			name:           "later frame with zero completion keeps the completion count",
			body:           slices.Concat(streamChunk(7, 1, 3), complete(7, 0, 3)),
			wantUsage:      true,
			wantPrompt:     7,
			wantCompletion: 1,
			wantTotal:      8,
		},
		{
			name:       "embed response still falls back",
			body:       createGrpcPayload(t, &pb.EmbedResponse{PromptTokens: 9}),
			wantUsage:  true,
			wantPrompt: 9,
			wantTotal:  9,
		},
		{
			name:           "partial last frame keeps earlier usage and reports an error",
			body:           slices.Concat(done, noUsage[:7]),
			wantUsage:      true,
			wantPrompt:     7,
			wantCompletion: 10,
			wantTotal:      17,
			wantErr:        true,
		},
		{
			name:           "compressed last frame keeps earlier usage and reports an error",
			body:           slices.Concat(done, []byte{1, 0, 0, 0, 1, 9}),
			wantUsage:      true,
			wantPrompt:     7,
			wantCompletion: 10,
			wantTotal:      17,
			wantErr:        true,
		},
		{
			name:           "unreadable frame between usage frames is skipped and reported",
			body:           slices.Concat(streamChunk(7, 1, 0), []byte{0, 0, 0, 0, 1, 0xff}, done),
			wantUsage:      true,
			wantPrompt:     7,
			wantCompletion: 10,
			wantTotal:      17,
			wantErr:        true,
		},
		{
			name:    "claimed length past the body reports an error with no usage",
			body:    []byte{0, 0xff, 0xff, 0xff, 0xff, 1},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewVllmGRPCParser().ParseResponse(context.Background(), tt.body, map[string]string{}, false)

			if (err != nil) != tt.wantErr {
				t.Errorf("error = %v, wantErr %v", err, tt.wantErr)
			}
			if got == nil {
				t.Fatal("want a parsed response even when a frame is unreadable, got nil")
			}
			if got.StreamedEvents != 0 {
				t.Errorf("StreamedEvents = %d, want 0", got.StreamedEvents)
			}
			if !tt.wantUsage {
				if got.Usage != nil {
					t.Errorf("want no usage, got %+v", got.Usage)
				}
				return
			}
			if got.Usage == nil {
				t.Fatal("want usage, got nil")
			}
			if got.Usage.PromptTokens != tt.wantPrompt || got.Usage.CompletionTokens != tt.wantCompletion || got.Usage.TotalTokens != tt.wantTotal {
				t.Errorf("usage = prompt %d, completion %d, total %d; want prompt %d, completion %d, total %d",
					got.Usage.PromptTokens, got.Usage.CompletionTokens, got.Usage.TotalTokens,
					tt.wantPrompt, tt.wantCompletion, tt.wantTotal)
			}
		})
	}
}
