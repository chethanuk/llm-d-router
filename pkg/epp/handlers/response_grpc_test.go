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

package handlers

import (
	"context"
	"encoding/binary"
	"slices"
	"strings"
	"testing"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"github.com/go-logr/logr/funcr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genproto/googleapis/api/httpbody"
	"google.golang.org/protobuf/proto"
	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vertexai"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vllmgrpc"
	pb "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/vllmgrpc/api/gen"
)

const grpcTestCap = 4 << 20

func grpcTestFrame(t *testing.T, msg proto.Message) []byte {
	t.Helper()
	payload, err := proto.Marshal(msg)
	require.NoError(t, err)
	return append(binary.BigEndian.AppendUint32([]byte{0}, uint32(len(payload))), payload...)
}

func grpcChunkMsg(prompt, completion, cached uint32) *pb.GenerateResponse {
	return &pb.GenerateResponse{Response: &pb.GenerateResponse_Chunk{
		Chunk: &pb.GenerateStreamChunk{TokenIds: []uint32{1, 2, 3}, PromptTokens: prompt, CompletionTokens: completion, CachedTokens: cached},
	}}
}

func grpcCompleteMsg(prompt, completion, cached uint32, finishReason string) *pb.GenerateResponse {
	return &pb.GenerateResponse{Response: &pb.GenerateResponse_Complete{
		Complete: &pb.GenerateComplete{FinishReason: finishReason, PromptTokens: prompt, CompletionTokens: completion, CachedTokens: cached},
	}}
}

// grpcDoneOfSize returns a frame holding usage {7, 10} whose message is exactly size bytes.
func grpcDoneOfSize(t *testing.T, size int) []byte {
	t.Helper()
	padding := size
	for {
		msg := grpcCompleteMsg(7, 10, 0, strings.Repeat("x", padding))
		got := proto.Size(msg)
		if got == size {
			return grpcTestFrame(t, msg)
		}
		padding += size - got
	}
}

type grpcRun struct {
	usage          fwkrh.Usage
	oversizedLogs  int
	parseErrors    int
	carryBeforeEnd int
}

// feedGRPCResponse delivers chunks to HandleResponseBody, then ends the stream with an empty
// chunk and checks that the end of the stream leaves nothing carried.
func feedGRPCResponse(t *testing.T, parser fwkrh.Parser, contentType string, chunks [][]byte) grpcRun {
	t.Helper()
	run := grpcRun{}
	ctx := log.IntoContext(context.Background(), funcr.New(func(_, args string) {
		if strings.Contains(args, "oversized gRPC message") {
			run.oversizedLogs++
		}
		if strings.Contains(args, "parsing response") {
			run.parseErrors++
		}
	}, funcr.Options{Verbosity: 10}))
	headers := map[string]string{}
	if contentType != "" {
		headers["content-type"] = contentType
	}
	server := NewStreamingServer(nil, &mockDirector{}, NewParserRegistry([]fwkrh.Parser{parser}, logr.Discard()), 0)
	reqCtx := &RequestContext{Request: &Request{}, Response: &Response{Headers: headers}, Parser: parser, modelServerStreaming: true}
	for _, chunk := range chunks {
		server.HandleResponseBody(ctx, reqCtx, chunk, false)
	}
	run.carryBeforeEnd = len(reqCtx.grpcFrameCarry)
	server.HandleResponseBody(ctx, reqCtx, nil, true)
	assert.Empty(t, reqCtx.grpcFrameCarry)
	run.usage = reqCtx.Usage
	return run
}

func feedVllmGRPC(t *testing.T, chunks ...[]byte) grpcRun {
	t.Helper()
	return feedGRPCResponse(t, vllmgrpc.NewVllmGRPCParser(), "application/grpc", chunks)
}

func TestGRPCUsageIndependentOfChunkBoundaries(t *testing.T) {
	type want struct{ prompt, completion, total, cached int }
	tests := []struct {
		name   string
		frames []proto.Message
		want   want // cached -1 means no prompt token details
	}{
		{
			name:   "later zero prompt keeps prompt",
			frames: []proto.Message{grpcChunkMsg(7, 0, 3), grpcChunkMsg(0, 1, 0)},
			want:   want{7, 1, 8, 0},
		},
		{
			name:   "later partial usage overwrites cached",
			frames: []proto.Message{grpcChunkMsg(7, 1, 3), grpcChunkMsg(0, 1, 0)},
			want:   want{7, 1, 8, 0},
		},
		{
			name:   "complete with zero completion keeps completion",
			frames: []proto.Message{grpcChunkMsg(7, 1, 3), grpcChunkMsg(7, 1, 3), grpcChunkMsg(7, 1, 3), grpcCompleteMsg(7, 0, 3, "stop")},
			want:   want{7, 1, 8, 3},
		},
		{
			name:   "cumulative usage takes the last value",
			frames: []proto.Message{grpcChunkMsg(7, 1, 3), grpcChunkMsg(7, 2, 3), grpcCompleteMsg(7, 3, 3, "stop")},
			want:   want{7, 3, 10, 3},
		},
		{
			name:   "later frame without usage keeps usage",
			frames: []proto.Message{grpcCompleteMsg(9, 3, 3, "stop"), grpcChunkMsg(0, 0, 0)},
			want:   want{9, 3, 12, 3},
		},
		{
			name:   "embed response",
			frames: []proto.Message{&pb.EmbedResponse{PromptTokens: 9}},
			want:   want{9, 0, 9, -1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var perFrame [][]byte
			for _, msg := range tt.frames {
				perFrame = append(perFrame, grpcTestFrame(t, msg))
			}
			whole := slices.Concat(perFrame...)
			check := func(t *testing.T, chunks ...[]byte) {
				t.Helper()
				got := feedVllmGRPC(t, chunks...).usage
				cached := -1
				if got.PromptTokenDetails != nil {
					cached = got.PromptTokenDetails.CachedTokens
				}
				assert.Equal(t, tt.want, want{got.PromptTokens, got.CompletionTokens, got.TotalTokens, cached})
			}
			t.Run("one frame per chunk", func(t *testing.T) { check(t, perFrame...) })
			t.Run("coalesced", func(t *testing.T) { check(t, whole) })
			t.Run("split at every offset", func(t *testing.T) {
				for i := 1; i < len(whole); i++ {
					check(t, whole[:i], whole[i:])
				}
			})
		})
	}
}

func TestGRPCFramesSplitAcrossChunks(t *testing.T) {
	noUsage := grpcTestFrame(t, grpcChunkMsg(0, 0, 0))
	done := grpcTestFrame(t, grpcCompleteMsg(7, 10, 0, "stop"))
	assertDone := func(t *testing.T, got fwkrh.Usage) {
		t.Helper()
		if got.PromptTokens != 7 || got.CompletionTokens != 10 {
			t.Fatalf("usage = %+v, want prompt 7, completion 10", got)
		}
	}

	t.Run("two frames split at every offset", func(t *testing.T) {
		whole := slices.Concat(noUsage, done)
		for i := 1; i < len(whole); i++ {
			assertDone(t, feedVllmGRPC(t, whole[:i], whole[i:]).usage)
		}
	})
	t.Run("three frames split at every pair of offsets", func(t *testing.T) {
		whole := slices.Concat(noUsage, noUsage, done)
		for i := 1; i < len(whole); i++ {
			for j := i + 1; j < len(whole); j += 3 {
				assertDone(t, feedVllmGRPC(t, whole[:i], whole[i:j], whole[j:]).usage)
			}
		}
	})
	t.Run("three frames delivered a byte at a time", func(t *testing.T) {
		whole := slices.Concat(noUsage, noUsage, done)
		chunks := make([][]byte, 0, len(whole))
		for i := range whole {
			chunks = append(chunks, whole[i:i+1])
		}
		assertDone(t, feedVllmGRPC(t, chunks...).usage)
	})
	t.Run("zero-length frame between frames", func(t *testing.T) {
		empty := []byte{0, 0, 0, 0, 0}
		run := feedVllmGRPC(t, slices.Concat(noUsage, empty, done))
		assertDone(t, run.usage)
		assert.Zero(t, run.oversizedLogs)
		run = feedVllmGRPC(t, slices.Concat(noUsage, empty[:2]), slices.Concat(empty[2:], done))
		assertDone(t, run.usage)
		assert.Zero(t, run.oversizedLogs)
	})
}

func TestGRPCPartialFrameAtEndOfStream(t *testing.T) {
	noUsage := grpcTestFrame(t, grpcChunkMsg(0, 0, 0))
	done := grpcTestFrame(t, grpcCompleteMsg(7, 10, 0, "stop"))
	tests := []struct {
		name           string
		chunks         [][]byte
		wantCompletion int
		wantCarried    int
	}{
		{name: "earlier usage stays", chunks: [][]byte{done, noUsage[:3]}, wantCompletion: 10, wantCarried: 3},
		{name: "partial header alone reports no usage", chunks: [][]byte{done[:3]}, wantCarried: 3},
		{name: "partial message alone reports no usage", chunks: [][]byte{done[:7]}, wantCarried: 7},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			run := feedVllmGRPC(t, tt.chunks...)
			assert.Equal(t, tt.wantCompletion, run.usage.CompletionTokens)
			assert.Equal(t, tt.wantCarried, run.carryBeforeEnd)
			assert.Zero(t, run.oversizedLogs)
			assert.Equal(t, 1, run.parseErrors, "the parser reports the partial frame")
		})
	}
}

func TestGRPCCarryCap(t *testing.T) {
	noUsage := grpcTestFrame(t, grpcChunkMsg(0, 0, 0))
	done := grpcTestFrame(t, grpcCompleteMsg(7, 10, 0, "stop"))
	atCap := grpcDoneOfSize(t, grpcTestCap)
	over := grpcDoneOfSize(t, grpcTestCap+1)
	hugeClaim := []byte{0, 0xff, 0xff, 0xff, 0xff, 1, 2, 3}

	tests := []struct {
		name           string
		chunks         [][]byte
		wantCompletion int
		wantLogs       int
	}{
		{
			name:           "message at the cap split across chunks parses",
			chunks:         [][]byte{atCap[:100], atCap[100:]},
			wantCompletion: 10,
		},
		{
			name:     "message one byte over the cap stops parsing",
			chunks:   [][]byte{over[:100], over[100:]},
			wantLogs: 1,
		},
		{
			name:     "whole message one byte over the cap in a non-final chunk stops parsing",
			chunks:   [][]byte{over, done},
			wantLogs: 1,
		},
		{
			name:     "oversized claim split inside the header stops parsing",
			chunks:   [][]byte{over[:3], over[3:100], over[100:]},
			wantLogs: 1,
		},
		{
			name:     "near 4 GiB claim ignores later chunks",
			chunks:   [][]byte{hugeClaim, done},
			wantLogs: 1,
		},
		{
			name:     "near 4 GiB claim split inside the header ignores later chunks",
			chunks:   [][]byte{hugeClaim[:3], hugeClaim[3:5], hugeClaim[5:], done},
			wantLogs: 1,
		},
		{
			name:           "usage before an oversized claim stays",
			chunks:         [][]byte{done, over[:100], noUsage, done},
			wantCompletion: 10,
			wantLogs:       1,
		},
		{
			name:           "usage sharing a chunk with an oversized claim stays",
			chunks:         [][]byte{slices.Concat(done, over[:100]), noUsage},
			wantCompletion: 10,
			wantLogs:       1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			run := feedGRPCResponse(t, vllmgrpc.NewVllmGRPCParser(), "application/grpc", tt.chunks)
			assert.Equal(t, tt.wantCompletion, run.usage.CompletionTokens)
			assert.Equal(t, tt.wantLogs, run.oversizedLogs)
			assert.Zero(t, run.carryBeforeEnd)
		})
	}

	// A buffered response arrives whole in the final chunk, so nothing is carried.
	t.Run("whole message over the cap in the final chunk parses", func(t *testing.T) {
		parser := vllmgrpc.NewVllmGRPCParser()
		server := NewStreamingServer(nil, &mockDirector{}, NewParserRegistry([]fwkrh.Parser{parser}, logr.Discard()), 0)
		reqCtx := &RequestContext{Request: &Request{}, Response: &Response{Headers: map[string]string{"content-type": "application/grpc"}}, Parser: parser}
		server.HandleResponseBody(context.Background(), reqCtx, grpcDoneOfSize(t, 5<<20), true)
		assert.Equal(t, 10, reqCtx.Usage.CompletionTokens)
	})
}

func TestGRPCCompressedFrames(t *testing.T) {
	done := grpcTestFrame(t, grpcCompleteMsg(7, 10, 0, "stop"))
	compressed := []byte{1, 0, 0, 0, 1, 9}
	tests := []struct {
		name            string
		chunks          [][]byte
		wantParseErrors int
	}{
		{name: "compressed frame sharing a chunk keeps earlier usage", chunks: [][]byte{slices.Concat(done, compressed)}, wantParseErrors: 1},
		{name: "chunks after a split compressed header reach the parser", chunks: [][]byte{compressed[:3], compressed[3:], done}, wantParseErrors: 1},
		{name: "chunks after a compressed message that continues reach the parser", chunks: [][]byte{{1, 0, 0, 0, 200, 9}, done}, wantParseErrors: 1},
		{
			name:            "compressed continuation reading as an oversized header does not stop parsing",
			chunks:          [][]byte{{1, 0, 0, 0, 12, 9}, {0, 0xff, 0xff, 0xff, 0xff, 1, 2, 3, 4, 5, 6}, done},
			wantParseErrors: 2,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			run := feedVllmGRPC(t, tt.chunks...)
			assert.Equal(t, 7, run.usage.PromptTokens)
			assert.Equal(t, 10, run.usage.CompletionTokens)
			assert.Zero(t, run.oversizedLogs)
			assert.Equal(t, tt.wantParseErrors, run.parseErrors)
			assert.Zero(t, run.carryBeforeEnd)
		})
	}
}

func TestGRPCCarryContentTypes(t *testing.T) {
	noUsage := grpcTestFrame(t, grpcChunkMsg(0, 0, 0))
	done := grpcTestFrame(t, grpcCompleteMsg(7, 10, 0, "stop"))
	whole := slices.Concat(noUsage, done)
	cut := len(noUsage) + 2
	tests := []struct {
		contentType    string
		wantCompletion int
	}{
		{contentType: "application/grpc", wantCompletion: 10},
		{contentType: "application/grpc; charset=utf-8", wantCompletion: 10},
		{contentType: "Application/GRPC+proto", wantCompletion: 10},
		{contentType: "application/grpc+proto", wantCompletion: 10},
		{contentType: "application/grpc-web"},
		{contentType: "application/grpc-web-text"},
		{contentType: "application/json"},
		{contentType: ""},
	}
	for _, tt := range tests {
		t.Run("content type "+tt.contentType, func(t *testing.T) {
			run := feedGRPCResponse(t, vllmgrpc.NewVllmGRPCParser(), tt.contentType, [][]byte{whole[:cut], whole[cut:]})
			assert.Equal(t, tt.wantCompletion, run.usage.CompletionTokens)
		})
	}
}

func TestGRPCCarryAppliesToVertexAIFrames(t *testing.T) {
	frame := grpcTestFrame(t, &httpbody.HttpBody{
		ContentType: "application/json",
		Data:        []byte(`{"usage":{"prompt_tokens":7,"completion_tokens":10,"total_tokens":17}}`),
	})
	for i := 1; i < len(frame); i++ {
		run := feedGRPCResponse(t, vertexai.NewVertexAIParser(), "application/grpc", [][]byte{frame[:i], frame[i:]})
		if run.usage.CompletionTokens != 10 {
			t.Fatalf("split at %d: usage = %+v, want completion 10", i, run.usage)
		}
	}
}

type usageRecordingDirector struct {
	mockDirector
	usage fwkrh.Usage
}

func (d *usageRecordingDirector) HandleResponseBody(_ context.Context, reqCtx *RequestContext, endOfStream bool) *RequestContext {
	if endOfStream {
		d.usage = reqCtx.Usage
	}
	return reqCtx
}

func grpcRespBody(chunk []byte, endOfStream bool) *extProcPb.ProcessingRequest {
	return &extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseBody{
		ResponseBody: &extProcPb.HttpBody{Body: chunk, EndOfStream: endOfStream},
	}}
}

func TestProcessRecordsGRPCUsageAndForwardsBytesUnchanged(t *testing.T) {
	noUsage := grpcTestFrame(t, grpcChunkMsg(0, 0, 0))
	done := grpcTestFrame(t, grpcCompleteMsg(7, 10, 0, "stop"))
	whole := slices.Concat(noUsage, done)
	cut := len(noUsage) + 2
	trailers := &extProcPb.ProcessingRequest{Request: &extProcPb.ProcessingRequest_ResponseTrailers{
		ResponseTrailers: &extProcPb.HttpTrailers{Trailers: &configPb.HeaderMap{}},
	}}
	withPartial := slices.Concat(done, noUsage[:3])

	tests := []struct {
		name     string
		messages []*extProcPb.ProcessingRequest
		upstream []byte
	}{
		{
			name:     "split inside the header, ends on trailers",
			messages: []*extProcPb.ProcessingRequest{grpcRespBody(whole[:cut], false), grpcRespBody(whole[cut:], false), trailers},
			upstream: whole,
		},
		{
			name:     "split inside the header, ends on end of stream",
			messages: []*extProcPb.ProcessingRequest{grpcRespBody(whole[:cut], false), grpcRespBody(whole[cut:], true)},
			upstream: whole,
		},
		{
			name:     "coalesced, ends on trailers",
			messages: []*extProcPb.ProcessingRequest{grpcRespBody(whole, false), trailers},
			upstream: whole,
		},
		{
			name:     "trailing partial frame, ends on trailers",
			messages: []*extProcPb.ProcessingRequest{grpcRespBody(withPartial, false), trailers},
			upstream: withPartial,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reqs := []*extProcPb.ProcessingRequest{
				newRequestHeaders(map[string]string{":path": "/vllm.grpc.engine.VllmEngine/Generate"}),
				{Request: &extProcPb.ProcessingRequest_RequestBody{RequestBody: &extProcPb.HttpBody{
					Body:        grpcTestFrame(t, &pb.GenerateRequest{Input: &pb.GenerateRequest_Text{Text: "hi"}, Stream: true}),
					EndOfStream: true,
				}}},
				{Request: &extProcPb.ProcessingRequest_ResponseHeaders{ResponseHeaders: &extProcPb.HttpHeaders{
					Headers: &configPb.HeaderMap{Headers: []*configPb.HeaderValue{{Key: "content-type", RawValue: []byte("application/grpc")}}},
				}}},
			}
			srv := &replayProcessServer{ctx: context.Background(), reqs: append(reqs, tt.messages...)}
			director := &usageRecordingDirector{mockDirector: mockDirector{stream: true}}
			registry := NewParserRegistry([]fwkrh.Parser{vllmgrpc.NewVllmGRPCParser()}, logr.Discard())

			require.NoError(t, NewStreamingServer(nil, director, registry, 0).Process(srv))

			assert.Equal(t, 10, director.usage.CompletionTokens)
			var forwarded []byte
			for _, sent := range srv.sentResponses {
				forwarded = append(forwarded, sent.GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse().GetBody()...)
			}
			assert.Equal(t, tt.upstream, forwarded)
		})
	}
}
