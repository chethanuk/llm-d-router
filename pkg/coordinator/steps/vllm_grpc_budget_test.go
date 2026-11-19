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

package steps

import (
	"context"
	"errors"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/testing/protocmp"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpctest"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// stepBudget returns the force-stream budget EnableForceStream gave step.
func stepBudget(t *testing.T, step pipeline.Step) *forceStreamBudget {
	t.Helper()
	switch s := step.(type) {
	case *DecodeStep:
		return s.budget
	case *ConditionalDecodeStep:
		return s.budget
	}
	t.Fatalf("step %T cannot force-stream", step)
	return nil
}

func TestUnaryTokenClientMergeIsBoundedByForceStreamBudget(t *testing.T) {
	const (
		bufferSize = "1MiB"
		// perRequest bounds one request with no declared max_new_tokens to an
		// 8KiB reservation, so bigFrames below overrun it.
		perRequest = "8KiB"
	)
	oneFrame := []*vllmpb.GenerateResponse{grpctest.DefaultVLLMResponse()}
	threeFrames := []*vllmpb.GenerateResponse{
		{
			PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 5},
			Outputs:    &vllmpb.SequenceOutput{Text: "Hello", NumTokens: 2, TokenIds: []uint32{11, 12}},
		},
		{Outputs: &vllmpb.SequenceOutput{Text: " ", NumTokens: 1, TokenIds: []uint32{13}}},
		{Outputs: &vllmpb.SequenceOutput{
			Text: "world", NumTokens: 1, TokenIds: []uint32{14},
			FinishInfo: &vllmpb.FinishInfo{FinishReason: vllmpb.FinishInfo_STOP, NumOutputTokens: 4},
		}},
	}
	mergedThree := []*vllmpb.GenerateResponse{{
		PromptInfo: &vllmpb.PromptInfo{NumPromptTokens: 5},
		Outputs: &vllmpb.SequenceOutput{
			Text: "Hello world", NumTokens: 4, TokenIds: []uint32{11, 12, 13, 14},
			FinishInfo: &vllmpb.FinishInfo{FinishReason: vllmpb.FinishInfo_STOP, NumOutputTokens: 4},
		},
	}}
	// Two frames, each far under the 64 MiB message limit, whose sum exceeds
	// the 8KiB reservation.
	bigText := strings.Repeat("a", 6<<10)
	bigFrames := []*vllmpb.GenerateResponse{
		{Outputs: &vllmpb.SequenceOutput{Text: bigText, NumTokens: 1, TokenIds: []uint32{11}}},
		{Outputs: &vllmpb.SequenceOutput{Text: bigText, NumTokens: 1, TokenIds: []uint32{12}}},
	}

	textReq := &vllmpb.GenerateRequest{Prompt: &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 2, 3, 4, 5}}}}
	// hugeReq declares max_new_tokens x n far past the per-request cap.
	hugeReq := &vllmpb.GenerateRequest{
		Prompt:   &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 2, 3, 4, 5}}},
		Sampling: &vllmpb.RandomSampling{NumSequences: 4},
		Stopping: &vllmpb.StoppingCriteria{MaxNewTokens: 1 << 20},
	}

	// logprobsReq asks for per-token data the byte estimate does not size.
	logprobsReq := &vllmpb.GenerateRequest{
		Prompt:   &vllmpb.GenerateRequest_TokenIds{TokenIds: &vllmpb.TokenIds{Ids: []uint32{1, 2, 3, 4, 5}}},
		Response: &vllmpb.ResponseOptions{OutputLogprobs: true},
	}

	type row struct {
		name   string
		step   string // decode, conditional-decode
		stream bool
		// noBudget leaves pipeline force_stream off.
		noBudget bool
		// fillBudget reserves the whole shared budget before the request.
		fillBudget bool
		req        *vllmpb.GenerateRequest
		frames     []*vllmpb.GenerateResponse
		respErr    error

		wantPath string
		// wantReservedDuringCall means the request holds a reservation of its
		// own while the worker answers.
		wantReservedDuringCall bool
		wantClient             []*vllmpb.GenerateResponse
		wantErr                bool
		wantDeferred           bool
		// wantResult is the force_stream_total result recorded for the step;
		// empty means none.
		wantResult string
	}

	tests := make([]row, 0, 16)
	for _, step := range []string{DecodeStepName, ConditionalDecodeStepName} {
		tests = append(tests,
			row{
				name: step + " unary client with force_stream off calls unary Generate", step: step,
				noBudget: true, req: textReq, frames: oneFrame,
				wantPath: vllmpb.Inference_Generate_FullMethodName, wantClient: oneFrame,
			},
			row{
				name: step + " unary client within the budget gets one merged frame from GenerateStream", step: step,
				req: textReq, frames: threeFrames,
				wantPath: vllmpb.Inference_GenerateStream_FullMethodName, wantReservedDuringCall: true,
				wantClient: mergedThree, wantResult: coordmetrics.ForceStreamResultForced,
			},
			row{
				name: step + " unary client calls unary Generate when the budget is full", step: step,
				fillBudget: true, req: textReq, frames: oneFrame,
				wantPath: vllmpb.Inference_Generate_FullMethodName, wantClient: oneFrame,
				wantResult: coordmetrics.ForceStreamResultFallbackBudget,
			},
			row{
				name: step + " unary client calls unary Generate when max_new_tokens x n exceeds the per-request cap", step: step,
				req: hugeReq, frames: oneFrame,
				wantPath: vllmpb.Inference_Generate_FullMethodName, wantClient: oneFrame,
				wantResult: coordmetrics.ForceStreamResultFallbackUnbounded,
			},
			row{
				name: step + " unary client fails when merged frames exceed the reservation", step: step,
				req: textReq, frames: bigFrames,
				wantPath: vllmpb.Inference_GenerateStream_FullMethodName, wantReservedDuringCall: true,
				wantErr: true, wantResult: coordmetrics.ForceStreamResultErrorCeiling,
			},
			row{
				name: step + " unary client asking for logprobs calls unary Generate", step: step,
				req: logprobsReq, frames: oneFrame,
				wantPath: vllmpb.Inference_Generate_FullMethodName, wantClient: oneFrame,
				wantResult: coordmetrics.ForceStreamResultFallbackUnsupported,
			},
			row{
				name: step + " streaming client gets worker frames unchanged with no reservation", step: step,
				stream: true, req: textReq, frames: threeFrames,
				wantPath: vllmpb.Inference_GenerateStream_FullMethodName, wantClient: threeFrames,
			},
		)
	}
	tests = append(tests,
		row{
			name: "conditional-decode unary client defers on cache miss and releases its reservation", step: ConditionalDecodeStepName,
			req: textReq, respErr: status.Error(codes.FailedPrecondition, "kv cache miss"),
			wantPath: vllmpb.Inference_GenerateStream_FullMethodName, wantReservedDuringCall: true,
			wantDeferred: true,
		},
		row{
			name: "conditional-decode unary client defers on cache miss after frames and releases its reservation", step: ConditionalDecodeStepName,
			req: textReq, frames: oneFrame, respErr: status.Error(codes.FailedPrecondition, "kv cache miss"),
			wantPath: vllmpb.Inference_GenerateStream_FullMethodName, wantReservedDuringCall: true,
			wantDeferred: true,
		},
	)

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			reg := prometheus.NewRegistry()
			require.NoError(t, coordmetrics.Register(reg))
			coordmetrics.Reset()

			var (
				budget     *forceStreamBudget
				preserved  int64
				duringCall atomic.Int64
			)
			up := grpctest.NewVLLMUpstream(t, grpctest.VLLMUpstreamOptions{
				Respond: func(grpctest.VLLMCall) ([]*vllmpb.GenerateResponse, error) {
					if budget != nil {
						duringCall.Store(budget.current.Load() - preserved)
					}
					return tc.frames, tc.respErr
				},
			})
			gwClient := gateway.New(config.GatewayConfig{Address: up.URL()})

			var (
				step pipeline.Step
				err  error
			)
			if tc.step == DecodeStepName {
				step, err = NewDecodeStep(gwClient, map[string]any{ParamKVConnector: kv.SharedStorage})
			} else {
				step, err = NewConditionalDecodeStep(gwClient, nil)
			}
			require.NoError(t, err)
			if !tc.noBudget {
				require.NoError(t, EnableForceStream([]pipeline.Step{step}, bufferSize, perRequest))
				budget = stepBudget(t, step)
				if tc.fillBudget {
					require.True(t, budget.tryReserve(DecodeStepName, budget.max), "hold the whole shared budget")
					preserved = budget.max
					t.Cleanup(func() { budget.release(DecodeStepName, preserved) })
				}
			}

			rec := httptest.NewRecorder()
			reqCtx := &pipeline.RequestContext{
				RequestID:        "req-budget",
				Model:            "llama",
				Stream:           tc.stream,
				KVTransferParams: map[string]any{},
				ResponseWriter:   rec,
				GRPC:             &pipeline.VLLMGRPCRequest{Req: scanTestVLLMRequest(t, tc.req)},
			}

			execErr := step.Execute(ctx, reqCtx)
			switch {
			case tc.wantErr:
				require.Error(t, execErr)
				require.NotErrorIs(t, execErr, pipeline.ErrPipelineDone)
				require.True(t, errors.Is(execErr, grpcframe.ErrTooLarge) || errors.Is(execErr, errForceStreamCeiling),
					"overrun must fail with a resource error, got %v", execErr)
				gotStatus := rec.Header().Get(grpcframe.HeaderStatus)
				require.NotEmpty(t, gotStatus, "client must get a grpc-status")
				require.NotEqual(t, strconv.Itoa(int(codes.OK)), gotStatus, "client grpc-status")
			case tc.wantDeferred:
				require.NoError(t, execErr)
			case tc.step == ConditionalDecodeStepName:
				require.ErrorIs(t, execErr, pipeline.ErrPipelineDone)
			default:
				require.NoError(t, execErr)
			}

			calls := up.Calls()
			require.Len(t, calls, 1)
			require.Equal(t, tc.wantPath, calls[0].Path, "worker RPC method")

			switch {
			case tc.wantClient != nil:
				raw := grpctest.ReadFrames(t, rec.Body)
				got := make([]*vllmpb.GenerateResponse, len(raw))
				for i, f := range raw {
					got[i] = &vllmpb.GenerateResponse{}
					grpctest.Unmarshal(t, f, got[i])
				}
				require.Empty(t, cmp.Diff(tc.wantClient, got, protocmp.Transform()), "client frames (-want +got)")
			case tc.wantDeferred:
				require.Zero(t, rec.Body.Len(), "step must not write to the client")
			default:
				require.Empty(t, grpctest.ReadFrames(t, rec.Body), "client must get no message")
			}

			if budget != nil {
				if tc.wantReservedDuringCall {
					require.Positive(t, duringCall.Load(), "request must hold a reservation while the worker answers")
				} else {
					require.Zero(t, duringCall.Load(), "request must not reserve")
				}
				require.Equal(t, preserved, budget.current.Load(), "request must release its whole reservation")
			}

			stepLabel := map[string]string{"step": tc.step}
			if tc.wantResult == "" {
				require.Zero(t, counterValue(t, reg, forceStreamTotalMetric, stepLabel), "no force_stream result recorded")
			} else {
				require.InDelta(t, 1.0, counterValue(t, reg, forceStreamTotalMetric,
					map[string]string{"step": tc.step, "result": tc.wantResult}), 1e-9, "force_stream result")
				require.InDelta(t, 1.0, counterValue(t, reg, forceStreamTotalMetric, stepLabel), 1e-9, "one force_stream result")
			}
		})
	}
}
