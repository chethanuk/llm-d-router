package programaware

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

func decoder(s string) *json.Decoder { return json.NewDecoder(strings.NewReader(s)) }

func TestFactory_DefaultConfig(t *testing.T) {
	p, err := ProgramAwarePluginFactory("test", nil, nil)
	require.NoError(t, err)
	require.NotNil(t, p)
}

func TestFactory_LASConfig(t *testing.T) {
	cfg := `{"strategy":"las","lasWeightService":0.7,"lasWeightHeadWait":0.3,"lasHalfLifeSeconds":60}`
	p, err := ProgramAwarePluginFactory("test", decoder(cfg), nil)
	require.NoError(t, err)
	require.NotNil(t, p)
}

func TestFactory_UnknownStrategy(t *testing.T) {
	_, err := ProgramAwarePluginFactory("test", decoder(`{"strategy":"wfq"}`), nil)
	require.Error(t, err)
}

func TestFactory_InvalidConfig(t *testing.T) {
	cases := map[string]string{
		"negative ttl":       `{"evictionTtlSeconds":-1}`,
		"zero sweep":         `{"evictionSweepSeconds":0}`,
		"negative weight":    `{"lasWeightService":-0.1}`,
		"decay factor > 1":   `{"lasDecayFactor":1.5}`,
		"decay factor 0":     `{"lasDecayFactor":0}`,
		"negative half life": `{"lasHalfLifeSeconds":-1}`,
	}
	for name, cfg := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := ProgramAwarePluginFactory("test", decoder(cfg), nil)
			require.Error(t, err)
		})
	}
}

func TestPick_NilBand(t *testing.T) {
	p := &ProgramAwarePlugin{}
	got, err := p.Pick(context.Background(), nil)
	require.NoError(t, err)
	assert.Nil(t, got)
}

func TestPick_AllQueuesEmpty(t *testing.T) {
	band := &fwkfcmocks.MockPriorityBandAccessor{
		PriorityV: 0,
		IterateQueuesFunc: func(cb func(flowcontrol.FlowQueueAccessor) bool) {
			cb(&fwkfcmocks.MockFlowQueueAccessor{LenV: 0, FlowKeyV: flowcontrol.FlowKey{ID: "p1"}})
			cb(&fwkfcmocks.MockFlowQueueAccessor{LenV: 0, FlowKeyV: flowcontrol.FlowKey{ID: "p2"}})
		},
	}
	p := &ProgramAwarePlugin{}
	got, err := p.Pick(context.Background(), band)
	require.NoError(t, err)
	assert.Nil(t, got)
}

func TestPick_SingleNonEmptyQueue_StashesEnqueueTime(t *testing.T) {
	enqueue := time.Now().Add(-100 * time.Millisecond)
	req := &fwksched.InferenceRequest{FairnessID: "alpha"}
	item := &fwkfcmocks.MockQueueItemAccessor{
		EnqueueTimeV:     enqueue,
		OriginalRequestV: &fwkfcmocks.MockFlowControlRequest{InferenceRequestV: req},
	}
	queue := &fwkfcmocks.MockFlowQueueAccessor{
		LenV:     1,
		FlowKeyV: flowcontrol.FlowKey{ID: "alpha"},
		PeekV:    item,
	}
	band := &fwkfcmocks.MockPriorityBandAccessor{
		IterateQueuesFunc: func(cb func(flowcontrol.FlowQueueAccessor) bool) { cb(queue) },
	}

	p := &ProgramAwarePlugin{}
	got, err := p.Pick(context.Background(), band)
	require.NoError(t, err)
	assert.Equal(t, queue, got)

	stashed, ok := fwksched.ReadRequestAttribute[time.Time](req, enqueueTimeAttributeKey)
	require.True(t, ok)
	assert.Equal(t, enqueue, stashed)
}

func TestPreRequest_RecordsDispatchAndWait(t *testing.T) {
	enqueue := time.Now().Add(-50 * time.Millisecond)
	req := &fwksched.InferenceRequest{FairnessID: "alpha"}
	req.PutAttribute(enqueueTimeAttributeKey, enqueue)

	p := &ProgramAwarePlugin{}
	p.PreRequest(context.Background(), req, nil)

	m := p.getOrCreateMetrics("alpha")
	assert.Equal(t, int64(1), m.DispatchedCount())
	assert.Equal(t, int64(1), m.InFlight())
	assert.Equal(t, int64(1), m.WaitCount())
	assert.Greater(t, m.AverageWaitTime(), 0.0)
}

func TestPreRequest_NoEnqueueAttribute_StillDispatches(t *testing.T) {
	req := &fwksched.InferenceRequest{FairnessID: "alpha"}
	p := &ProgramAwarePlugin{}
	p.PreRequest(context.Background(), req, nil)

	m := p.getOrCreateMetrics("alpha")
	assert.Equal(t, int64(1), m.DispatchedCount())
	assert.Equal(t, int64(1), m.InFlight())
	assert.Equal(t, int64(0), m.WaitCount())
}

func TestPreRequest_NoFairnessID_FallsBackToDefault(t *testing.T) {
	req := &fwksched.InferenceRequest{}
	p := &ProgramAwarePlugin{}
	p.PreRequest(context.Background(), req, nil)

	got, ok := p.programMetrics.Load(metadata.DefaultFairnessID)
	require.True(t, ok, "default fairness ID entry should be created")
	m, ok := got.(*ProgramMetrics)
	require.True(t, ok)
	assert.Equal(t, int64(1), m.DispatchedCount())
}

func TestResponseBody_FinalChunkOnly(t *testing.T) {
	req := &fwksched.InferenceRequest{FairnessID: "alpha"}
	p := &ProgramAwarePlugin{}
	m := p.getOrCreateMetrics("alpha")
	seedTime := m.LastCompletionTime()
	m.RecordDispatched(time.Time{})

	// Intermediate chunk: in-flight unchanged, completion time unchanged.
	p.ResponseBody(context.Background(), req, &fwkrc.Response{EndOfStream: false}, nil)
	assert.Equal(t, int64(1), m.InFlight())
	assert.Equal(t, seedTime, m.LastCompletionTime())

	// Final chunk: completion advanced, in-flight decremented.
	time.Sleep(time.Millisecond)
	p.ResponseBody(context.Background(), req, &fwkrc.Response{EndOfStream: true}, nil)
	assert.Equal(t, int64(0), m.InFlight())
	assert.True(t, m.LastCompletionTime().After(seedTime))
}

func TestResponseBody_NilSafe(t *testing.T) {
	p := &ProgramAwarePlugin{}
	p.ResponseBody(context.Background(), nil, &fwkrc.Response{EndOfStream: true}, nil)
	p.ResponseBody(context.Background(), &fwksched.InferenceRequest{}, nil, nil)
}

func TestEvictIdle_RemovesIdle(t *testing.T) {
	p := &ProgramAwarePlugin{}
	m := p.getOrCreateMetrics("alpha")
	m.RecordDispatched(time.Time{})
	m.RecordCompletion(time.Now().Add(-10 * time.Second))

	p.evictIdle(time.Second)

	_, exists := p.programMetrics.Load("alpha")
	assert.False(t, exists)
}

func TestEvictIdle_KeepsInFlight(t *testing.T) {
	p := &ProgramAwarePlugin{}
	m := p.getOrCreateMetrics("alpha")
	m.RecordDispatched(time.Time{}) // inFlight = 1
	// Force lastCompletionTime old; in-flight should still gate eviction.
	m.mu.Lock()
	m.lastCompletionTime = time.Now().Add(-10 * time.Second)
	m.mu.Unlock()

	p.evictIdle(time.Second)

	_, exists := p.programMetrics.Load("alpha")
	assert.True(t, exists)
}

func TestEvictIdle_KeepsRecent(t *testing.T) {
	p := &ProgramAwarePlugin{}
	m := p.getOrCreateMetrics("alpha")
	m.RecordDispatched(time.Time{})
	m.RecordCompletion(time.Now())

	p.evictIdle(time.Hour)

	_, exists := p.programMetrics.Load("alpha")
	assert.True(t, exists)
}

func TestEvictIdle_EvictsNeverCompletedAfterTTL(t *testing.T) {
	p := &ProgramAwarePlugin{}
	m := p.getOrCreateMetrics("alpha")
	// Force the seed time into the past so the TTL gate trips.
	m.mu.Lock()
	m.lastCompletionTime = time.Now().Add(-10 * time.Second)
	m.mu.Unlock()

	p.evictIdle(time.Second)

	_, exists := p.programMetrics.Load("alpha")
	assert.False(t, exists)
}

func TestComputeFairnessIndex_EqualWaits(t *testing.T) {
	p := &ProgramAwarePlugin{}
	for _, id := range []string{"a", "b", "c"} {
		m := p.getOrCreateMetrics(id)
		m.RecordDispatched(time.Now().Add(-100 * time.Millisecond))
	}
	got := p.computeFairnessIndex()
	assert.InDelta(t, 1.0, got, 0.05)
}

func TestComputeFairnessIndex_SingleProgram(t *testing.T) {
	p := &ProgramAwarePlugin{}
	m := p.getOrCreateMetrics("a")
	m.RecordDispatched(time.Now().Add(-50 * time.Millisecond))
	assert.Equal(t, 1.0, p.computeFairnessIndex())
}

func TestComputeFairnessIndex_NoData(t *testing.T) {
	p := &ProgramAwarePlugin{}
	assert.Equal(t, 1.0, p.computeFairnessIndex())
}

func TestComputeFairnessIndex_SkewedWaits(t *testing.T) {
	p := &ProgramAwarePlugin{}
	a := p.getOrCreateMetrics("a")
	b := p.getOrCreateMetrics("b")
	a.RecordDispatched(time.Now().Add(-10 * time.Millisecond))
	b.RecordDispatched(time.Now().Add(-1000 * time.Millisecond))
	got := p.computeFairnessIndex()
	assert.Less(t, got, 0.9, "skewed waits should produce sub-1.0 fairness")
}

func TestGetOrCreateMetrics_Idempotent(t *testing.T) {
	p := &ProgramAwarePlugin{}
	a := p.getOrCreateMetrics("alpha")
	b := p.getOrCreateMetrics("alpha")
	assert.Same(t, a, b)
}

func newDumpPlugin(t *testing.T) *ProgramAwarePlugin {
	t.Helper()
	s, err := newStrategy(DefaultConfig())
	require.NoError(t, err)
	return &ProgramAwarePlugin{strategy: s}
}

func TestProgramAwarePlugin_DumpState(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		p := &ProgramAwarePlugin{}
		payload, err := p.DumpState()
		require.NoError(t, err)
		require.True(t, json.Valid(payload))
		var state programAwareState
		require.NoError(t, json.Unmarshal(payload, &state))
		assert.Equal(t, "las", state.Strategy)
		assert.Equal(t, 1.0, state.JainsFairnessIndex)
		assert.Empty(t, state.Programs)
		assert.Equal(t, 0, state.TotalPrograms)
		assert.Equal(t, maxDebugDumpPrograms, state.MaxPrograms)
		assert.False(t, state.Truncated)
	})

	t.Run("single program completed", func(t *testing.T) {
		p := newDumpPlugin(t)
		req := &fwksched.InferenceRequest{FairnessID: "alpha"}
		req.PutAttribute(enqueueTimeAttributeKey, time.Now().Add(-50*time.Millisecond))
		p.PreRequest(context.Background(), req, nil)
		p.ResponseBody(context.Background(), req, &fwkrc.Response{
			EndOfStream: true,
			Usage:       fwkrh.Usage{PromptTokens: 10, CompletionTokens: 5},
		}, nil)

		var state programAwareState
		payload, err := p.DumpState()
		require.NoError(t, err)
		require.NoError(t, json.Unmarshal(payload, &state))
		require.Len(t, state.Programs, 1)
		ps := state.Programs[0]
		assert.Equal(t, "alpha", ps.ProgramID)
		assert.Equal(t, int64(1), ps.DispatchedCount)
		assert.Equal(t, int64(0), ps.InFlight)
		assert.Equal(t, int64(1), ps.WaitCount)
		assert.Equal(t, float64(1*10+2*5), ps.AttainedService)
		assert.NotEmpty(t, ps.LastCompletionTime)
	})

	t.Run("in-flight has zero service and empty completion time", func(t *testing.T) {
		// Use the production PreRequest path: getOrCreateMetrics seeds
		// lastCompletionTime to "now", so this verifies the dump does NOT report
		// that seed as a completion time for a request still in flight.
		p := newDumpPlugin(t)
		req := &fwksched.InferenceRequest{FairnessID: "beta"}
		p.PreRequest(context.Background(), req, nil) // dispatched, in-flight, no completion

		var state programAwareState
		payload, err := p.DumpState()
		require.NoError(t, err)
		require.NoError(t, json.Unmarshal(payload, &state))
		require.Len(t, state.Programs, 1)
		assert.Equal(t, int64(1), state.Programs[0].DispatchedCount)
		assert.Equal(t, int64(1), state.Programs[0].InFlight)
		assert.Equal(t, float64(0), state.Programs[0].AttainedService)
		assert.Empty(t, state.Programs[0].LastCompletionTime)
	})

	t.Run("multi-program ordering busiest first", func(t *testing.T) {
		p := newDumpPlugin(t)
		a := &ProgramMetrics{}
		a.RecordDispatched(time.Time{})
		a.RecordDispatched(time.Time{}) // inFlight 2
		p.programMetrics.Store("prog-a", a)
		b := &ProgramMetrics{}
		b.RecordDispatched(time.Time{}) // inFlight 1
		p.programMetrics.Store("prog-b", b)

		var state programAwareState
		payload, err := p.DumpState()
		require.NoError(t, err)
		require.NoError(t, json.Unmarshal(payload, &state))
		require.Equal(t, 2, state.TotalPrograms)
		assert.Equal(t, "prog-a", state.Programs[0].ProgramID)
		assert.Equal(t, "prog-b", state.Programs[1].ProgramID)
	})

	t.Run("truncation", func(t *testing.T) {
		p := newDumpPlugin(t)
		for i := 0; i < maxDebugDumpPrograms+5; i++ {
			m := p.getOrCreateMetrics(fmt.Sprintf("prog-%03d", i))
			// More in-flight for higher indices so the head is deterministic.
			for j := 0; j <= i; j++ {
				m.RecordDispatched(time.Time{})
			}
		}
		var state programAwareState
		payload, err := p.DumpState()
		require.NoError(t, err)
		require.NoError(t, json.Unmarshal(payload, &state))
		assert.True(t, state.Truncated)
		assert.Len(t, state.Programs, maxDebugDumpPrograms)
		assert.Equal(t, maxDebugDumpPrograms+5, state.TotalPrograms)
		assert.Equal(t, maxDebugDumpPrograms, state.MaxPrograms)
	})

	t.Run("valid json no unexpected keys", func(t *testing.T) {
		p := newDumpPlugin(t)
		m := p.getOrCreateMetrics("alpha")
		m.RecordDispatched(time.Time{})
		payload, err := p.DumpState()
		require.NoError(t, err)
		require.True(t, json.Valid(payload))

		var top map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(payload, &top))
		for k := range top {
			require.Truef(t, map[string]bool{
				"strategy": true, "jainsFairnessIndex": true, "programs": true,
				"totalPrograms": true, "maxPrograms": true, "truncated": true,
			}[k], "unexpected top-level key %q", k)
		}
		var progs struct {
			Programs []map[string]json.RawMessage `json:"programs"`
		}
		require.NoError(t, json.Unmarshal(payload, &progs))
		for _, pr := range progs.Programs {
			for k := range pr {
				require.Truef(t, map[string]bool{
					"programID": true, "dispatchedCount": true, "inFlight": true,
					"averageWaitTimeMs": true, "waitCount": true, "attainedService": true,
					"lastCompletionTime": true,
				}[k], "unexpected program key %q", k)
			}
		}
	})
}
