package nohitlru_test

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/nohitlru"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/prefix"
	"github.com/llm-d/llm-d-router/test/utils"
)

var _ plugin.Handle = &fakeHandle{}

type fakeHandle struct {
	ctx     context.Context
	plugins map[string]plugin.Plugin
}

func newFakeHandle(ctx context.Context) *fakeHandle {
	return &fakeHandle{ctx: ctx, plugins: map[string]plugin.Plugin{}}
}

func (h *fakeHandle) Context() context.Context {
	return h.ctx
}

func (h *fakeHandle) Plugin(name string) plugin.Plugin {
	return h.plugins[name]
}

func (h *fakeHandle) AddPlugin(name string, plugin plugin.Plugin) {
	h.plugins[name] = plugin
}

func (h *fakeHandle) GetAllPlugins() []plugin.Plugin {
	result := make([]plugin.Plugin, 0, len(h.plugins))
	for _, plugin := range h.plugins {
		result = append(result, plugin)
	}
	return result
}

func (h *fakeHandle) GetAllPluginsWithNames() map[string]plugin.Plugin {
	return h.plugins
}

func (h *fakeHandle) PodList() []k8stypes.NamespacedName {
	return make([]k8stypes.NamespacedName, 0)
}

func (h *fakeHandle) Metrics() plugin.MetricsRecorder {
	return nil
}

type stubPlugin struct {
	name plugin.TypedName
}

func (p *stubPlugin) TypedName() plugin.TypedName {
	return p.name
}

// newCold returns an endpoint without any prefix-cache match info (cold).
func newCold(name string) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: name}},
		&fwkdl.Metrics{},
		nil,
	)
}

// newColdNS returns a cold endpoint in the "default" namespace.
func newColdNS(name string) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: name, Namespace: "default"}},
		&fwkdl.Metrics{},
		nil,
	)
}

// newWarm returns an endpoint with prefix-cache match info indicating a cache hit.
func newWarm(name string) scheduling.Endpoint {
	ep := newCold(name)
	ep.Put(attrprefix.PrefixCacheMatchInfoDataKey.String(), attrprefix.NewPrefixCacheMatchInfo(5, 10, 1))
	return ep
}

func TestNoHitLRUFactoryDependencyValidation(t *testing.T) {
	tests := []struct {
		name         string
		handle       *fakeHandle
		params       map[string]any
		expectError  bool
		errorMessage string
	}{
		{
			name:        "missing prefix cache plugin - should work as optimization",
			handle:      newFakeHandle(utils.NewTestContext(t)),
			expectError: false,
		},
		{
			name: "prefix plugin present - should work",
			handle: func() *fakeHandle {
				h := newFakeHandle(utils.NewTestContext(t))
				h.AddPlugin(prefix.PrefixCacheScorerPluginType, &stubPlugin{name: plugin.TypedName{Type: prefix.PrefixCacheScorerPluginType, Name: prefix.PrefixCacheScorerPluginType}})
				return h
			}(),
			expectError: false,
		},
	}

	for _, tt := range tests {
		// Marshal params if provided
		var raw json.RawMessage
		if tt.params != nil {
			bytes, err := json.Marshal(tt.params)
			if err != nil {
				t.Fatalf("failed to marshal parameters: %v", err)
			}
			raw = bytes
		}

		plugin, err := nohitlru.Factory("test", plugin.StrictDecoder(raw), tt.handle)
		if tt.expectError {
			if err == nil {
				t.Fatalf("expected error for case %q, got none", tt.name)
			}
			if tt.errorMessage != "" && !strings.Contains(err.Error(), tt.errorMessage) {
				t.Fatalf("error message mismatch for case %q: %v", tt.name, err)
			}
			continue
		}

		if err != nil {
			t.Fatalf("unexpected error for case %q: %v", tt.name, err)
		}
		if plugin == nil {
			t.Fatalf("expected plugin instance for case %q", tt.name)
		}
	}
}

func TestNoHitLRUScorer(t *testing.T) {
	// Each test case creates its own endpoints to avoid cross-test attribute pollution.
	tests := []struct {
		name        string
		scorer      scheduling.Scorer
		req         *scheduling.InferenceRequest
		input       func() []scheduling.Endpoint
		wantScores  func([]scheduling.Endpoint) map[scheduling.Endpoint]float64
		description string
	}{
		{
			name:   "cold request - all endpoints never used",
			scorer: nohitlru.NewNoHitLRU(utils.NewTestContext(t), nohitlru.NoHitLRUType, nil),
			req: &scheduling.InferenceRequest{
				TargetModel: "test-model",
			},
			input: func() []scheduling.Endpoint {
				return []scheduling.Endpoint{newCold("pod-a"), newCold("pod-b"), newCold("pod-c")}
			},
			wantScores: func(eps []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
				return map[scheduling.Endpoint]float64{
					eps[0]: 1.0,
					eps[1]: 0.5,
					eps[2]: 0.0,
				}
			},
			description: "Never-used endpoints should get high scores for cold requests",
		},
		{
			name:   "cache hit - neutral scores",
			scorer: nohitlru.NewNoHitLRU(utils.NewTestContext(t), nohitlru.NoHitLRUType, nil),
			req: &scheduling.InferenceRequest{
				TargetModel: "test-model",
			},
			input: func() []scheduling.Endpoint {
				// At least one warm endpoint signals a cache hit.
				return []scheduling.Endpoint{newWarm("pod-a"), newCold("pod-b"), newCold("pod-c")}
			},
			wantScores: func(eps []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
				return map[scheduling.Endpoint]float64{
					eps[0]: 0.5,
					eps[1]: 0.5,
					eps[2]: 0.5,
				}
			},
			description: "Cache hits should return neutral scores",
		},
		{
			name:   "single endpoint - max score",
			scorer: nohitlru.NewNoHitLRU(utils.NewTestContext(t), nohitlru.NoHitLRUType, nil),
			req: &scheduling.InferenceRequest{
				TargetModel: "test-model",
			},
			input: func() []scheduling.Endpoint {
				return []scheduling.Endpoint{newCold("pod-a")}
			},
			wantScores: func(eps []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
				return map[scheduling.Endpoint]float64{eps[0]: 1.0}
			},
			description: "Single endpoint should get maximum score",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			eps := test.input()
			want := test.wantScores(eps)
			got := test.scorer.Score(utils.NewTestContext(t), test.req, eps)
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("%s: Unexpected output (-want +got): %v", test.description, diff)
			}
		})
	}
}

func TestNoHitLRUBasicFunctionality(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := nohitlru.NewNoHitLRU(ctx, nohitlru.NoHitLRUType, nil)

	endpointA := newCold("pod-a")
	endpointB := newCold("pod-b")
	endpoints := []scheduling.Endpoint{endpointA, endpointB}

	// Cold request (no attributes): should not crash and should return valid scores.
	scores := scorer.Score(ctx, &scheduling.InferenceRequest{}, endpoints)

	if len(scores) != 2 {
		t.Errorf("Expected 2 scores, got %d", len(scores))
	}

	for endpoint, score := range scores {
		if score < 0 || score > 1 {
			t.Errorf("Invalid score %f for endpoint %s", score, endpoint.GetMetadata().NamespacedName.String())
		}
	}

	if scores[endpointA] == scores[endpointB] {
		t.Errorf("Expected different scores for different endpoints, both got %f", scores[endpointA])
	}
}

func TestNoPrefixCacheStateFound(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := nohitlru.NewNoHitLRU(ctx, nohitlru.NoHitLRUType, nil)

	// No attributes on the endpoint → treated as cold request.
	endpointA := newCold("pod-a")
	endpoints := []scheduling.Endpoint{endpointA}

	scores := scorer.Score(ctx, &scheduling.InferenceRequest{}, endpoints)

	if scores[endpointA] != 1.0 {
		t.Errorf("No prefix cache attributes should result in cold request scoring (expected 1.0, got %f).", scores[endpointA])
	}
}

func TestNoHitLRUPreferLeastRecentlyUsedAfterColdRequests(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := nohitlru.NewNoHitLRU(ctx, nohitlru.NoHitLRUType, nil)

	// Shared cold endpoints — no PrefixCacheMatchInfo attributes.
	endpointA := newColdNS("pod-a")
	endpointB := newColdNS("pod-b")
	endpointC := newColdNS("pod-c")
	endpoints := []scheduling.Endpoint{endpointA, endpointB, endpointC}

	primaryProfile := "primary-profile"

	// warmEndpoints returns a fresh slice with at least one endpoint carrying cache-hit attributes.
	// The NamespacedName matches the cold endpoints, so LRU tracking is shared.
	warmEndpoints := func() []scheduling.Endpoint {
		w := scheduling.NewEndpoint(
			&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod-a", Namespace: "default"}},
			&fwkdl.Metrics{},
			nil,
		)
		w.Put(attrprefix.PrefixCacheMatchInfoDataKey.String(), attrprefix.NewPrefixCacheMatchInfo(5, 10, 1))
		return []scheduling.Endpoint{w, endpointB, endpointC}
	}

	requestToEndpoint := func(target scheduling.Endpoint) *scheduling.SchedulingResult {
		return &scheduling.SchedulingResult{
			PrimaryProfileName: primaryProfile,
			ProfileResults: map[string]*scheduling.ProfileRunResult{
				primaryProfile: {
					TargetEndpoints: []scheduling.Endpoint{target},
				},
			},
		}
	}

	assertHighestScoredPod := func(expectedEndpoint scheduling.Endpoint, testName string) {
		t.Helper()
		coldReq := &scheduling.InferenceRequest{RequestID: testName + "-scoring-check"}
		scores := scorer.Score(ctx, coldReq, endpoints)

		highestScore := -1.0
		var highestEndpoint scheduling.Endpoint
		for endpoint, score := range scores {
			if score > highestScore {
				highestScore = score
				highestEndpoint = endpoint
			}
		}

		if highestEndpoint.GetMetadata().NamespacedName.String() != expectedEndpoint.GetMetadata().NamespacedName.String() {
			t.Fatalf("expected %s to have highest score for LRU behavior, but %s had highest score (%f). All scores: %+v",
				expectedEndpoint.GetMetadata().NamespacedName.String(),
				highestEndpoint.GetMetadata().NamespacedName.String(),
				highestScore,
				scores)
		}
	}

	t.Run("initial cold request seeds cache", func(_ *testing.T) {
		coldReqA := &scheduling.InferenceRequest{RequestID: "cold-1"}
		scorer.Score(ctx, coldReqA, endpoints)
		scorer.PreRequest(ctx, coldReqA, requestToEndpoint(endpointA))
		assertHighestScoredPod(endpointB, "after-endpointA-used")
	})

	t.Run("unused endpoints rank above existing ones", func(t *testing.T) {
		coldReqCheck := &scheduling.InferenceRequest{RequestID: "cold-check"}
		coldScores := scorer.Score(ctx, coldReqCheck, endpoints)
		if coldScores[endpointB] <= coldScores[endpointA] {
			t.Fatalf("expected endpoint-b to outrank endpoint-a after endpoint-a handled previous cold request, scores=%+v", coldScores)
		}
		if coldScores[endpointB] != 1.0 {
			t.Fatalf("expected endpoint-b to score 1.0, scores=%+v", coldScores)
		}
		if coldScores[endpointC] != 0.5 {
			t.Fatalf("expected endpoint-c to score 0.5, scores=%+v", coldScores)
		}
	})

	t.Run("warm request leaves LRU untouched", func(t *testing.T) {
		warmReq := &scheduling.InferenceRequest{RequestID: "warm-1"}
		warmScores := scorer.Score(ctx, warmReq, warmEndpoints())
		for _, score := range warmScores {
			if score != 0.5 {
				t.Fatalf("expected neutral score for warm request, got %f", score)
			}
		}
		scorer.PreRequest(ctx, warmReq, requestToEndpoint(endpointB))
		postWarmReq := &scheduling.InferenceRequest{RequestID: "cold-after-warm"}
		postWarmScores := scorer.Score(ctx, postWarmReq, endpoints)
		if postWarmScores[endpointB] <= postWarmScores[endpointA] {
			t.Fatalf("expected warm request to leave ordering unchanged, scores=%+v", postWarmScores)
		}
	})

	t.Run("second cold request rotates to endpointB", func(_ *testing.T) {
		coldReqB := &scheduling.InferenceRequest{RequestID: "cold-2"}
		scorer.Score(ctx, coldReqB, endpoints)
		scorer.PreRequest(ctx, coldReqB, requestToEndpoint(endpointB))
		assertHighestScoredPod(endpointC, "after-endpointB-used")
	})

	t.Run("third cold request rotates back to endpointA", func(_ *testing.T) {
		coldReqC := &scheduling.InferenceRequest{RequestID: "cold-3"}
		scorer.Score(ctx, coldReqC, endpoints)
		scorer.PreRequest(ctx, coldReqC, requestToEndpoint(endpointC))
		assertHighestScoredPod(endpointA, "after-endpointC-used")
	})
}

func TestNoHitLRU_DumpState(t *testing.T) {
	ctx := utils.NewTestContext(t)

	// resultTo builds a SchedulingResult routing a cold request to target.
	resultTo := func(target scheduling.Endpoint) *scheduling.SchedulingResult {
		const profile = "primary"
		return &scheduling.SchedulingResult{
			PrimaryProfileName: profile,
			ProfileResults: map[string]*scheduling.ProfileRunResult{
				profile: {TargetEndpoints: []scheduling.Endpoint{target}},
			},
		}
	}

	// seedCold routes one cold request to each endpoint in order, advancing the LRU.
	seedCold := func(s *nohitlru.NoHitLRU, eps []scheduling.Endpoint) {
		for i, ep := range eps {
			req := &scheduling.InferenceRequest{RequestID: fmt.Sprintf("seed-%d", i)}
			s.Score(ctx, req, []scheduling.Endpoint{ep})
			s.PreRequest(ctx, req, resultTo(ep))
		}
	}

	type want struct {
		endpoints    []string
		totalEntries int
		maxEntries   int
		truncated    bool
	}

	tests := []struct {
		name  string
		build func() *nohitlru.NoHitLRU
		want  want
	}{
		{
			name:  "empty",
			build: func() *nohitlru.NoHitLRU { return nohitlru.NewNoHitLRU(ctx, nohitlru.NoHitLRUType, nil) },
			want:  want{endpoints: []string{}, totalEntries: 0, maxEntries: 100, truncated: false},
		},
		{
			name: "recency_order_oldest_first",
			build: func() *nohitlru.NoHitLRU {
				s := nohitlru.NewNoHitLRU(ctx, nohitlru.NoHitLRUType, nil)
				seedCold(s, []scheduling.Endpoint{newColdNS("pod-a"), newColdNS("pod-b"), newColdNS("pod-c")})
				return s
			},
			want: want{
				endpoints:    []string{"default/pod-a", "default/pod-b", "default/pod-c"},
				totalEntries: 3, maxEntries: 100, truncated: false,
			},
		},
		{
			name: "exact_cap_not_truncated",
			build: func() *nohitlru.NoHitLRU {
				s := nohitlru.NewNoHitLRU(ctx, nohitlru.NoHitLRUType, nil)
				eps := make([]scheduling.Endpoint, 0, 100)
				for i := 0; i < 100; i++ {
					eps = append(eps, newColdNS(fmt.Sprintf("pod-%03d", i)))
				}
				seedCold(s, eps)
				return s
			},
			// exactly at the cap: full set emitted, not truncated.
			want: want{totalEntries: 100, maxEntries: 100, truncated: false},
		},
		{
			name: "capped_and_truncated",
			build: func() *nohitlru.NoHitLRU {
				s := nohitlru.NewNoHitLRU(ctx, nohitlru.NoHitLRUType, nil)
				eps := make([]scheduling.Endpoint, 0, 105)
				for i := 0; i < 105; i++ {
					eps = append(eps, newColdNS(fmt.Sprintf("pod-%03d", i)))
				}
				seedCold(s, eps)
				return s
			},
			// expect 100 entries, the oldest-first head (pod-000..pod-099), truncated.
			want: want{totalEntries: 105, maxEntries: 100, truncated: true},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			payload, err := tt.build().DumpState()
			if err != nil {
				t.Fatalf("DumpState error: %v", err)
			}
			if !json.Valid(payload) {
				t.Fatalf("DumpState returned invalid JSON: %s", payload)
			}
			var got struct {
				Endpoints    []string `json:"endpoints"`
				TotalEntries int      `json:"totalEntries"`
				MaxEntries   int      `json:"maxEntries"`
				Truncated    bool     `json:"truncated"`
			}
			if err := json.Unmarshal(payload, &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			assertDumpStateKeys(t, payload)
			if got.TotalEntries != tt.want.totalEntries || got.MaxEntries != tt.want.maxEntries || got.Truncated != tt.want.truncated {
				t.Fatalf("counts mismatch: got %+v want %+v", got, tt.want)
			}
			if len(got.Endpoints) > got.MaxEntries {
				t.Fatalf("endpoints exceed cap: %d", len(got.Endpoints))
			}
			if tt.want.endpoints != nil {
				if diff := cmp.Diff(tt.want.endpoints, got.Endpoints); diff != "" {
					t.Fatalf("endpoints mismatch (-want +got): %s", diff)
				}
			}
			if tt.name == "capped_and_truncated" {
				if len(got.Endpoints) != 100 || got.Endpoints[0] != "default/pod-000" {
					t.Fatalf("expected oldest-first head of 100, got len=%d first=%q", len(got.Endpoints), got.Endpoints[0])
				}
			}
		})
	}
}

// assertDumpStateKeys asserts the DumpState payload exposes exactly the four
// sanitized fields and nothing else, so a future field that leaked request data
// would fail the test rather than be silently dropped on unmarshal.
func assertDumpStateKeys(t *testing.T, payload []byte) {
	t.Helper()
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(payload, &raw); err != nil {
		t.Fatalf("unmarshal raw keys: %v", err)
	}
	wantKeys := map[string]bool{"endpoints": true, "totalEntries": true, "maxEntries": true, "truncated": true}
	for k := range raw {
		if !wantKeys[k] {
			t.Fatalf("unexpected key %q in DumpState payload: %s", k, payload)
		}
	}
	for k := range wantKeys {
		if _, ok := raw[k]; !ok {
			t.Fatalf("missing key %q in DumpState payload: %s", k, payload)
		}
	}
}

// TestNoHitLRU_DumpState_NoRequestDataLeak seeds the LRU through a cold request
// carrying canary request data (ID, model, headers, body) and asserts none of it
// surfaces in the dump: only the endpoint identity (the LRU key) is emitted.
func TestNoHitLRU_DumpState_NoRequestDataLeak(t *testing.T) {
	ctx := utils.NewTestContext(t)
	s := nohitlru.NewNoHitLRU(ctx, nohitlru.NoHitLRUType, nil)

	const (
		canaryReqID  = "canary-request-id-DO-NOT-LEAK"
		canaryModel  = "canary-model-DO-NOT-LEAK"
		canaryHeader = "canary-header-DO-NOT-LEAK"
	)
	ep := newColdNS("pod-a")
	req := &scheduling.InferenceRequest{
		RequestID:   canaryReqID,
		TargetModel: canaryModel,
		Headers:     map[string]string{"X-Canary": canaryHeader},
	}
	s.Score(ctx, req, []scheduling.Endpoint{ep})
	s.PreRequest(ctx, req, &scheduling.SchedulingResult{
		PrimaryProfileName: "primary",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"primary": {TargetEndpoints: []scheduling.Endpoint{ep}},
		},
	})

	payload, err := s.DumpState()
	if err != nil {
		t.Fatalf("DumpState error: %v", err)
	}
	assertDumpStateKeys(t, payload)
	for _, canary := range []string{canaryReqID, canaryModel, canaryHeader} {
		if strings.Contains(string(payload), canary) {
			t.Fatalf("request canary %q leaked into DumpState payload: %s", canary, payload)
		}
	}
	// Sanity: the endpoint identity (the LRU key) is what should appear.
	if !strings.Contains(string(payload), "default/pod-a") {
		t.Fatalf("expected endpoint identity in payload, got: %s", payload)
	}
}

func TestNoHitLRUEdgeCases(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := nohitlru.NewNoHitLRU(ctx, nohitlru.NoHitLRUType, nil)

	t.Run("empty endpoints list", func(t *testing.T) {
		emptyEndpoints := []scheduling.Endpoint{}
		scores := scorer.Score(ctx, &scheduling.InferenceRequest{}, emptyEndpoints)
		if len(scores) != 0 {
			t.Errorf("Expected empty scores for empty endpoints list, got %d scores", len(scores))
		}
	})

	t.Run("nil endpoints list", func(t *testing.T) {
		scores := scorer.Score(ctx, &scheduling.InferenceRequest{}, nil)
		if scores == nil {
			t.Errorf("Expected non-nil scores map for nil endpoints list")
		}
		if len(scores) != 0 {
			t.Errorf("Expected empty scores for nil endpoints list, got %d scores", len(scores))
		}
	})

	t.Run("single endpoint returns 1.0", func(t *testing.T) {
		endpoints := []scheduling.Endpoint{newCold("pod-a")}
		scores := scorer.Score(ctx, &scheduling.InferenceRequest{}, endpoints)
		if scores[endpoints[0]] != 1.0 {
			t.Errorf("Expected single endpoint to get score 1.0, got %f", scores[endpoints[0]])
		}
	})
}

func TestNoHitLRUPrefillDecodeTracking(t *testing.T) {
	// Prefill worker endpoints
	prefillEndpointA := newColdNS("prefill-a")
	prefillEndpointB := newColdNS("prefill-b")

	// Decode worker endpoints
	decodeEndpointA := newColdNS("decode-a")
	decodeEndpointB := newColdNS("decode-b")

	prefillEndpoints := []scheduling.Endpoint{prefillEndpointA, prefillEndpointB}
	decodeEndpoints := []scheduling.Endpoint{decodeEndpointA, decodeEndpointB}

	ctx := context.Background()

	t.Run("P/D scenario - both profiles tracked separately", func(t *testing.T) {
		scorer := nohitlru.NewNoHitLRU(ctx, nohitlru.NoHitLRUType, nil)

		// First cold request with P/D (no attributes = cold).
		req1 := &scheduling.InferenceRequest{RequestID: "pd-request-1"}
		scorer.Score(ctx, req1, append(prefillEndpoints, decodeEndpoints...))

		pdResult := &scheduling.SchedulingResult{
			PrimaryProfileName: "decode",
			ProfileResults: map[string]*scheduling.ProfileRunResult{
				"prefill": {
					TargetEndpoints: []scheduling.Endpoint{prefillEndpointA},
				},
				"decode": {
					TargetEndpoints: []scheduling.Endpoint{decodeEndpointA},
				},
			},
		}
		scorer.PreRequest(ctx, req1, pdResult)

		req2 := &scheduling.InferenceRequest{RequestID: "pd-request-2"}
		prefillScores := scorer.Score(ctx, req2, prefillEndpoints)
		decodeScores := scorer.Score(ctx, req2, decodeEndpoints)

		if prefillScores[prefillEndpointB] <= prefillScores[prefillEndpointA] {
			t.Errorf("Expected prefill-b to score higher than prefill-a after prefill-a was used: %+v", prefillScores)
		}

		if decodeScores[decodeEndpointB] <= decodeScores[decodeEndpointA] {
			t.Errorf("Expected decode-b to score higher than decode-a after decode-a was used: %+v", decodeScores)
		}
	})

	t.Run("non-P/D scenario - only primary profile exists", func(t *testing.T) {
		req := &scheduling.InferenceRequest{RequestID: "non-pd-request"}
		scorer := nohitlru.NewNoHitLRU(ctx, nohitlru.NoHitLRUType, nil)
		scorer.Score(ctx, req, decodeEndpoints)

		result := &scheduling.SchedulingResult{
			PrimaryProfileName: "decode",
			ProfileResults: map[string]*scheduling.ProfileRunResult{
				"decode": {
					TargetEndpoints: []scheduling.Endpoint{decodeEndpointA},
				},
			},
		}
		scorer.PreRequest(ctx, req, result)

		req2 := &scheduling.InferenceRequest{RequestID: "non-pd-request-2"}
		scores := scorer.Score(ctx, req2, decodeEndpoints)

		if scores[decodeEndpointB] <= scores[decodeEndpointA] {
			t.Errorf("Expected decode-b to score higher than decode-a: %+v", scores)
		}
	})

	t.Run("nil scheduling result - graceful handling", func(_ *testing.T) {
		req := &scheduling.InferenceRequest{RequestID: "nil-result"}
		scorer := nohitlru.NewNoHitLRU(ctx, nohitlru.NoHitLRUType, nil)
		scorer.Score(ctx, req, decodeEndpoints)
		scorer.PreRequest(ctx, req, nil)
	})

	t.Run("empty profile results - graceful handling", func(_ *testing.T) {
		req := &scheduling.InferenceRequest{RequestID: "empty-results"}
		scorer := nohitlru.NewNoHitLRU(ctx, nohitlru.NoHitLRUType, nil)
		scorer.Score(ctx, req, decodeEndpoints)

		result := &scheduling.SchedulingResult{
			PrimaryProfileName: "decode",
			ProfileResults:     map[string]*scheduling.ProfileRunResult{},
		}
		scorer.PreRequest(ctx, req, result)
	})
}
