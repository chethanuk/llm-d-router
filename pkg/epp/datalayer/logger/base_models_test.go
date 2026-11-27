/*
Copyright 2025 The Kubernetes Authors.
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
package logger

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr/funcr"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	attrmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/models"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
	poolutil "github.com/llm-d/llm-d-router/pkg/epp/util/pool"
)

const baseModelsGauge = "llm_d_epp_pool_base_models"

// reported is the /v1/models list of one endpoint; nil means it has not reported.
type reported []attrmodels.ModelData

type baseModelsDataStore struct {
	name      string
	endpoints []reported
}

func (d *baseModelsDataStore) PoolGet() (*datalayer.EndpointPool, error) {
	pool := &v1.InferencePool{Spec: v1.InferencePoolSpec{TargetPorts: []v1.Port{{Number: 8000}}}}
	pool.Name = d.name
	return poolutil.InferencePoolToEndpointPool(pool), nil
}

func (d *baseModelsDataStore) PodList(predicate func(fwkdl.Endpoint) bool) []fwkdl.Endpoint {
	slot := fwkdl.NewSlot[attrmodels.ModelDataCollection](attrmodels.ModelsAttributeKey)
	var res []fwkdl.Endpoint
	for i, models := range d.endpoints {
		meta := &fwkdl.EndpointMetadata{
			ID:      types.NamespacedName{Name: fmt.Sprintf("pod%d", i), Namespace: "default"},
			Address: fmt.Sprintf("10.0.0.%d:8000", i),
		}
		ep := fwkdl.NewEndpoint(meta, &fwkdl.Metrics{UpdateTime: time.Now()})
		if models != nil {
			slot.Put(ep.GetAttributes(), attrmodels.ModelDataCollection(models))
		}
		if predicate(ep) {
			res = append(res, ep)
		}
	}
	return res
}

func TestRefreshPrometheusMetricsPoolBaseModels(t *testing.T) {
	llama := attrmodels.ModelData{ID: "llama"}
	qwen := attrmodels.ModelData{ID: "qwen"}
	noID := attrmodels.ModelData{}
	loraOnLlama := attrmodels.ModelData{ID: "sql-lora", Parent: "llama"}
	loraOnQwen := attrmodels.ModelData{ID: "chat-lora", Parent: "qwen"}

	const (
		warn = `"level"=0 "msg"="Warning: endpoints in the pool report different base models" "pool"="test-pool" "baseModels"=["llama" "qwen"]`
		info = `"level"=0 "msg"="Endpoints in the pool report the same base model again" "pool"="test-pool"`
	)

	tests := []struct {
		name      string
		steps     [][]reported // ready endpoints seen on each refresh
		wantGauge float64
		wantLogs  []string // base model log lines over all steps, in order
	}{
		{
			name:      "endpoints_on_different_base_models",
			steps:     [][]reported{{{llama}, {qwen}}, {{llama}, {qwen}}},
			wantGauge: 2,
			wantLogs:  []string{warn},
		},
		{
			name:      "lora_adapters_do_not_count",
			steps:     [][]reported{{{llama, loraOnLlama}, {llama, loraOnQwen}}},
			wantGauge: 1,
		},
		{
			name:      "different_loras_on_different_bases_still_mismatch",
			steps:     [][]reported{{{llama, loraOnLlama}, {qwen, loraOnQwen}}},
			wantGauge: 2,
			wantLogs:  []string{warn},
		},
		{
			name:      "unreported_endpoint_is_ignored",
			steps:     [][]reported{{{llama}, nil}},
			wantGauge: 1,
		},
		{
			name:      "entry_without_id_is_ignored",
			steps:     [][]reported{{{llama}, {noID}}},
			wantGauge: 1,
		},
		{
			name:      "mismatch_clears",
			steps:     [][]reported{{{llama}, {qwen}}, {{llama}, {llama}}},
			wantGauge: 1,
			wantLogs:  []string{warn, info},
		},
		{
			name:      "mismatch_then_no_ready_endpoints",
			steps:     [][]reported{{{llama}, {qwen}}, {}},
			wantGauge: 0,
			wantLogs:  []string{warn},
		},
		{
			name:      "mismatch_survives_endpoints_going_away_and_back",
			steps:     [][]reported{{{llama}, {qwen}}, {}, {{llama}, {qwen}}},
			wantGauge: 2,
			wantLogs:  []string{warn},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metrics.Register()
			metrics.Reset()

			var lines []string
			logger := funcr.New(func(prefix, args string) {
				if strings.Contains(args, "base model") {
					lines = append(lines, args)
				}
			}, funcr.Options{Verbosity: 5})

			// Fresh state per case, the way runPrometheusRefresher owns it.
			mismatched := map[string]bool{}
			ds := &baseModelsDataStore{name: "test-pool"}
			for _, step := range tt.steps {
				ds.endpoints = step
				refreshPrometheusMetrics(logger, ds, time.Minute, mismatched)
			}

			families, err := ctrlmetrics.Registry.Gather()
			require.NoError(t, err)
			var series []*dto.Metric
			for _, f := range families {
				if f.GetName() == baseModelsGauge {
					series = f.GetMetric()
				}
			}
			require.Len(t, series, 1, "metric %s series", baseModelsGauge)
			require.Len(t, series[0].GetLabel(), 1)
			assert.Equal(t, "name", series[0].GetLabel()[0].GetName())
			assert.Equal(t, "test-pool", series[0].GetLabel()[0].GetValue())
			assert.Equal(t, tt.wantGauge, series[0].GetGauge().GetValue())

			assert.Equal(t, tt.wantLogs, lines)
		})
	}
}
