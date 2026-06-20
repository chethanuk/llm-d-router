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

package collectors

import (
	"strings"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
)

var (
	descInFlightRequests = prometheus.NewDesc(
		"llm_d_epp_inflight_requests",
		metricsutil.HelpMsgWithStability("Current number of in-flight requests per endpoint, as tracked by the in-flight load producer.", compbasemetrics.ALPHA),
		[]string{"endpoint_name", "namespace", "producer_name"}, nil,
	)
	descInFlightTokens = prometheus.NewDesc(
		"llm_d_epp_inflight_tokens",
		metricsutil.HelpMsgWithStability("Current number of in-flight tokens per endpoint (uncached prompt tokens, optionally plus estimated output), as tracked by the in-flight load producer.", compbasemetrics.ALPHA),
		[]string{"endpoint_name", "namespace", "producer_name"}, nil,
	)
)

// InFlightLoadSnapshotter is the read side of an in-flight load producer the
// collector needs: per-endpoint in-flight request and token counts keyed by the
// endpoint's "namespace/name".
type InFlightLoadSnapshotter interface {
	InFlightRequestsSnapshot() map[string]int64
	InFlightTokensSnapshot() map[string]int64
}

type inFlightLoadCollector struct {
	producerName string
	snapshotter  InFlightLoadSnapshotter
}

var _ prometheus.Collector = &inFlightLoadCollector{}

// NewInFlightLoadCollector returns a prometheus.Collector that emits per-endpoint
// in-flight request and token gauges from the given producer's live snapshot.
// The producer_name label carries producerName so multiple configured producers
// emit distinct, collision-free series.
func NewInFlightLoadCollector(producerName string, s InFlightLoadSnapshotter) prometheus.Collector {
	return &inFlightLoadCollector{producerName: producerName, snapshotter: s}
}

// RegisterInFlightLoadCollector registers a collector for the producer into the
// controller-runtime metrics registry. Returns the registry error (e.g. a
// duplicate registration) for the caller to log; it never panics.
func RegisterInFlightLoadCollector(producerName string, s InFlightLoadSnapshotter) error {
	return ctrlmetrics.Registry.Register(NewInFlightLoadCollector(producerName, s))
}

func (c *inFlightLoadCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- descInFlightRequests
	ch <- descInFlightTokens
}

func (c *inFlightLoadCollector) Collect(ch chan<- prometheus.Metric) {
	if c.snapshotter == nil {
		return
	}
	c.emit(ch, descInFlightRequests, c.snapshotter.InFlightRequestsSnapshot())
	c.emit(ch, descInFlightTokens, c.snapshotter.InFlightTokensSnapshot())
}

func (c *inFlightLoadCollector) emit(ch chan<- prometheus.Metric, desc *prometheus.Desc, counts map[string]int64) {
	for endpointID, v := range counts {
		name, namespace := splitNamespacedName(endpointID)
		ch <- prometheus.MustNewConstMetric(desc, prometheus.GaugeValue, float64(v), name, namespace, c.producerName)
	}
}

// splitNamespacedName splits a "namespace/name" key (the string form of a
// k8s types.NamespacedName) into its name and namespace. A key with no
// separator is treated as a bare name with an empty namespace.
func splitNamespacedName(id string) (name, namespace string) {
	if i := strings.IndexByte(id, '/'); i >= 0 {
		return id[i+1:], id[:i]
	}
	return id, ""
}
