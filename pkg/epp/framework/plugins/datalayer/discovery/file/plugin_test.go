/*
Copyright 2025 The Kubernetes Authors.

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

package file

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// recordingNotifier captures Upsert and Delete calls for assertions.
type recordingNotifier struct {
	mu       sync.Mutex
	upserted []*fwkdl.EndpointMetadata
	deleted  []types.NamespacedName
}

func (r *recordingNotifier) Upsert(meta *fwkdl.EndpointMetadata) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.upserted = append(r.upserted, meta)
}

func (r *recordingNotifier) Delete(id types.NamespacedName) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.deleted = append(r.deleted, id)
}

func (r *recordingNotifier) upsertedNames() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	names := make([]string, len(r.upserted))
	for i, m := range r.upserted {
		names[i] = m.NamespacedName.String()
	}
	return names
}

func writeTemp(t *testing.T, content string) string {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "endpoints-*.yaml")
	require.NoError(t, err)
	_, err = f.WriteString(content)
	require.NoError(t, err)
	require.NoError(t, f.Close())
	return f.Name()
}

func newFD(path string, watch bool) *FileDiscovery {
	return &FileDiscovery{
		path:      path,
		watchFile: watch,
		endpoints: make(map[types.NamespacedName]struct{}),
		ready:     make(chan struct{}),
	}
}

const validYAML = `
endpoints:
  - name: ep1
    namespace: ns1
    address: "10.0.0.1"
    port: "8000"
  - name: ep2
    address: "10.0.0.2"
    port: "8001"
`

func TestFactory_MissingPath(t *testing.T) {
	_, err := Factory("", fwkplugin.StrictDecoder(json.RawMessage(`{}`)), nil)
	assert.ErrorContains(t, err, "'path' parameter is required")
}

func TestFactory_InvalidJSON(t *testing.T) {
	_, err := Factory("", fwkplugin.StrictDecoder(json.RawMessage(`{bad json`)), nil)
	assert.ErrorContains(t, err, "failed to parse parameters")
}

func TestFactory_ValidParams(t *testing.T) {
	path := writeTemp(t, validYAML)
	plugin, err := Factory("my-discovery", fwkplugin.StrictDecoder(json.RawMessage(`{"path":"`+path+`"}`)), nil)
	require.NoError(t, err)
	assert.Equal(t, PluginType, plugin.TypedName().Type)
	assert.Equal(t, "my-discovery", plugin.TypedName().Name)
}

func TestFactory_DefaultName(t *testing.T) {
	path := writeTemp(t, validYAML)
	plugin, err := Factory("", fwkplugin.StrictDecoder(json.RawMessage(`{"path":"`+path+`"}`)), nil)
	require.NoError(t, err)
	assert.Equal(t, PluginType, plugin.TypedName().Name)
}

func TestStart_LoadsEndpoints(t *testing.T) {
	path := writeTemp(t, validYAML)
	notifier := &recordingNotifier{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	fd := newFD(path, false)
	require.NoError(t, fd.Start(ctx, notifier))

	assert.ElementsMatch(t, []string{"ns1/ep1", "default/ep2"}, notifier.upsertedNames())
	assert.Empty(t, notifier.deleted)

	select {
	case <-fd.Ready():
	default:
		t.Fatal("Ready() channel should be closed after a successful initial load")
	}
}

func TestReady_StaysOpenWhenInitialLoadFails(t *testing.T) {
	fd := newFD("/nonexistent/endpoints.yaml", false)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := fd.Start(ctx, &recordingNotifier{})
	require.Error(t, err)

	select {
	case <-fd.Ready():
		t.Fatal("Ready() must not be closed when initial load fails")
	default:
	}
}

func TestStart_DefaultNamespace(t *testing.T) {
	path := writeTemp(t, `
endpoints:
  - name: ep1
    address: "10.0.0.1"
    port: "8000"
`)
	notifier := &recordingNotifier{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	require.NoError(t, newFD(path, false).Start(ctx, notifier))
	assert.Equal(t, "default", notifier.upserted[0].NamespacedName.Namespace)
}

func TestStart_MetricsHostIsAddressPort(t *testing.T) {
	path := writeTemp(t, `
endpoints:
  - name: ep1
    address: "10.0.0.1"
    port: "8000"
`)
	notifier := &recordingNotifier{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	require.NoError(t, newFD(path, false).Start(ctx, notifier))
	assert.Equal(t, "10.0.0.1:8000", notifier.upserted[0].MetricsHost)
}

func TestStart_MissingFile(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := newFD("/nonexistent/endpoints.yaml", false).Start(ctx, &recordingNotifier{})
	assert.ErrorContains(t, err, "initial load failed")
}

func TestStart_InvalidIP(t *testing.T) {
	path := writeTemp(t, `
endpoints:
  - name: ep1
    address: "not-an-ip"
    port: "8000"
`)
	err := newFD(path, false).Start(context.Background(), &recordingNotifier{})
	assert.ErrorContains(t, err, "invalid IPv4 address")
}

func TestStart_RejectsIPv6(t *testing.T) {
	path := writeTemp(t, `
endpoints:
  - name: ep1
    address: "::1"
    port: "8000"
`)
	err := newFD(path, false).Start(context.Background(), &recordingNotifier{})
	assert.ErrorContains(t, err, "invalid IPv4 address")
}

func TestStart_InvalidPort(t *testing.T) {
	path := writeTemp(t, `
endpoints:
  - name: ep1
    address: "10.0.0.1"
    port: "99999"
`)
	err := newFD(path, false).Start(context.Background(), &recordingNotifier{})
	assert.ErrorContains(t, err, "invalid port")
}

func TestStart_FileTooLarge(t *testing.T) {
	content := strings.Repeat("x", maxEndpointsFileSize+1)
	path := writeTemp(t, content)
	err := newFD(path, false).Start(context.Background(), &recordingNotifier{})
	assert.ErrorContains(t, err, "exceeds 1 MiB limit")
}

func TestStart_DeletesRemovedEndpoints(t *testing.T) {
	path := writeTemp(t, validYAML)
	fd := newFD(path, false)
	notifier := &recordingNotifier{}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	require.NoError(t, fd.Start(ctx, notifier))
	assert.Len(t, notifier.upserted, 2)

	require.NoError(t, os.WriteFile(path, []byte(`
endpoints:
  - name: ep1
    namespace: ns1
    address: "10.0.0.1"
    port: "8000"
`), 0o600))
	notifier2 := &recordingNotifier{}
	ctx2, cancel2 := context.WithCancel(context.Background())
	cancel2()
	require.NoError(t, fd.Start(ctx2, notifier2))

	assert.Len(t, notifier2.upserted, 1)
	assert.Len(t, notifier2.deleted, 1)
	assert.Equal(t, types.NamespacedName{Name: "ep2", Namespace: "default"}, notifier2.deleted[0])
}

func TestStart_WatchFileReloadsOnWrite(t *testing.T) {
	path := writeTemp(t, validYAML)
	fd := newFD(path, true)
	notifier := &recordingNotifier{}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	done := make(chan error, 1)
	go func() { done <- fd.Start(ctx, notifier) }()

	newContent := []byte(`
endpoints:
  - name: ep3
    address: "10.0.0.3"
    port: "9000"
`)
	// Re-touch the file each poll so the write that lands after the watcher
	// is attached is the one that triggers the reload. Avoids racing on the
	// gap between Start()'s initial load and watcher.Add().
	require.Eventually(t, func() bool {
		if err := os.WriteFile(path, newContent, 0o600); err != nil {
			return false
		}
		notifier.mu.Lock()
		defer notifier.mu.Unlock()
		for _, m := range notifier.upserted {
			if m.NamespacedName.Name == "ep3" {
				return true
			}
		}
		return false
	}, 2*time.Second, 50*time.Millisecond)

	cancel()
	assert.NoError(t, <-done)
}

func TestDumpState(t *testing.T) {
	tests := []struct {
		name   string
		build  func(t *testing.T) *FileDiscovery
		assert func(t *testing.T, s fileDiscoveryState)
	}{
		{
			name: "populated",
			build: func(t *testing.T) *FileDiscovery {
				fd := newFD(writeTemp(t, validYAML), false)
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				require.NoError(t, fd.Start(ctx, &recordingNotifier{}))
				return fd
			},
			assert: func(t *testing.T, s fileDiscoveryState) {
				assert.Equal(t, []string{"default/ep2", "ns1/ep1"}, s.Endpoints)
				assert.Equal(t, 2, s.TotalEndpoints)
				assert.Equal(t, maxDebugDumpEndpoints, s.MaxEndpoints)
				assert.False(t, s.Truncated)
				assert.True(t, s.Ready)
				assert.False(t, s.WatchFile)
				assert.NotEmpty(t, s.Path)
			},
		},
		{
			name:  "empty",
			build: func(t *testing.T) *FileDiscovery { return newFD("/x.yaml", false) },
			assert: func(t *testing.T, s fileDiscoveryState) {
				assert.Empty(t, s.Endpoints)
				assert.Equal(t, 0, s.TotalEndpoints)
				assert.False(t, s.Truncated)
				assert.False(t, s.Ready)
			},
		},
		{
			name:  "watch_enabled",
			build: func(t *testing.T) *FileDiscovery { return newFD("/x.yaml", true) },
			assert: func(t *testing.T, s fileDiscoveryState) {
				assert.True(t, s.WatchFile)
			},
		},
		{
			name: "bounded_truncates",
			build: func(t *testing.T) *FileDiscovery {
				fd := newFD("/x.yaml", false)
				for i := 0; i < maxDebugDumpEndpoints+50; i++ {
					fd.endpoints[types.NamespacedName{
						Namespace: "ns",
						Name:      fmt.Sprintf("ep-%04d", i),
					}] = struct{}{}
				}
				return fd
			},
			assert: func(t *testing.T, s fileDiscoveryState) {
				assert.Len(t, s.Endpoints, maxDebugDumpEndpoints)
				assert.Equal(t, maxDebugDumpEndpoints+50, s.TotalEndpoints)
				assert.True(t, s.Truncated)
				assert.True(t, sort.StringsAreSorted(s.Endpoints))
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fd := tc.build(t)
			raw, err := fd.DumpState()
			require.NoError(t, err)
			require.True(t, json.Valid(raw))
			var s fileDiscoveryState
			require.NoError(t, json.Unmarshal(raw, &s))
			tc.assert(t, s)
		})
	}
}

// TestDumpState_ConcurrentReload proves the endpoints mutex is necessary: load()
// (the reload goroutine) and DumpState() (the HTTP handler goroutine) run
// concurrently against the same map. It must pass under -race.
func TestDumpState_ConcurrentReload(t *testing.T) {
	fd := newFD(writeTemp(t, validYAML), false)
	notifier := &recordingNotifier{}
	// Seed once so there is a map to read while reloads swap it.
	require.NoError(t, fd.load(notifier))

	var wg sync.WaitGroup
	const iterations = 200

	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			if err := fd.load(notifier); err != nil {
				t.Errorf("load: %v", err)
				return
			}
		}
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			raw, err := fd.DumpState()
			if err != nil {
				t.Errorf("DumpState: %v", err)
				return
			}
			if !json.Valid(raw) {
				t.Errorf("DumpState returned invalid JSON")
				return
			}
		}
	}()

	wg.Wait()
}
