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

package kv

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

func TestSGLangKV_Params(t *testing.T) {
	t.Setenv(envSGLangBootstrapPort, "")
	c, err := Build(SGLang, nil)
	if err != nil {
		t.Fatalf("Build(%q): %v", SGLang, err)
	}
	if c.Name() != SGLang {
		t.Fatalf("Name() = %q, want %q", c.Name(), SGLang)
	}

	reqCtx := &pipeline.RequestContext{
		KVTransferParams: map[string]any{
			fieldBootstrapHost: "10.0.0.42",
			fieldBootstrapPort: 8998,
			fieldBootstrapRoom: int64(12345),
		},
	}

	// Prefill: must have the required bootstrap fields; bootstrap_room is random so check type.
	prefill := c.PreparePrefillKVParams(context.Background(), reqCtx)
	if prefill["do_remote_decode"] != true {
		t.Errorf("prefill: do_remote_decode = %v, want true", prefill["do_remote_decode"])
	}
	if prefill["do_remote_prefill"] != false {
		t.Errorf("prefill: do_remote_prefill = %v, want false", prefill["do_remote_prefill"])
	}
	if prefill[fieldBootstrapPort] != defaultSGLangBootstrapPort {
		t.Errorf("prefill: %s = %v, want %d", fieldBootstrapPort, prefill[fieldBootstrapPort], defaultSGLangBootstrapPort)
	}
	room, ok := prefill[fieldBootstrapRoom].(string)
	if !ok || room == "" {
		t.Errorf("prefill: %s = %v (%T), want non-empty string", fieldBootstrapRoom, prefill[fieldBootstrapRoom], prefill[fieldBootstrapRoom])
	}

	// Decode: forwards prefill-response kv_transfer_params plus remote flags.
	wantDecode := map[string]any{
		fieldBootstrapHost:  "10.0.0.42",
		fieldBootstrapPort:  8998,
		fieldBootstrapRoom:  int64(12345),
		"do_remote_decode":  false,
		"do_remote_prefill": true,
	}
	if got := c.PrepareDecodeKVParams(context.Background(), reqCtx); !reflect.DeepEqual(got, wantDecode) {
		t.Errorf("decode params:\n got=%v\nwant=%v", got, wantDecode)
	}
}

func TestBuild_UnknownReturnsError(t *testing.T) {
	if _, err := Build("does-not-exist", nil); err == nil {
		t.Fatal("expected error for unknown connector")
	}
}

func TestBuild_EmptyReturnsDefault(t *testing.T) {
	c, err := Build("", nil)
	if err != nil {
		t.Fatal(err)
	}
	if c.Name() != DefaultKVConnectorName {
		t.Fatalf("default = %q, want %q", c.Name(), DefaultKVConnectorName)
	}
}

func TestConnectors_KVParams(t *testing.T) {
	cases := []struct {
		name           string
		decodeIncoming map[string]any
		wantPrefill    map[string]any
		wantDecode     map[string]any
	}{
		{
			name: NIXL,
			decodeIncoming: map[string]any{
				"block_id":  "block-999",
				"peer_host": "10.0.0.42",
				"peer_port": float64(7777),
			},
			wantPrefill: map[string]any{
				"do_remote_decode":  true,
				"do_remote_prefill": false,
				"remote_engine_id":  nil,
				"remote_block_ids":  nil,
				"remote_host":       nil,
				"remote_port":       nil,
			},
			wantDecode: map[string]any{
				"do_remote_decode":  false,
				"do_remote_prefill": true,
				"block_id":          "block-999",
				"peer_host":         "10.0.0.42",
				"peer_port":         float64(7777),
			},
		},
		{
			name:           SharedStorage,
			decodeIncoming: map[string]any{"ignored": "field"},
			wantPrefill:    map[string]any{"do_remote_decode": true, "do_remote_prefill": false},
			wantDecode:     map[string]any{"do_remote_decode": false, "do_remote_prefill": true},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c, err := Build(tc.name, nil)
			if err != nil {
				t.Fatalf("Build(%q): %v", tc.name, err)
			}
			if c.Name() != tc.name {
				t.Fatalf("Name() = %q, want %q", c.Name(), tc.name)
			}

			reqCtx := &pipeline.RequestContext{KVTransferParams: tc.decodeIncoming}

			if got := c.PreparePrefillKVParams(context.Background(), reqCtx); !reflect.DeepEqual(got, tc.wantPrefill) {
				t.Errorf("prefill params:\n got=%v\nwant=%v", got, tc.wantPrefill)
			}
			if got := c.PrepareDecodeKVParams(context.Background(), reqCtx); !reflect.DeepEqual(got, tc.wantDecode) {
				t.Errorf("decode params:\n got=%v\nwant=%v", got, tc.wantDecode)
			}
		})
	}
}

func TestBuild_SGLangBootstrapPortParam(t *testing.T) {
	tests := []struct {
		name    string
		value   any
		want    int
		wantErr bool
	}{
		{name: "int", value: 9100, want: 9100},
		{name: "int64", value: int64(9100), want: 9100},
		{name: "integral float64", value: float64(9100), want: 9100},
		{name: "json.Number", value: json.Number("9100"), want: 9100},
		{name: "numeric string", value: "9100", want: 9100},
		{name: "max valid", value: 65535, want: 65535},
		{name: "fractional float64", value: 9100.5, wantErr: true},
		{name: "bool", value: true, wantErr: true},
		{name: "nil", value: nil, wantErr: true},
		{name: "empty string", value: "", wantErr: true},
		{name: "zero", value: 0, wantErr: true},
		{name: "negative", value: -1, wantErr: true},
		{name: "above range", value: 65536, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, err := Build(SGLang, map[string]any{paramBootstrapPort: tt.value})
			if tt.wantErr {
				if err == nil {
					t.Fatalf("Build() = %v, want error", c)
				}
				return
			}
			if err != nil {
				t.Fatalf("Build(): %v", err)
			}
			got := c.PreparePrefillKVParams(context.Background(), &pipeline.RequestContext{})[fieldBootstrapPort]
			if got != tt.want {
				t.Errorf("%s = %v, want %d", fieldBootstrapPort, got, tt.want)
			}
		})
	}
}

func TestBuild_ParamsRejectedForParamlessConnectors(t *testing.T) {
	for _, name := range []string{NIXL, SharedStorage, ""} {
		t.Run("connector="+name, func(t *testing.T) {
			if _, err := Build(name, map[string]any{"x": 1}); err == nil {
				t.Error("Build() with params: want error")
			}
			if _, err := Build(name, map[string]any{}); err != nil {
				t.Errorf("Build() with empty params: %v", err)
			}
		})
	}
}
