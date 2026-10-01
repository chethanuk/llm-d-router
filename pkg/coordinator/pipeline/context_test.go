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

package pipeline

import (
	"encoding/json"
	"net/http"
	"testing"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

func TestForwardedHeaders_NormalizesKeysToLowercase(t *testing.T) {
	rc := &RequestContext{
		OriginalHeaders: http.Header{
			"X-Request-Id":    {"abc-123"},
			"X-Forwarded-For": {"10.0.0.1"},
		},
	}

	out := rc.ForwardedHeaders()

	if got := out["x-request-id"]; got != "abc-123" {
		t.Fatalf("x-request-id = %q, want %q", got, "abc-123")
	}
	if _, ok := out["X-Request-Id"]; ok {
		t.Errorf("canonical key X-Request-Id should not be present; keys must be lowercased")
	}
	if got := out["x-forwarded-for"]; got != "10.0.0.1" {
		t.Errorf("x-forwarded-for = %q, want %q", got, "10.0.0.1")
	}
}

// A forwarding step re-stamps x-request-id from the lowercase constant. The
// forwarded copy must use the same lowercase key so the two do not coexist as
// distinct map entries.
func TestForwardedHeaders_RequestIDDoesNotDuplicateOnRestamp(t *testing.T) {
	rc := &RequestContext{
		RequestID: "abc-123",
		OriginalHeaders: http.Header{
			"X-Request-Id": {"abc-123"},
		},
	}

	headers := rc.ForwardedHeaders()
	headers["x-request-id"] = rc.RequestID

	count := 0
	for k := range headers {
		if k == "x-request-id" || k == "X-Request-Id" {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("request-id header present %d times, want 1: %v", count, headers)
	}
}

func TestForwardedHeaders_ExcludesHopByHopAndContentHeaders(t *testing.T) {
	rc := &RequestContext{
		OriginalHeaders: http.Header{
			"Connection":     {"keep-alive"},
			"Content-Length": {"42"},
			"Host":           {"example.com"},
			"Content-Type":   {"application/json"},
			"X-Request-Id":   {"abc-123"},
		},
	}

	out := rc.ForwardedHeaders()

	for _, excluded := range []string{"connection", "content-length", "host", "content-type"} {
		if _, ok := out[excluded]; ok {
			t.Errorf("header %q should be excluded from forwarded headers", excluded)
		}
	}
	if _, ok := out["x-request-id"]; !ok {
		t.Errorf("x-request-id should be forwarded")
	}
}

func TestForwardedHeaders_ExcludesInternalRoutingHeaders(t *testing.T) {
	rc := &RequestContext{
		OriginalHeaders: http.Header{
			"EPP-Profile":   {"decode"},
			"X-Request-Id":  {"abc-123"},
			"Authorization": {"Bearer token"},
		},
	}

	out := rc.ForwardedHeaders()

	if _, ok := out["epp-profile"]; ok {
		t.Fatalf("epp-profile should not be forwarded: %v", out)
	}
	if got := out["x-request-id"]; got != "abc-123" {
		t.Errorf("x-request-id = %q, want %q", got, "abc-123")
	}
	if got := out["authorization"]; got != "Bearer token" {
		t.Errorf("authorization = %q, want %q", got, "Bearer token")
	}
}

func TestForwardedHeaders_UsesCoordinatorRevisionDecisionID(t *testing.T) {
	rc := &RequestContext{
		RevisionDecisionID: "coordinator-decision-id",
		OriginalHeaders: http.Header{
			"X-Llm-D-Revision-Decision-Id": {"client-decision-id"},
		},
	}

	out := rc.ForwardedHeaders()
	if got := out[reqcommon.RevisionDecisionIDHeaderKey]; got != rc.RevisionDecisionID {
		t.Errorf("revision decision ID = %q, want %q", got, rc.RevisionDecisionID)
	}
}

func TestForwardedHeaders_NilOriginalHeaders(t *testing.T) {
	rc := &RequestContext{}
	if out := rc.ForwardedHeaders(); len(out) != 0 {
		t.Fatalf("expected empty map for nil OriginalHeaders, got %v", out)
	}
}

func TestForwardedHeaders_UsesRevisionDecisionIDWithoutOriginalHeaders(t *testing.T) {
	rc := &RequestContext{RevisionDecisionID: "coordinator-decision-id"}
	out := rc.ForwardedHeaders()
	if got := out[reqcommon.RevisionDecisionIDHeaderKey]; got != rc.RevisionDecisionID {
		t.Errorf("revision decision ID = %q, want %q", got, rc.RevisionDecisionID)
	}
}

func TestRequestContext_MarshalBody(t *testing.T) {
	tests := []struct {
		name     string
		original string
		mutate   func(map[string]any)
		want     string
	}{
		{"untouched fields keep client order", `{"b":{"z":1,"a":2},"a":[1, 2]}`, nil, `{"a":[1, 2],"b":{"z":1,"a":2}}`},
		{"changed value re-marshaled, others raw", `{"t":{"z":1,"a":2},"m":{"z":1,"a":2}}`, func(b map[string]any) { b["m"] = map[string]any{"z": 1.0, "a": 3.0} }, `{"m":{"a":3,"z":1},"t":{"z":1,"a":2}}`},
		{"injected key marshaled fresh", `{"t":{"z":1,"a":2}}`, func(b map[string]any) { b["kv"] = "x" }, `{"kv":"x","t":{"z":1,"a":2}}`},
		{"null preserved", `{"tool_choice":null}`, nil, `{"tool_choice":null}`},
		{"empty original falls back", ``, nil, `{"a":{"a":1,"z":2}}`},
		{"unparsable original falls back", `not json`, nil, `{"a":{"a":1,"z":2}}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			body := map[string]any{"a": map[string]any{"z": 2.0, "a": 1.0}}
			if tc.original != "" && tc.original != "not json" {
				body = nil
				if err := json.Unmarshal([]byte(tc.original), &body); err != nil {
					t.Fatal(err)
				}
			}
			if tc.mutate != nil {
				tc.mutate(body)
			}
			rc := &RequestContext{OriginalBody: []byte(tc.original)}
			got, err := rc.MarshalBody(body)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != tc.want {
				t.Errorf("got %s want %s", got, tc.want)
			}
		})
	}
}
