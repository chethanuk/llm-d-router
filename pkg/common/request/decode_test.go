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

package request

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDecodeRequestBody(t *testing.T) {
	ins := map[string]struct{}{"stream": {}, "params": {}}
	toolsRaw := `[{"function":{"parameters":{"properties":{"b":{},"a":{}}}}}]`

	tests := []struct {
		name      string
		raw       string
		inspected map[string]struct{}
		want      map[string]any
		// wantJSON, when set, is the expected json.Marshal of the result.
		wantJSON   string
		wantErrMsg string
		// wantErrAs, when set, is an errors.As target. Type error text comes from
		// encoding/json and is not asserted.
		wantErrAs any
	}{
		{
			name:      "decodes inspected fields to Go values",
			raw:       `{"stream":true,"params":{"n":5}}`,
			inspected: ins,
			want:      map[string]any{"stream": true, "params": map[string]any{"n": 5.0}},
		},
		{
			name:      "keeps uninspected key order on re-marshal",
			raw:       `{"tools":` + toolsRaw + `}`,
			inspected: ins,
			want:      map[string]any{"tools": json.RawMessage(toolsRaw)},
			wantJSON:  `{"tools":` + toolsRaw + `}`,
		},
		{
			name:      "keeps uninspected bytes including whitespace",
			raw:       `{"other":{"b":1,  "a":2}}`,
			inspected: ins,
			want:      map[string]any{"other": json.RawMessage(`{"b":1,  "a":2}`)},
		},
		{
			name:      "sorts keys of an inspected field on re-marshal",
			raw:       `{"params":{"b":1,"a":2}}`,
			inspected: ins,
			want:      map[string]any{"params": map[string]any{"a": 2.0, "b": 1.0}},
			wantJSON:  `{"params":{"a":2,"b":1}}`,
		},
		{
			name:      "nil set leaves every field raw",
			raw:       `{"stream":true}`,
			inspected: nil,
			want:      map[string]any{"stream": json.RawMessage(`true`)},
		},
		{
			name:      "empty set leaves every field raw",
			raw:       `{"stream":true}`,
			inspected: map[string]struct{}{},
			want:      map[string]any{"stream": json.RawMessage(`true`)},
		},
		{
			name:      "inspected null is present with a nil value",
			raw:       `{"stream":null}`,
			inspected: ins,
			want:      map[string]any{"stream": nil},
		},
		{
			name:      "uninspected null stays raw",
			raw:       `{"other":null}`,
			inspected: ins,
			want:      map[string]any{"other": json.RawMessage(`null`)},
		},
		{
			name:      "field match is case sensitive",
			raw:       `{"Stream":true}`,
			inspected: ins,
			want:      map[string]any{"Stream": json.RawMessage(`true`)},
		},
		{
			name:      "empty object gives an empty non-nil map",
			raw:       `{}`,
			inspected: ins,
			want:      map[string]any{},
		},
		{
			name:       "rejects a null body as not an object",
			raw:        `null`,
			inspected:  ins,
			wantErrMsg: "request body is not a JSON object",
		},
		{
			name:      "rejects an array body with a type error",
			raw:       `[1,2]`,
			inspected: ins,
			wantErrAs: new(*json.UnmarshalTypeError),
		},
		{
			name:      "rejects a string body with a type error",
			raw:       `"text"`,
			inspected: ins,
			wantErrAs: new(*json.UnmarshalTypeError),
		},
		{
			name:      "rejects a number body with a type error",
			raw:       `12`,
			inspected: ins,
			wantErrAs: new(*json.UnmarshalTypeError),
		},
		{
			name:      "rejects empty input with a syntax error",
			raw:       ``,
			inspected: ins,
			wantErrAs: new(*json.SyntaxError),
		},
		{
			name:      "rejects a truncated object with a syntax error",
			raw:       `{"a":`,
			inspected: ins,
			wantErrAs: new(*json.SyntaxError),
		},
		{
			name:      "rejects trailing data with a syntax error",
			raw:       `{} x`,
			inspected: ins,
			wantErrAs: new(*json.SyntaxError),
		},
		{
			name:      "rejects a malformed field value with a syntax error",
			raw:       `{"stream":x}`,
			inspected: ins,
			wantErrAs: new(*json.SyntaxError),
		},
		{
			name:      "rejects an inspected number outside float64 range with a type error",
			raw:       `{"stream":1e999}`,
			inspected: ins,
			wantErrAs: new(*json.UnmarshalTypeError),
		},
		{
			name:      "keeps an uninspected number outside float64 range raw",
			raw:       `{"other":1e999}`,
			inspected: ins,
			want:      map[string]any{"other": json.RawMessage(`1e999`)},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := DecodeRequestBody([]byte(tc.raw), tc.inspected)

			if tc.wantErrMsg != "" || tc.wantErrAs != nil {
				require.Error(t, err)
				require.Nil(t, got)
				if tc.wantErrMsg != "" {
					require.EqualError(t, err, tc.wantErrMsg)
				}
				if tc.wantErrAs != nil {
					require.ErrorAs(t, err, tc.wantErrAs)
				}
				return
			}

			require.NoError(t, err)
			require.Equal(t, tc.want, got)
			if tc.wantJSON != "" {
				encoded, err := json.Marshal(got)
				require.NoError(t, err)
				require.Equal(t, tc.wantJSON, string(encoded))
			}
		})
	}
}
