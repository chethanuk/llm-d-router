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

package runner

import (
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/go-logr/logr/funcr"
	"github.com/spf13/pflag"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRun_StartupBuildLogUsesExecutableName(t *testing.T) {
	tests := []struct {
		name    string
		runner  func() *Runner
		wantMsg string
		wantNot string
	}{
		{
			name:    "default_name_is_llm_d_epp",
			runner:  NewRunner,
			wantMsg: `"llm-d-epp build"`,
		},
		{
			name:    "custom_name_overrides_default",
			runner:  func() *Runner { return NewRunner().WithExecutableName("custom-epp") },
			wantMsg: `"custom-epp build"`,
			wantNot: `"llm-d-epp build"`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			prevFlagSet := pflag.CommandLine
			prevArgs := os.Args
			pflag.CommandLine = pflag.NewFlagSet("epp-test", pflag.ContinueOnError)
			// Invalid TLS version makes Run return right after the build log.
			os.Args = []string{"epp", "--tls-min-version=1.3"}
			t.Cleanup(func() {
				pflag.CommandLine = prevFlagSet
				os.Args = prevArgs
			})

			var mu sync.Mutex
			var logged strings.Builder
			previous := setupLog
			setupLog = funcr.New(func(prefix, args string) {
				mu.Lock()
				defer mu.Unlock()
				logged.WriteString(prefix + " " + args + "\n")
			}, funcr.Options{})
			t.Cleanup(func() { setupLog = previous })

			err := tc.runner().Run(t.Context())
			require.ErrorContains(t, err, `invalid tls-min-version "1.3"`)

			mu.Lock()
			logs := logged.String()
			mu.Unlock()
			assert.Contains(t, logs, tc.wantMsg)
			if tc.wantNot != "" {
				assert.NotContains(t, logs, tc.wantNot)
			}
		})
	}
}
