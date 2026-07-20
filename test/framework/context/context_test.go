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

package context

import (
	"context"
	"testing"

	"github.com/go-logr/logr"
)

func TestNewTestContext(t *testing.T) {
	tests := []struct {
		name  string
		check func(t *testing.T, ctx context.Context)
	}{
		{
			// logr.FromContext is the error-returning form: it reports a missing logger.
			// controller-runtime's log.FromContext returns a delegating logger even for a
			// bare context, so it cannot detect a forgotten IntoContext.
			name: "logger is actually attached to the context",
			check: func(t *testing.T, ctx context.Context) {
				if _, err := logr.FromContext(ctx); err != nil {
					t.Fatalf("logr.FromContext() returned error: %v", err)
				}
			},
		},
		{
			name: "context is not canceled at creation",
			check: func(t *testing.T, ctx context.Context) {
				select {
				case <-ctx.Done():
					t.Fatalf("context is already done at creation: %v", ctx.Err())
				default:
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := NewTestContext(t)
			if ctx == nil {
				t.Fatal("NewTestContext() returned nil context")
			}
			tt.check(t, ctx)
		})
	}
}
