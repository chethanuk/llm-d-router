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

package integration

import (
	"errors"
	"fmt"
	"net"
	"os"
	"syscall"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	testutils "github.com/llm-d/llm-d-router/test/utils"
)

func TestIsAddrInUse(t *testing.T) {
	// A realistic wrap chain: the runnable does net.Listen and wraps the *net.OpError,
	// which itself wraps *os.SyscallError -> syscall.EADDRINUSE.
	wrapped := fmt.Errorf("gRPC server failed to listen - %w",
		&net.OpError{Op: "listen", Net: "tcp", Err: os.NewSyscallError("bind", syscall.EADDRINUSE)})

	tests := []struct {
		name string
		err  error
		want bool
	}{
		{name: "nil", err: nil, want: false},
		{name: "raw EADDRINUSE", err: syscall.EADDRINUSE, want: true},
		{name: "wrapped OpError chain", err: wrapped, want: true},
		{name: "stringified message", err: errors.New("listen tcp :8080: bind: address already in use"), want: true},
		{name: "stringified message mixed case", err: errors.New("bind: Address already in use"), want: true},
		{name: "unrelated error", err: errors.New("connection refused"), want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, isAddrInUse(tc.err))
		})
	}
}

func TestRetryOnAddrInUse(t *testing.T) {
	inUse := syscall.EADDRINUSE
	other := errors.New("boom")

	tests := []struct {
		name         string
		maxAttempts  int
		attemptErrs  []error // returned by attempt on successive calls
		wantAttempts int
		wantResets   int
		wantErr      bool
		wantErrIs    error // if set, the returned error must match this via errors.Is
	}{
		{
			name:         "success first try",
			maxAttempts:  5,
			attemptErrs:  []error{nil},
			wantAttempts: 1,
			wantResets:   0,
		},
		{
			name:         "one in-use then success",
			maxAttempts:  5,
			attemptErrs:  []error{inUse, nil},
			wantAttempts: 2,
			wantResets:   1,
		},
		{
			name:         "two in-use then success",
			maxAttempts:  5,
			attemptErrs:  []error{inUse, inUse, nil},
			wantAttempts: 3,
			wantResets:   2,
		},
		{
			name:         "budget exhausted",
			maxAttempts:  3,
			attemptErrs:  []error{inUse, inUse, inUse},
			wantAttempts: 3,
			wantResets:   2,
			wantErr:      true,
			wantErrIs:    inUse,
		},
		{
			name:         "non-retryable error returned immediately",
			maxAttempts:  5,
			attemptErrs:  []error{other},
			wantAttempts: 1,
			wantResets:   0,
			wantErr:      true,
			wantErrIs:    other,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			attempts, resets := 0, 0
			attempt := func() error {
				err := tc.attemptErrs[attempts]
				attempts++
				return err
			}
			reset := func() { resets++ }

			err := RetryOnAddrInUse(tc.maxAttempts, 0, attempt, reset)

			assert.Equal(t, tc.wantAttempts, attempts, "attempt count")
			assert.Equal(t, tc.wantResets, resets, "reset count")
			if tc.wantErr {
				require.Error(t, err)
				if tc.wantErrIs != nil {
					assert.ErrorIs(t, err, tc.wantErrIs)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestWaitExtProcReady(t *testing.T) {
	t.Run("ready when port is open", func(t *testing.T) {
		lis, err := net.Listen("tcp", "127.0.0.1:0")
		require.NoError(t, err)
		defer lis.Close()
		port := lis.Addr().(*net.TCPAddr).Port

		mgrErr := make(chan error, 1) // nothing sent: manager still running
		assert.NoError(t, WaitExtProcReady(port, mgrErr))
	})

	t.Run("returns manager bind error when port never comes up", func(t *testing.T) {
		// A closed port: acquire one then release it so nothing is listening.
		p, err := testutils.GetFreePort()
		require.NoError(t, err)

		bindErr := fmt.Errorf("gRPC server failed to listen - %w",
			&net.OpError{Op: "listen", Net: "tcp", Err: os.NewSyscallError("bind", syscall.EADDRINUSE)})
		mgrErr := make(chan error, 1)
		mgrErr <- bindErr

		got := WaitExtProcReady(p, mgrErr)
		require.Error(t, got)
		assert.True(t, isAddrInUse(got), "bind error should be classified as address-in-use")
	})
}
