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

package epp

import (
	"context"
	"fmt"
	"io"
	"net"
	"testing"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// --- Execution Helpers ---

// StreamedRequest is a helper for Full-Duplex Streaming test scenarios.
// It performs the following actions:
//  1. Sends all requests in the provided slice to the server.
//  2. Listens for responses on the stream until 'expectedResponses' count is reached.
//  3. Enforces a 10-second timeout to prevent deadlocks if the server hangs.
//  4. Handles io.EOF gracefully (server closed stream).
func StreamedRequest(
	t *testing.T,
	client extProcPb.ExternalProcessor_ProcessClient,
	requests []*extProcPb.ProcessingRequest,
	expectedResponses int,
) ([]*extProcPb.ProcessingResponse, error) {
	t.Helper()

	// 1. Send Phase
	for _, req := range requests {
		t.Logf("Sending request: %v", req)
		if err := client.Send(req); err != nil {
			t.Logf("Failed to send request: %v", err)
			return nil, err
		}
	}

	// 2. Receive Phase
	// We use a channel and a separate goroutine for receiving to allow for a strict timeout via select{}.
	type recvResult struct {
		res *extProcPb.ProcessingResponse
		err error
	}

	// Buffered channel avoids blocking the goroutine on the last read.
	recvChan := make(chan recvResult, expectedResponses+1)

	// Start reading in background.
	go func() {
		for range expectedResponses {
			res, err := client.Recv()
			recvChan <- recvResult{res, err}
			if err != nil {
				return // Stop reading on error or EOF.
			}
		}
	}()

	var responses []*extProcPb.ProcessingResponse

	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()

	// Collect results with timeout.
	for i := range expectedResponses {
		select {
		case <-ctx.Done():
			t.Logf("Timeout waiting for response %d of %d: %v", i+1, expectedResponses, ctx.Err())
			return responses, fmt.Errorf("timeout waiting for responses: %w", ctx.Err())

		case result := <-recvChan:
			if result.err != nil {
				// io.EOF is a valid termination from the server side (e.g. rejection).
				if result.err == io.EOF {
					return responses, nil
				}
				t.Logf("Failed to receive: %v", result.err)
				return nil, result.err
			}
			t.Logf("Received response: %+v", result.res)
			responses = append(responses, result.res)
		}
	}

	return responses, nil
}

// --- System Utilities ---

// ExtProcServerClient returns a ExternalProcessor_ProcessClient listen to localhost on given port.
func ExtProcServerClient(
	ctx context.Context,
	t *testing.T,
	port int,
	logger logr.Logger,
) (extProcPb.ExternalProcessor_ProcessClient, *grpc.ClientConn) {
	t.Helper()

	// Force IPv4 to match GetFreePort's binding and avoid IPv6 race conditions in CI.
	serverAddr := fmt.Sprintf("127.0.0.1:%d", port)

	// Wait for TCP readiness.
	// We must poll the port until the server successfully binds and listens.
	require.Eventually(t, func() bool {
		// Check if the port is open.
		conn, err := net.DialTimeout("tcp", serverAddr, 50*time.Millisecond)
		if err != nil {
			return false
		}
		conn.Close()
		return true
	}, extprocConnSetupTimeout, extPorcConnSetupPollInterval, "Server failed to bind port %s", serverAddr)

	// Connect client.
	// Blocking dial is safe because we know the port is open.
	conn, err := grpc.NewClient(serverAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err, "failed to create grpc connection")

	extProcClient, err := extProcPb.NewExternalProcessorClient(conn).Process(ctx)
	require.NoError(t, err, "failed to initialize ext_proc stream client")

	return extProcClient, conn
}
