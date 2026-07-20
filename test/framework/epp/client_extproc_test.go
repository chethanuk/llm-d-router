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
	"errors"
	"fmt"
	"io"
	"net"
	"testing"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"

	fwknet "github.com/llm-d/llm-d-router/test/framework/net"
)

// echoServer is a minimal ext_proc server: it replies to every request with one
// response, which is all the client-side executors need to be exercised.
type echoServer struct {
	extProcPb.UnimplementedExternalProcessorServer
	responses int // responses to emit per request; 0 means echo one per request
}

func (s *echoServer) Process(stream extProcPb.ExternalProcessor_ProcessServer) error {
	for {
		if _, err := stream.Recv(); err != nil {
			return nil // client closed or context ended
		}
		for range max(s.responses, 1) {
			if err := stream.Send(NewResponseStreamChunk("ack", false)); err != nil {
				return err
			}
		}
	}
}

// startEchoServer serves an echoServer on a free port and returns that port.
func startEchoServer(t *testing.T, srv *echoServer) int {
	t.Helper()

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)

	grpcSrv := grpc.NewServer()
	extProcPb.RegisterExternalProcessorServer(grpcSrv, srv)
	go func() { _ = grpcSrv.Serve(lis) }()
	t.Cleanup(grpcSrv.Stop)

	return lis.Addr().(*net.TCPAddr).Port
}

func TestExtProcServerClient(t *testing.T) {
	port := startEchoServer(t, &echoServer{})

	client, conn := ExtProcServerClient(t.Context(), t, port, logr.Discard())
	t.Cleanup(func() { _ = conn.Close() })

	require.NotNil(t, client)
	require.Equal(t, fmt.Sprintf("127.0.0.1:%d", port), conn.Target(), "client must target the requested port over IPv4")

	// The returned client is a live ext_proc stream, not just a non-nil handle.
	require.NoError(t, client.Send(ReqHeaderOnly(nil)[0]))
	res, err := client.Recv()
	require.NoError(t, err)
	assert.Equal(t, []byte("ack"),
		res.GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse().GetBody())
}

func TestStreamedRequest(t *testing.T) {
	tests := []struct {
		name              string
		requests          int
		serverResponses   int
		expectedResponses int
		wantResponses     int
	}{
		{name: "one response per request", requests: 2, expectedResponses: 2, wantResponses: 2},
		{name: "server fans out extra responses", requests: 1, serverResponses: 3, expectedResponses: 3, wantResponses: 3},
		{name: "collects fewer than the server sends", requests: 2, expectedResponses: 1, wantResponses: 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			port := startEchoServer(t, &echoServer{responses: tc.serverResponses})
			client, conn := ExtProcServerClient(t.Context(), t, port, logr.Discard())
			defer conn.Close()

			reqs := make([]*extProcPb.ProcessingRequest, 0, tc.requests)
			for range tc.requests {
				reqs = append(reqs, ReqHeaderOnly(map[string]string{"x-a": "1"})...)
			}

			got, err := StreamedRequest(t, client, reqs, tc.expectedResponses)

			require.NoError(t, err)
			assert.Len(t, got, tc.wantResponses)
		})
	}
}

func TestStreamedRequestServerClosesStream(t *testing.T) {
	// The server stops after the first response, so the client sees io.EOF while
	// waiting for the second: a valid termination, not an error.
	port := startEchoServer(t, &echoServer{})
	client, conn := ExtProcServerClient(t.Context(), t, port, logr.Discard())
	defer conn.Close()

	require.NoError(t, client.CloseSend())

	got, err := StreamedRequest(t, client, nil, 1)

	require.NoError(t, err)
	assert.Empty(t, got)
}

// stubStream drives StreamedRequest's failure paths deterministically; the embedded
// nil ClientStream is never touched because only Send and Recv are called.
type stubStream struct {
	grpc.ClientStream
	sendErr  error
	recvErrs []error
}

func (s *stubStream) Send(*extProcPb.ProcessingRequest) error { return s.sendErr }

func (s *stubStream) Recv() (*extProcPb.ProcessingResponse, error) {
	if len(s.recvErrs) == 0 {
		return NewResponseStreamChunk("ack", false), nil
	}
	err := s.recvErrs[0]
	s.recvErrs = s.recvErrs[1:]
	if err != nil {
		return nil, err
	}
	return NewResponseStreamChunk("ack", false), nil
}

func TestStreamedRequestFailurePaths(t *testing.T) {
	boom := errors.New("boom")

	tests := []struct {
		name          string
		stream        *stubStream
		wantErrIs     error
		wantResponses int
	}{
		{
			name:      "send failure aborts before receiving",
			stream:    &stubStream{sendErr: boom},
			wantErrIs: boom,
		},
		{
			name:      "receive failure is returned",
			stream:    &stubStream{recvErrs: []error{boom}},
			wantErrIs: boom,
		},
		{
			// io.EOF means the server closed the stream early (e.g. it rejected the
			// request); the responses collected so far are returned without an error.
			name:          "server EOF terminates collection cleanly",
			stream:        &stubStream{recvErrs: []error{nil, io.EOF}},
			wantResponses: 1,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := StreamedRequest(t, tc.stream, ReqHeaderOnly(nil), 2)

			if tc.wantErrIs != nil {
				require.ErrorIs(t, err, tc.wantErrIs)
				assert.Nil(t, got)
				return
			}
			require.NoError(t, err)
			assert.Len(t, got, tc.wantResponses)
		})
	}
}

// TestClientWaitsForALateServer checks that ExtProcServerClient polls a closed port
// instead of failing on the first dial: the listener is bound only once the client is
// observably still waiting, so both facts are asserted rather than assumed.
func TestClientWaitsForALateServer(t *testing.T) {
	port, err := fwknet.GetFreePort()
	require.NoError(t, err)
	addr := fmt.Sprintf("127.0.0.1:%d", port)

	// Precondition for the scenario: the port must refuse connections right now.
	c, derr := net.DialTimeout("tcp", addr, extPorcConnSetupPollInterval)
	if derr == nil {
		_ = c.Close()
	}
	require.Error(t, derr, "port %d was taken before the test could bind it", port)

	// The connection comes back over the channel instead of being closed here: on the
	// timeout path the test has already returned, and a t.Cleanup registered from this
	// goroutine after that point panics.
	type dialed struct {
		client extProcPb.ExternalProcessor_ProcessClient
		conn   *grpc.ClientConn
	}
	done := make(chan dialed, 1)
	go func() {
		client, conn := ExtProcServerClient(t.Context(), t, port, logr.Discard())
		done <- dialed{client: client, conn: conn}
	}()

	require.Never(t, func() bool {
		select {
		case <-done:
			return true
		default:
			return false
		}
	}, 10*extPorcConnSetupPollInterval, extPorcConnSetupPollInterval,
		"client resolved while the port was still closed")

	lis, err := net.Listen("tcp", addr)
	require.NoError(t, err)
	defer lis.Close()

	grpcSrv := grpc.NewServer()
	extProcPb.RegisterExternalProcessorServer(grpcSrv, &echoServer{})
	defer grpcSrv.Stop()
	go func() { _ = grpcSrv.Serve(lis) }()

	select {
	case got := <-done:
		defer got.conn.Close()
		require.NotNil(t, got.client)
	case <-time.After(extprocConnSetupTimeout):
		t.Fatal("client did not observe the late bind")
	}
}
