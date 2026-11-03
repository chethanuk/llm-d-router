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

package parserutil

import (
	"encoding/binary"
	"errors"
	"fmt"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
)

// ParseGrpcPayload extracts the message payload from a gRPC frame.
// A standard gRPC frame consists of a 1-byte compression flag, a 4-byte message length,
// and the actual message payload.
// It returns an error if the payload is compressed.
func ParseGrpcPayload(data []byte) ([]byte, error) {
	if len(data) < fwkrequest.GRPCFrameHeaderLen {
		return nil, fmt.Errorf("invalid gRPC frame: expected at least %d bytes for header, got %d", fwkrequest.GRPCFrameHeaderLen, len(data))
	}

	isCompressed := data[0] == 1
	if isCompressed {
		// TODO(#895): handle compressed payload.
		return nil, errors.New("compressed gRPC payload is not supported")
	}
	msgLen := binary.BigEndian.Uint32(data[1:5])

	// Compare in uint64 so fwkrequest.GRPCFrameHeaderLen+msgLen cannot overflow when
	// msgLen is near math.MaxUint32, which would wrap the check and let the
	// slice below panic with an out-of-range low>high index.
	if uint64(len(data)) < uint64(fwkrequest.GRPCFrameHeaderLen)+uint64(msgLen) {
		return nil, fmt.Errorf("incomplete gRPC payload: header indicates %d bytes, but only %d bytes are available", msgLen, len(data)-fwkrequest.GRPCFrameHeaderLen)
	}
	return data[fwkrequest.GRPCFrameHeaderLen : fwkrequest.GRPCFrameHeaderLen+msgLen], nil
}

// ParseGrpcPayloads extracts the message payload of every gRPC frame in data.
// A streamed response can deliver several frames in one body chunk. When a
// frame cannot be parsed, the payloads of the frames before it are returned
// together with the error, so a caller can still read them.
func ParseGrpcPayloads(data []byte) ([][]byte, error) {
	var payloads [][]byte
	for len(data) > 0 {
		payload, err := ParseGrpcPayload(data)
		if err != nil {
			return payloads, err
		}
		payloads = append(payloads, payload)
		data = data[fwkrequest.GRPCFrameHeaderLen+len(payload):]
	}
	return payloads, nil
}
