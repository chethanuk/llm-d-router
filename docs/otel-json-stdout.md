# OpenTelemetry JSON stdout logs

The EPP, coordinator, and routing sidecar emit one JSON object per line to
stdout using the OpenTelemetry Logs Data Model fields below. The output is not
OTLP/JSON. A log collector must apply the mapping before exporting OTLP.

## Record contract

| JSON field | Type | OpenTelemetry field |
| --- | --- | --- |
| `timestamp` | RFC 3339 UTC string | `Timestamp` |
| `severity_text` | string | `SeverityText` |
| `severity_number` | integer | `SeverityNumber` |
| `body` | string | `Body` |
| `service.name` | string | `Resource.Attributes["service.name"]` |
| `logger` | string | `InstrumentationScope.Name` |
| `trace_id` | optional 32-character lowercase hex string | `TraceId` |
| `span_id` | optional 16-character lowercase hex string | `SpanId` |
| all other fields | any JSON value | `Attributes` |

Collectors must parse `timestamp`, move `service.name` to the resource, move
`logger` to the instrumentation scope, decode valid trace and span identifiers,
and retain every other application field as a log attribute.

Severity numbers use the OpenTelemetry bands: `DEBUG=5`, `INFO=9`, `WARN=13`,
`ERROR=17`, and `FATAL=21`. Negative Zap levels used by logr verbosity map to
`DEBUG/5`. Zap `DPANIC` and `PANIC` retain their defined intermediate values,
18 and 19.

Example:

```json
{"timestamp":"2026-09-11T11:34:56.123Z","severity_text":"INFO","severity_number":9,"body":"request complete","logger":"epp","service.name":"llm-d-router","trace_id":"01234567890123456789012345678901","span_id":"0123456789012345"}
```

Trace fields are omitted when no valid span is active.

## Compatibility

Releases before v0.11.0 wrote `level` and `msg`. From v0.11.0 the fields
are `severity_text`, `severity_number` and `body`, and verbosity maps as
follows (Zap level `-n` is logr `V(n)`):

| logr verbosity | `level` before v0.11.0 | `severity_text` / `severity_number` now |
| --- | --- | --- |
| `V(0)` | `info` | `INFO` / 9 |
| `V(1)` to `V(3)`, including the default `V(2)` | `info` | `DEBUG` / 5 |
| `V(4)` | `debug` | `DEBUG` / 5 |
| `V(5)` and above | `trace` | `DEBUG` / 5 |

A collector filter or alert on `severity_text == "INFO"` or
`severity_number >= 9` therefore no longer matches `V(1)` to `V(3)`
records. Use `severity_number >= 5` to keep them. Warnings and errors are
unchanged.
