package main

import (
	"bytes"
	"encoding/json"
	"io"
	"time"
)

// nowMS is the Driver's only clock: event timestamps and the durations
// derived from them are read through this package var so tests can pin a
// fixed value — without a clock seam the golden Pass log could never compare
// byte for byte (same injection idiom as bashToolMaxOutput and
// chatRetryAttempts). Exposed as epoch milliseconds because that is the unit
// the canonical event timestamp carries.
var nowMS = func() int64 { return time.Now().UnixMilli() }

// event is one JSONL line of the Driver's stdout: the canonical event of
// issue #19, carrying only the fields the seven non-result kinds this Driver
// emits (start, reasoning, text, tool_call, tool_result, usage, error) can fill
// — the result kind rides resultPayload below. A kind leaves the fields it does not
// fill unset, and omitempty keeps them off the wire; kind and timestamp are
// always present.
//
// Field names are snake_case per the wire contract. ToolInput is `any`
// rather than a string so `tool_input` lands on the wire as a JSON value —
// the decoded arguments object — not as an escaped string of one.
type event struct {
	Kind string `json:"kind"`
	// Role is the one contract field this Driver declares and never fills: it
	// carries the attributed pass role, which only subagents produce, so it stays
	// unset until #25 lands the attribution. Declared now so #25 does not invent a
	// wire name and TestEventWireFieldNamesAreSnakeCase already pins it.
	Role string `json:"role,omitempty"`

	MessageID  string       `json:"message_id,omitempty"`
	Timestamp  int64        `json:"timestamp"`
	Text       string       `json:"text,omitempty"`
	Model      string       `json:"model,omitempty"`
	ToolID     string       `json:"tool_id,omitempty"`
	ToolName   string       `json:"tool_name,omitempty"`
	ToolInput  any          `json:"tool_input,omitempty"`
	ToolTarget string       `json:"tool_target,omitempty"`
	ToolOutput string       `json:"tool_output,omitempty"`
	IsError    bool         `json:"is_error,omitempty"`
	Tokens     *tokenTotals `json:"tokens,omitempty"`
}

// resultPayload is the result kind's own wire type because the run summary
// never leaves a field unreported: a duration of 0 — a loopback stub answering
// inside one millisecond — is a measurement, not an absence, so no field here
// carries omitempty. The shared event struct keeps omitempty for the fields the
// other seven kinds genuinely do not fill. Field declaration order is the golden
// log's byte order.
type resultPayload struct {
	Kind          string      `json:"kind"`
	Timestamp     int64       `json:"timestamp"`
	Text          string      `json:"text"`
	Tokens        tokenTotals `json:"tokens"`
	DurationMs    int64       `json:"duration_ms"`
	DurationAPIMs int64       `json:"duration_api_ms"`
	NumTurns      int         `json:"num_turns"`
	Models        []string    `json:"models"`
}

// tokenTotals is one model call's token tally, or a run's summed tally.
// No omitempty: a usage event always states all six fields, so a decoder
// never has to guess whether a missing key means zero or "not reported".
type tokenTotals struct {
	Input        float64 `json:"input"`
	Output       float64 `json:"output"`
	CacheRead    float64 `json:"cache_read"`
	CacheWrite5m float64 `json:"cache_write_5m"`
	CacheWrite1h float64 `json:"cache_write_1h"`
	CostUSD      float64 `json:"cost_usd"`
}

// add sums two tallies field by field. The result event's tokens are the
// per-call usages added up, and every field has to participate: a cost a future
// server starts reporting would otherwise vanish from the session total.
func (t tokenTotals) add(o tokenTotals) tokenTotals {
	return tokenTotals{
		Input:        t.Input + o.Input,
		Output:       t.Output + o.Output,
		CacheRead:    t.CacheRead + o.CacheRead,
		CacheWrite5m: t.CacheWrite5m + o.CacheWrite5m,
		CacheWrite1h: t.CacheWrite1h + o.CacheWrite1h,
		CostUSD:      t.CostUSD + o.CostUSD,
	}
}

// eventWriter writes canonical events to w as JSONL: exactly one JSON object
// per emit, terminated by a single '\n' and nothing else — no prefix, no
// wrapping. stdout carries only these lines, so any stray byte would be a
// malformed event.
//
// encoding/json escapes newlines, quotes and control characters, so one
// object per line holds even when server-controlled text (tool_output, error
// text) is embedded verbatim — text is never flattened through boundBody,
// which exists only for the one-line stderr progress.
//
// HTML escaping is switched off: assistant prose and tool output keep '<',
// '>' and '&' literally on the wire, and JSON's mandatory escaping still
// covers the line-separator characters that would otherwise break a line.
type eventWriter struct{ w io.Writer }

func (ew eventWriter) emit(e event) error {
	e.Timestamp = nowMS()
	return ew.encode(e)
}

// emitResult writes the run's closing summary; it takes its own type rather
// than an event so the always-present result fields cannot be dropped by an
// omitempty meant for the kinds that leave them unset.
func (ew eventWriter) emitResult(p resultPayload) error {
	p.Timestamp = nowMS()
	return ew.encode(p)
}

func (ew eventWriter) encode(v any) error {
	var line bytes.Buffer
	enc := json.NewEncoder(&line)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return err
	}
	// Encode terminated the object with exactly one '\n', and marshalling
	// into the buffer first keeps a mid-object failure from writing half a
	// line to stdout.
	_, err := ew.w.Write(line.Bytes())
	return err
}
