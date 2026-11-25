package main

import "testing"

// The repeated-call breaker's verdicts, tested at the struct level rather than
// through a run: decide takes the two strings a tool call already carries and
// answers without a Model call, a child process or a scripted server, and the
// rows that matter — how a streak counts, what a refusal says, when the run ends
// — are far cheaper to spell as calls here than to script a server into.

func TestCallBreakerLetsTheFirstTwoIdenticalCallsRun(t *testing.T) {
	var b callBreaker
	for _, row := range []struct {
		name string
		args string
	}{
		{"first call", `{"command":"pwd"}`},
		{"second identical call", `{"command":"pwd"}`},
	} {
		t.Run(row.name, func(t *testing.T) {
			v := b.decide("bash", row.args)
			if !v.runs() {
				t.Errorf("decide() = %+v, want the call to run: only the third identical call in a row is refused", v)
			}
			// A call the breaker lets through authors no Tool Result: the Tool's own
			// result is the only one that call gets.
			if v.result != (toolResult{}) {
				t.Errorf("verdict carries result %+v, want the zero toolResult for a call that runs", v.result)
			}
		})
	}
}

// TestCallBreakerRefusesTheThirdIdenticalCall pins the refusal itself: the third
// call in a row is answered without running, and the wording is a wire contract
// the Model reads to change course, so it is compared byte for byte — including
// the "already called 2 times" count, which names the calls that did run rather
// than the one being refused.
func TestCallBreakerRefusesTheThirdIdenticalCall(t *testing.T) {
	var b callBreaker
	b.decide("bash", `{"command":"pwd"}`)
	b.decide("bash", `{"command":"pwd"}`)
	v := b.decide("bash", `{"command":"pwd"}`)

	if v.runs() {
		t.Fatalf("third identical call verdict = %+v, want a refusal", v)
	}
	want := "error: bash was already called 2 times in a row with these exact arguments, so it was not run again; take a different action"
	if v.result.text != want {
		t.Errorf("refusal text = %q, want %q", v.result.text, want)
	}
	// The refusal failed because the call the Model asked for did not happen: the
	// Tool Result carrying that verdict is what stamps is_error on the event.
	if !v.result.failed {
		t.Error("refusal result failed = false, want a refusal to state its own failure")
	}
	if v.endsRun() {
		t.Errorf("first refusal ends the run, want the run to continue and give the Model the refusal: %v", v.reason)
	}
	if v.reason != nil {
		t.Errorf("reason = %v, want none for a refusal the run continues past", v.reason)
	}
}

// TestCallBreakerRefusalCountsEveryCallOfTheStreak: a refusal keeps counting as
// the streak grows, so the Model is told how often it has already repeated the
// call instead of reading the same "2 times" every time. Each row runs the whole
// streak and reads only the last verdict, which is what a refused call is.
func TestCallBreakerRefusalCountsEveryCallOfTheStreak(t *testing.T) {
	for _, row := range []struct {
		name  string
		calls int
		text  string
	}{
		{
			name:  "fourth identical call",
			calls: 4,
			text:  "error: bash was already called 3 times in a row with these exact arguments, so it was not run again; take a different action",
		},
		{
			name:  "fifth identical call",
			calls: 5,
			text:  "error: bash was already called 4 times in a row with these exact arguments, so it was not run again; take a different action",
		},
		{
			name:  "sixth identical call",
			calls: 6,
			text:  "error: bash was already called 5 times in a row with these exact arguments, so it was not run again; take a different action",
		},
	} {
		t.Run(row.name, func(t *testing.T) {
			var b callBreaker
			var v breakerVerdict
			for i := 0; i < row.calls; i++ {
				v = b.decide("bash", `{"command":"pwd"}`)
			}
			if v.runs() {
				t.Fatalf("after %d identical calls the breaker answered %+v, want a refusal", row.calls, v)
			}
			if v.result.text != row.text {
				t.Errorf("refusal text = %q, want %q", v.result.text, row.text)
			}
			if !v.result.failed {
				t.Error("refusal result failed = false, want a refusal to state its own failure")
			}
		})
	}
}

// TestCallBreakerDifferingCallResetsTheCount is what keeps the breaker off a
// Model that is making progress: only the exact same call repeated counts, so
// arguments that differ — or a different tool given the same arguments — start a
// streak of their own instead of pushing an older one over the limit.
func TestCallBreakerDifferingCallResetsTheCount(t *testing.T) {
	for _, row := range []struct {
		name  string
		calls []struct{ tool, args string }
	}{
		{
			name: "two, then two more with one different argument set between them",
			calls: []struct{ tool, args string }{
				{"bash", `{"command":"pwd"}`},
				{"bash", `{"command":"pwd"}`},
				{"bash", `{"command":"ls"}`},
				{"bash", `{"command":"pwd"}`},
				{"bash", `{"command":"pwd"}`},
			},
		},
		{
			name: "same arguments handed to two tools interleaved",
			calls: []struct{ tool, args string }{
				{"bash", `{"command":"pwd"}`},
				{"read_file", `{"command":"pwd"}`},
				{"bash", `{"command":"pwd"}`},
				{"read_file", `{"command":"pwd"}`},
				{"bash", `{"command":"pwd"}`},
				{"read_file", `{"command":"pwd"}`},
			},
		},
		{
			name: "argument order is part of the key, not the decoded object",
			calls: []struct{ tool, args string }{
				{"write_file", `{"path":"a","content":"x"}`},
				{"write_file", `{"content":"x","path":"a"}`},
				{"write_file", `{"path":"a","content":"x"}`},
				{"write_file", `{"content":"x","path":"a"}`},
				{"write_file", `{"path":"a","content":"x"}`},
				{"write_file", `{"content":"x","path":"a"}`},
			},
		},
	} {
		t.Run(row.name, func(t *testing.T) {
			var b callBreaker
			for i, call := range row.calls {
				if v := b.decide(call.tool, call.args); !v.runs() {
					t.Fatalf("call %d (%s %s) answered %+v, want every call to run: no call here is the third identical one in a row", i+1, call.tool, call.args, v)
				}
			}
		})
	}
}

// TestCallBreakerEndsTheRunOnTheThirdRefusal is the breaker's last resort: two
// refusals are a Model that might still change course, a third one in a row is a
// loop spending turns on itself. The ending verdict still carries the refusal,
// because the loop has to emit that Tool Result before it stops — the Model is
// owed the answer to the call it made even as the run closes.
func TestCallBreakerEndsTheRunOnTheThirdRefusal(t *testing.T) {
	for _, row := range []struct {
		name   string
		calls  int
		ends   bool
		reason string
	}{
		{name: "first refusal", calls: 3},
		{name: "second refusal", calls: 4},
		{
			name:   "third refusal in a row",
			calls:  5,
			ends:   true,
			reason: "repeated the same bash call so often it was refused 3 times in a row, so the run ended",
		},
	} {
		t.Run(row.name, func(t *testing.T) {
			var b callBreaker
			var v breakerVerdict
			for i := 0; i < row.calls; i++ {
				v = b.decide("bash", `{"command":"pwd"}`)
			}
			if v.endsRun() != row.ends {
				t.Errorf("endsRun() = %v, want %v (verdict %+v)", v.endsRun(), row.ends, v)
			}
			reason := ""
			if v.reason != nil {
				reason = v.reason.Error()
			}
			if reason != row.reason {
				t.Errorf("reason = %q, want %q", reason, row.reason)
			}
			if row.ends {
				want := "error: bash was already called 4 times in a row with these exact arguments, so it was not run again; take a different action"
				if v.result.text != want {
					t.Errorf("ending verdict carries result %q, want the refusal %q so the loop can still emit it", v.result.text, want)
				}
				if !v.result.failed {
					t.Error("ending verdict's result failed = false, want the refusal to state its own failure")
				}
			}
		})
	}
}

// verdictWord names a verdict by what the loop would do with it, so the rows of
// the streak test below read as behaviour rather than as enum constants.
func verdictWord(v breakerVerdict) string {
	switch {
	case v.endsRun():
		return "end"
	case !v.runs():
		return "refuse"
	default:
		return "run"
	}
}

// TestCallBreakerDispatchedCallClearsTheRefusalStreak pins the reset: the count
// that ends a run counts refusals in a row, so one call the breaker let through
// means the Model changed course and every refusal before it is water under the
// bridge. Without that reset, two refusals early in a run plus two later would
// end a run that was making progress.
func TestCallBreakerDispatchedCallClearsTheRefusalStreak(t *testing.T) {
	pwd := `{"command":"pwd"}`
	ls := `{"command":"ls"}`
	streak := []struct {
		name string
		tool string
		args string
		want string
	}{
		{"the call", "bash", pwd, "run"},
		{"its repeat", "bash", pwd, "run"},
		{"first refusal", "bash", pwd, "refuse"},
		{"second refusal", "bash", pwd, "refuse"},
		{"a different call the breaker dispatches", "bash", ls, "run"},
		{"the original call again", "bash", pwd, "run"},
		{"and again", "bash", pwd, "run"},
		{"refusal one of the new streak", "bash", pwd, "refuse"},
		{"refusal two of the new streak", "bash", pwd, "refuse"},
		{"refusal three in a row ends the run", "bash", pwd, "end"},
	}
	// Each row replays the calls before it so a single row can run on its own; a
	// shared breaker would make -run on one row answer something else.
	for i, row := range streak {
		t.Run(row.name, func(t *testing.T) {
			var b callBreaker
			var v breakerVerdict
			for _, call := range streak[:i+1] {
				v = b.decide(call.tool, call.args)
			}
			if got := verdictWord(v); got != row.want {
				t.Errorf("call %d (%s %s) answered %q, want %q", i+1, row.tool, row.args, got, row.want)
			}
		})
	}

	// The reason counts the refusals that ended the run, not every refusal the run
	// ever handed out: a reader of stderr has to be able to trust the number.
	var b callBreaker
	var v breakerVerdict
	for _, call := range streak {
		v = b.decide(call.tool, call.args)
	}
	want := "repeated the same bash call so often it was refused 3 times in a row, so the run ended"
	if v.reason == nil || v.reason.Error() != want {
		t.Errorf("reason = %v, want %q — the five refusals of the run were not 3 in a row", v.reason, want)
	}
}
