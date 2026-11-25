package main

import "fmt"

// repeatedCallRefuseLimit is how many identical calls in a row the breaker lets
// run: the third gets the synthetic Tool Result in place of running. The two 3s
// count different things — calls and refusals — so each gets its own name rather
// than one shared literal tying two independent limits together.
const repeatedCallRefuseLimit = 3

// repeatedCallEndLimit caps the refusals in a row the run survives: past that
// many the refusal is clearly not being heeded, and spending turns on it only
// repeats itself.
const repeatedCallEndLimit = 3

// repeatedCallRefusal is the synthetic Tool Result a refused call gets instead of
// running. Its "error: " prefix is the one the Guard Hook refusal established as
// how a refusal reads; it names the call so the Model can tell which action to
// change.
func repeatedCallRefusal(toolName string, repeat int) string {
	return fmt.Sprintf("error: %s was already called %d times in a row with these exact arguments, so it was not run again; take a different action", toolName, repeat-1)
}

// callBreaker is a run's repeated-call state. It is a value owned by one run
// rather than package state because a package counter would carry across every
// run() in the process — including the runs a Subagent Tool call starts — while
// the streak is per run. The zero value is a run that has seen no calls, so there
// is nothing to initialise.
type callBreaker struct {
	// lastKey is the previous call as decide builds it: tool name, NUL, raw
	// arguments.
	lastKey string
	// repeat counts the current call too, so the call that first arrives is 1.
	repeat int
	// refusals is how many calls in a row decide has refused.
	refusals int
}

// breakerOutcome is what the breaker answered about one call before it ran.
type breakerOutcome int

const (
	// breakerRun is "no objection": the Tool runs. The zero value is "run" so an
	// unfilled verdict fails open towards dispatching rather than refusing a Tool
	// that did nothing wrong.
	breakerRun breakerOutcome = iota
	// breakerRefuse is a call the breaker blocked with a synthetic Tool Result
	// while the run carries on.
	breakerRefuse
	// breakerEndRun is the refusal that stopped being heeded: the run ends after
	// the refusal's Tool Result goes out.
	breakerEndRun
)

// breakerVerdict is the breaker's answer about one call: whether it runs and,
// when it does not, what the loop sends and why the run stops.
type breakerVerdict struct {
	outcome breakerOutcome
	// result is the synthetic Tool Result a refused call is answered with, and
	// the zero toolResult when the call runs. It states failed because a refusal
	// is the Driver authoring the verdict a Tool states by refusing to run.
	result toolResult
	// reason is why the run ends, and nil unless outcome is breakerEndRun.
	reason error
}

// runs reports whether the breaker objects to the call at all. Answering "run"
// is itself the reset signal: decide has already cleared the refusal streak, so
// the loop needs no second call to reset anything.
func (v breakerVerdict) runs() bool { return v.outcome == breakerRun }

// endsRun reports whether the run stops on this verdict. The loop still emits
// verdict.result first, so the ending refusal is not silently dropped.
func (v breakerVerdict) endsRun() bool { return v.outcome == breakerEndRun }

// decide takes one call as the Model sent it and answers whether it may run.
func (b *callBreaker) decide(name, rawArgs string) breakerVerdict {
	// The key is the call exactly as the Model sent it: tool name and raw wire
	// arguments joined by a NUL neither can contain. Keying on the decoded
	// arguments would follow Go map order instead of what was sent.
	key := name + "\x00" + rawArgs
	if key == b.lastKey {
		b.repeat++
	} else {
		b.lastKey, b.repeat = key, 1
	}
	if b.repeat < repeatedCallRefuseLimit {
		// Only the breaker's own refusals chain: a Guard-Hook-blocked call comes
		// back through the Tool like any other result and the Model does get to
		// react, so a call that runs restarts the count.
		b.refusals = 0
		return breakerVerdict{}
	}
	b.refusals++
	refusal := toolResult{text: repeatedCallRefusal(name, b.repeat), failed: true}
	if b.refusals >= repeatedCallEndLimit {
		return breakerVerdict{
			outcome: breakerEndRun,
			result:  refusal,
			reason:  fmt.Errorf("repeated the same %s call so often it was refused %d times in a row, so the run ended", name, b.refusals),
		}
	}
	return breakerVerdict{outcome: breakerRefuse, result: refusal}
}
