package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

// modelEntry is one scripted answer a scriptedModelServer holds for a model id.
//
// An entry is spent per POST rather than per logical turn, so a retry eats the
// next entry in the queue exactly as a real endpoint would bill it a second
// completion.
type modelEntry struct {
	// frames is the whole SSE body to write, so the test owns the chunk
	// boundaries: split answers, split tool-call arguments, a lone usage chunk.
	frames string
	// completion is a whole non-streaming completion body, framed on the way out
	// by sseChatCompletion. It is read only when frames is empty.
	completion string
	// status marks the entry a rejection rather than a reply: a non-zero status is
	// what the endpoint answers with, and it is the only discriminator an entry
	// needs because a rejection carries no frames and a reply carries no status.
	status int
	// headers are set on the response before status is written, the way a real
	// endpoint sets Retry-After before its 429.
	headers map[string]string
	// body is the rejection body, written verbatim.
	body string
	// dropAfter marks the entry a connection dropped mid-body: it holds the raw
	// frames that go out first, after which the connection is torn down under the
	// client the way a proxy dying mid-stream does. Non-empty wins over frames, and
	// such an entry carries no status because the reply started successfully.
	dropAfter string
}

// replyFrames scripts one answer as raw SSE frames.
func replyFrames(frames string) modelEntry { return modelEntry{frames: frames} }

// replyCompletion scripts one answer as a whole completion body, served as the
// streaming frames the same answer arrives in.
func replyCompletion(body string) modelEntry { return modelEntry{completion: body} }

// dropAfterFrames scripts one answer as raw frames that reach the client before
// the connection is dropped, so the stream is truncated rather than completed.
func dropAfterFrames(frames string) modelEntry { return modelEntry{dropAfter: frames} }

// reject scripts the endpoint refusing one request: the status, the refusal
// headers and the error body, none of them reshaped.
func reject(status int, headers map[string]string, body string) modelEntry {
	return modelEntry{status: status, headers: headers, body: body}
}

// scriptedModelServer is the Model a run talks to over the wire, scripted one
// answer per model id.
//
// Answers route on the model field of the request body rather than on the POST
// index: a spawn inherits its parent's endpoint through the environment (see
// subagentEnv), so parent and children share one URL and the model they ask for
// is the only thing that tells their requests apart.
type scriptedModelServer struct {
	mu     sync.Mutex
	queues map[string][]modelEntry
	reqs   []recordedRequest
	// missed names, in arrival order, the models a POST asked for that no queue
	// could answer — exhausted or never scripted. The handler only records them:
	// it runs on the server's goroutines, where t.Fatalf would panic, so the test
	// that owns the server reads this and fails itself.
	missed []string
	srv    *httptest.Server
}

func newScriptedModelServer(t *testing.T, queues map[string][]modelEntry) *scriptedModelServer {
	t.Helper()
	s := &scriptedModelServer{queues: queues}
	s.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		// The handler runs on the server's goroutines, so recording and spending
		// share one critical section with requestBodies.
		s.mu.Lock()
		s.reqs = append(s.reqs, recordedRequest{
			method: r.Method,
			path:   r.URL.Path,
			header: r.Header.Clone(),
			at:     time.Now(),
			body:   body,
		})
		var req struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(body, &req)
		queue := s.queues[req.Model]
		var entry modelEntry
		// A queue that is empty and a model never scripted are the same mistake: an
		// answer nobody wrote. Serving the zero-value entry instead would hand the run
		// an empty SSE stream, which reads as a finished reply that said nothing.
		missed := len(queue) == 0
		if missed {
			s.missed = append(s.missed, req.Model)
		} else {
			entry = queue[0]
			s.queues[req.Model] = queue[1:]
		}
		s.mu.Unlock()

		switch {
		case missed:
			// http.Error rather than a hand-written body: unlike a scripted rejection,
			// nobody parses this one, so the trailing newline costs nothing next to the
			// status line and Content-Type a failing run gets for free.
			http.Error(w, "no scripted response left for model "+req.Model, http.StatusInternalServerError)
		case entry.status != 0:
			for k, v := range entry.headers {
				w.Header().Set(k, v)
			}
			w.WriteHeader(entry.status)
			// Written out by hand rather than through http.Error: that helper appends a
			// newline, and the rejection body reaches the error event verbatim.
			io.WriteString(w, entry.body)
		case entry.dropAfter != "":
			// Through writeSSE rather than a bare Write so the frames are flushed and
			// therefore provably with the client before the tear down.
			writeSSE(w, entry.dropAfter)
			// This runs on the server's goroutine, so t.Errorf is all a failure here can
			// report through; t.Fatalf would panic outside the test.
			conn, _, err := w.(http.Hijacker).Hijack()
			if err != nil {
				t.Errorf("scripted drop cannot hijack the connection to close it: %v", err)
				return
			}
			conn.Close()
		case entry.frames == "" && entry.completion == "":
			// No constructor builds an entry with neither field, so reaching this means a
			// new entry kind forgot its discriminator in the switch above.
			http.Error(w, "model "+req.Model+" has a scripted entry with no reply in it", http.StatusInternalServerError)
		default:
			writeSSE(w, entry.sseBody())
		}
	}))
	t.Cleanup(s.srv.Close)
	return s
}

// sseBody renders the entry into the SSE body to write.
func (e modelEntry) sseBody() string {
	if e.completion != "" {
		return sseChatCompletion(e.completion)
	}
	return e.frames
}

func (s *scriptedModelServer) url() string { return s.srv.URL }

// recordedRequest is one POST as it arrived on the wire, body and all: the facts a
// test asserts about how it asked, not only what it asked.
//
// header is a clone taken under the handler's hands. Keeping r.Header itself would
// hand the record a map the net/http read loop owns and may still be writing, so
// an absence checked after the run finished would be a verdict on nothing.
type recordedRequest struct {
	method string
	path   string
	header http.Header
	at     time.Time
	body   []byte
}

// requestBodies returns every recorded request body in arrival order.
func (s *scriptedModelServer) requestBodies(t *testing.T) [][]byte {
	t.Helper()
	recs := s.recordedRequests(t)
	out := make([][]byte, len(recs))
	for i, rec := range recs {
		out[i] = rec.body
	}
	return out
}

// recordedRequests returns every POST taken, in arrival order, with its wire facts
// intact.
func (s *scriptedModelServer) recordedRequests(t *testing.T) []recordedRequest {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.reqs)
}

// gaps returns the wait between consecutive POST arrivals, one entry per pair, so
// len(gaps) counts waits rather than requests. The timestamps are taken in the same
// critical section that spends the queue, so a gap is the wait between two entries
// being handed out and cannot straddle an unrecorded request.
func (s *scriptedModelServer) gaps() []time.Duration {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]time.Duration, 0, len(s.reqs))
	for i := 1; i < len(s.reqs); i++ {
		out = append(out, s.reqs[i].at.Sub(s.reqs[i-1].at))
	}
	return out
}

// unscriptedModels returns the model ids that got a POST no queue could answer,
// in arrival order: exhausted queues and models never scripted alike.
func (s *scriptedModelServer) unscriptedModels() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.missed)
}

// checkNoUnscriptedRequests fails the calling test for every POST the queues could
// not answer. A run-driven test cannot assert at the moment of the miss — the miss
// happens on a server goroutine mid-run — so it calls this once the run is over.
func (s *scriptedModelServer) checkNoUnscriptedRequests(t *testing.T) {
	t.Helper()
	for _, model := range s.unscriptedModels() {
		t.Errorf("model %q was asked for with no scripted response left", model)
	}
}

// A reply entry written as raw frames reaches the caller as exactly those bytes
// under the event stream content type: the scripted server leaves through the
// same writeSSE the existing fakes use, so what a test asserts about a stream is
// what the wire carried.
func TestScriptedModelServerReplyFramesAnswersExactBytesAndRecordsBody(t *testing.T) {
	const frames = "data: {\"model\":\"m-a\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n" + sseDone

	s := newScriptedModelServer(t, map[string][]modelEntry{
		"m-a": {replyFrames(frames)},
	})

	status, contentType, body := postModel(t, s.url()+"/chat/completions", `{"model":"m-a","messages":[{"role":"user","content":"go"}]}`)
	if status != http.StatusOK {
		t.Errorf("status = %d, want 200", status)
	}
	if contentType != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", contentType)
	}
	if body != frames {
		t.Errorf("body = %q, want exactly the scripted frames %q", body, frames)
	}

	bodies := s.requestBodies(t)
	if len(bodies) != 1 {
		t.Fatalf("recorded %d request bodies, want 1", len(bodies))
	}
	if got := string(bodies[0]); got != `{"model":"m-a","messages":[{"role":"user","content":"go"}]}` {
		t.Errorf("recorded body = %q, want the raw request as sent", got)
	}
}

// postModel sends one chat completion request over the wire and reports what came
// back: these tests pin the endpoint contract itself, so they post rather than
// drive a run.
func postModel(t *testing.T, url, body string) (int, string, string) {
	t.Helper()
	status, header, raw := postModelResponse(t, url, body)
	return status, header.Get("Content-Type"), raw
}

// postModelResponse is postModel with the whole response header map kept: a
// rejection carries its answer in the status and the headers as much as in the
// body.
func postModelResponse(t *testing.T, url, body string) (int, http.Header, string) {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read response body: %v", err)
	}
	return resp.StatusCode, resp.Header, string(raw)
}

// A rejection entry answers with the status, the headers and the body it was
// scripted with — the shape a rate-limited or over-quota endpoint replies in.
//
// The body is compared byte for byte because a rejection body reaches the error
// event verbatim: serving it through http.Error would append a newline and every
// consumer's parse of the error JSON would be looking at what this test asserts.
func TestScriptedModelServerRejectAnswersWithStatusHeadersAndExactBody(t *testing.T) {
	const rejectBody = `{"error":{"message":"Rate limit reached. Please try again in 7s.","type":"rate_limit_error","code":"rate_limit_exceeded"}}`

	s := newScriptedModelServer(t, map[string][]modelEntry{
		"m-a": {reject(http.StatusTooManyRequests, map[string]string{"Retry-After": "7"}, rejectBody)},
	})

	status, header, body := postModelResponse(t, s.url()+"/chat/completions", `{"model":"m-a","messages":[]}`)
	if status != http.StatusTooManyRequests {
		t.Errorf("status = %d, want 429", status)
	}
	if got := header.Get("Retry-After"); got != "7" {
		t.Errorf("Retry-After = %q, want %q", got, "7")
	}
	if body != rejectBody {
		t.Errorf("body = %q, want exactly the scripted rejection body %q", body, rejectBody)
	}
}

// requestBodies reports every POST the server took, in the order they arrived,
// not just the one that answered: a test asserting how many turns a run spent
// needs the whole conversation the endpoint saw, retries included.
func TestScriptedModelServerRequestBodiesKeepsEveryPostInOrder(t *testing.T) {
	s := newScriptedModelServer(t, map[string][]modelEntry{
		"m-a": {replyFrames(sseStream(textChunk("first"))), replyFrames(sseStream(textChunk("second")))},
	})

	first := `{"model":"m-a","messages":[{"role":"user","content":"one"}]}`
	second := `{"model":"m-a","messages":[{"role":"user","content":"two"}]}`
	postModel(t, s.url()+"/chat/completions", first)
	postModel(t, s.url()+"/chat/completions", second)

	bodies := s.requestBodies(t)
	if len(bodies) != 2 {
		t.Fatalf("recorded %d request bodies, want 2", len(bodies))
	}
	for i, want := range []string{first, second} {
		if got := string(bodies[i]); got != want {
			t.Errorf("body %d = %q, want %q", i, got, want)
		}
	}
}

// The headers are cloned rather than kept by reference: `_, ok :=
// header["Authorization"]` is only a verdict on a request if nothing hands the
// map out again after the handler returns.
func TestScriptedModelServerRecordsMethodPathAndHeadersOfEachPost(t *testing.T) {
	s := newScriptedModelServer(t, map[string][]modelEntry{
		"m-a": {replyFrames(sseStream(textChunk("one"))), replyFrames(sseStream(textChunk("two")))},
	})

	postModel(t, s.url()+"/chat/completions", `{"model":"m-a","messages":[]}`)
	postModelWithBearer(t, s.url()+"/chat/completions", "secret-token", `{"model":"m-a","messages":[]}`)

	recs := s.recordedRequests(t)
	if len(recs) != 2 {
		t.Fatalf("recorded %d requests, want 2", len(recs))
	}
	for i, rec := range recs {
		if rec.method != http.MethodPost {
			t.Errorf("request %d method = %q, want POST", i, rec.method)
		}
		if rec.path != "/chat/completions" {
			t.Errorf("request %d path = %q, want /chat/completions", i, rec.path)
		}
		if got := rec.header.Get("Content-Type"); got != "application/json" {
			t.Errorf("request %d Content-Type = %q, want application/json", i, got)
		}
		if !json.Valid(rec.body) {
			t.Errorf("request %d body = %q, want the raw request body alongside the wire facts", i, rec.body)
		}
	}
	if _, ok := recs[0].header["Authorization"]; ok {
		t.Errorf("request 0 carries Authorization %q, want the header absent from the record entirely", recs[0].header.Get("Authorization"))
	}
	vals, ok := recs[1].header["Authorization"]
	if !ok || len(vals) != 1 || vals[0] != "Bearer secret-token" {
		t.Errorf("request 1 Authorization = %v (present %v), want the one value %q", vals, ok, "Bearer secret-token")
	}
}

// postModelWithBearer posts with the Authorization header the Driver sends its
// requests under, so a test can tell an absent header in the record from a
// present one.
func postModelWithBearer(t *testing.T, url, bearer, body string) {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, url, strings.NewReader(body))
	if err != nil {
		t.Fatalf("build POST %s: %v", url, err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+bearer)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	defer resp.Body.Close()
	if _, err := io.Copy(io.Discard, resp.Body); err != nil {
		t.Fatalf("read response body: %v", err)
	}
}

// gaps reports one duration per pair of consecutive arrivals, so a backoff test
// counts waits rather than requests — which is also why a lone POST reports no
// gap at all instead of a zero one.
func TestScriptedModelServerGapsReportsWaitBetweenArrivals(t *testing.T) {
	s := newScriptedModelServer(t, map[string][]modelEntry{
		"m-a": {replyFrames(sseStream(textChunk("one"))), replyFrames(sseStream(textChunk("two")))},
	})

	postModel(t, s.url()+"/chat/completions", `{"model":"m-a","messages":[]}`)
	if got := s.gaps(); len(got) != 0 {
		t.Fatalf("gaps after one POST = %v, want none", got)
	}

	// The bound sits well under what is slept on purpose: this pins that an
	// arrival gap is measured at all, not how quickly this machine sleeps.
	time.Sleep(20 * time.Millisecond)
	postModel(t, s.url()+"/chat/completions", `{"model":"m-a","messages":[]}`)

	got := s.gaps()
	if len(got) != 1 {
		t.Fatalf("gaps after two POSTs = %d, want 1: %v", len(got), got)
	}
	if got[0] < 10*time.Millisecond {
		t.Errorf("gap = %v, want at least the 10ms floor below the 20ms slept", got[0])
	}
}

// A queue mixes kinds freely: a rejection spends the entry in front of it exactly
// like a reply does, so a run that retries a 429 gets the next entry on its second
// POST rather than the same refusal again (or the same reply twice).
func TestScriptedModelServerSpendsEntriesPerPostAcrossKinds(t *testing.T) {
	const refusal = `{"error":{"message":"slow down","type":"rate_limit_error"}}`

	s := newScriptedModelServer(t, map[string][]modelEntry{
		"m-a": {
			reject(http.StatusTooManyRequests, map[string]string{"Retry-After": "3"}, refusal),
			replyFrames(sseStream(textChunk("after the retry"))),
		},
	})

	status, header, body := postModelResponse(t, s.url()+"/chat/completions", `{"model":"m-a","messages":[]}`)
	if status != http.StatusTooManyRequests {
		t.Fatalf("first POST status = %d, want 429", status)
	}
	if got := header.Get("Retry-After"); got != "3" {
		t.Errorf("first POST Retry-After = %q, want %q", got, "3")
	}
	if body != refusal {
		t.Errorf("first POST body = %q, want %q", body, refusal)
	}

	status, contentType, body := postModel(t, s.url()+"/chat/completions", `{"model":"m-a","messages":[]}`)
	if status != http.StatusOK {
		t.Fatalf("second POST status = %d, want 200", status)
	}
	if contentType != "text/event-stream" {
		t.Errorf("second POST Content-Type = %q, want text/event-stream", contentType)
	}
	if got := streamedText(body); got != "after the retry" {
		t.Errorf("second POST answered %q, want %q", got, "after the retry")
	}

	if n := len(s.requestBodies(t)); n != 2 {
		t.Errorf("recorded %d request bodies, want 2", n)
	}
}

// Two model ids answered from their own FIFOs, interleaved in the order the POSTs
// arrived: m-a's second entry waits behind m-a's first while m-b answers its own
// request from its own queue. A POST-index server hands m-b whatever turn count it
// happens to be on, which is why routing is by model id (see scriptedModelServer).
func TestScriptedModelServerRoutesEachModelToItsOwnQueue(t *testing.T) {
	s := newScriptedModelServer(t, map[string][]modelEntry{
		"m-a": {replyFrames(sseStream(textChunk("a-one"))), replyFrames(sseStream(textChunk("a-two")))},
		"m-b": {replyFrames(sseStream(textChunk("b-one")))},
	})

	for i, step := range []struct{ model, want string }{
		{"m-a", "a-one"},
		{"m-b", "b-one"},
		{"m-a", "a-two"},
	} {
		_, _, body := postModel(t, s.url()+"/chat/completions", `{"model":"`+step.model+`","messages":[]}`)
		text, ok := streamedContent(body)
		if !ok {
			t.Fatalf("POST %d to %s carries no text delta: %q", i, step.model, body)
		}
		if text != step.want {
			t.Errorf("POST %d to %s answered %q, want %q", i, step.model, text, step.want)
		}
	}
}

// A rejection spends one entry on the refused model's queue and leaves every other
// queue where it was: scripting "rate-limited on one model, answered on the other"
// must not renumber the queue that keeps answering.
func TestScriptedModelServerRejectionSpendsOnlyItsOwnModelsQueue(t *testing.T) {
	s := newScriptedModelServer(t, map[string][]modelEntry{
		"m-refused": {reject(http.StatusTooManyRequests, map[string]string{"Retry-After": "1"}, `{"error":{"message":"no quota"}}`)},
		"m-ok":      {replyFrames(sseStream(textChunk("ok-one"))), replyFrames(sseStream(textChunk("ok-two")))},
	})

	status, _, _ := postModelResponse(t, s.url()+"/chat/completions", `{"model":"m-refused","messages":[]}`)
	if status != http.StatusTooManyRequests {
		t.Fatalf("refused model status = %d, want 429", status)
	}

	for i, want := range []string{"ok-one", "ok-two"} {
		status, contentType, body := postModel(t, s.url()+"/chat/completions", `{"model":"m-ok","messages":[]}`)
		if status != http.StatusOK {
			t.Fatalf("m-ok POST %d status = %d, want 200", i, status)
		}
		if contentType != "text/event-stream" {
			t.Errorf("m-ok POST %d Content-Type = %q, want text/event-stream", i, contentType)
		}
		if got := streamedText(body); got != want {
			t.Errorf("m-ok POST %d answered %q, want %q", i, got, want)
		}
	}

	// The refused model's own queue is the one that drained, so its second POST no
	// longer carries the scripted refusal.
	if status, _, body := postModelResponse(t, s.url()+"/chat/completions", `{"model":"m-refused","messages":[]}`); status == http.StatusTooManyRequests {
		t.Errorf("second POST to m-refused still answered the spent rejection: %d %q", status, body)
	}
}

// textChunk scripts one streamed delta carrying content, the chunk shape a reply
// entry streams a plain answer in.
func textChunk(content string) string {
	return fmt.Sprintf(`{"choices":[{"index":0,"delta":{"content":%q}}]}`, content)
}

// streamedContent pulls the answer out of the SSE frames a reply entry sent, so a
// routing test can name which entry answered. A payload that does not parse is
// skipped, not reported: the bool says "no delta readable", not "every frame
// parsed", so this cannot tell a malformed frame from an empty stream.
func streamedContent(frames string) (string, bool) {
	for _, frame := range strings.Split(frames, "\n\n") {
		payload, ok := strings.CutPrefix(frame, "data: ")
		if !ok || payload == "[DONE]" {
			continue
		}
		var chunk sseChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}
		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				return choice.Delta.Content, true
			}
		}
	}
	return "", false
}

// A completion entry is written as the whole non-streaming body an endpoint would
// answer a stream:false request with, and goes out framed: one entry can script a
// non-streaming-shaped answer without the test hand-assembling its chunks.
func TestScriptedModelServerReplyCompletionAnswersWithFramedBody(t *testing.T) {
	const whole = `{"model":"m-a","choices":[{"index":0,"message":{"role":"assistant","content":"whole answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}`

	s := newScriptedModelServer(t, map[string][]modelEntry{
		"m-a": {replyCompletion(whole)},
	})

	status, contentType, body := postModel(t, s.url()+"/chat/completions", `{"model":"m-a","messages":[]}`)
	if status != http.StatusOK {
		t.Errorf("status = %d, want 200", status)
	}
	if contentType != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", contentType)
	}
	if want := sseChatCompletion(whole); body != want {
		t.Errorf("body = %q, want the frames sseChatCompletion builds: %q", body, want)
	}
	if !strings.Contains(body, "whole answer") || !strings.Contains(body, "prompt_tokens") {
		t.Errorf("body = %q, want the message and the usage each in their own chunk", body)
	}
}

// The one-queue case: a run with a single model id is served straight down its
// queue, entry after entry, which is the whole contract the per-model routing
// leaves visible once there is nothing to route between.
func TestScriptedModelServerSingleQueueAnswersEntriesInOrder(t *testing.T) {
	s := newScriptedModelServer(t, map[string][]modelEntry{
		"m-solo": {
			replyFrames(sseStream(textChunk("one"))),
			replyFrames(sseStream(textChunk("two"))),
			replyFrames(sseStream(textChunk("three"))),
		},
	})

	for i, want := range []string{"one", "two", "three"} {
		_, _, body := postModel(t, s.url()+"/chat/completions", `{"model":"m-solo","messages":[]}`)
		if got := streamedText(body); got != want {
			t.Errorf("POST %d answered %q, want %q", i, got, want)
		}
	}
	if n := len(s.requestBodies(t)); n != 3 {
		t.Errorf("recorded %d request bodies, want 3", n)
	}
	// A run that stayed inside its script is silent: the check only fires on a miss.
	s.checkNoUnscriptedRequests(t)
}

// The unscripted-request case: a model id no queue was ever written for is met the
// same way an exhausted one is — 500 naming that model, and the miss on the record
// — while a model that is scripted keeps answering from its own queue untouched.
func TestScriptedModelServerUnscriptedModelFailsLoudlyNamingIt(t *testing.T) {
	s := newScriptedModelServer(t, map[string][]modelEntry{
		"m-a": {replyFrames(sseStream(textChunk("scripted")))},
	})

	status, _, body := postModelResponse(t, s.url()+"/chat/completions", `{"model":"m-never","messages":[]}`)
	if status != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", status)
	}
	if !strings.Contains(body, "m-never") {
		t.Errorf("body = %q, want it to name the unscripted model m-never", body)
	}

	// The miss neither drains nor poisons the queue that does exist.
	_, _, body = postModel(t, s.url()+"/chat/completions", `{"model":"m-a","messages":[]}`)
	if got := streamedText(body); got != "scripted" {
		t.Errorf("m-a answered %q, want %q", got, "scripted")
	}

	if got := s.unscriptedModels(); !slices.Equal(got, []string{"m-never"}) {
		t.Errorf("unscriptedModels = %q, want [m-never]", got)
	}
}

// streamedText reads the first delta of a streamed reply — all these tests
// need to name. An unspent queue, and a body carrying no readable delta, both read
// as "", which the callers' non-empty want then turns into a failure.
func streamedText(frames string) string {
	content, _ := streamedContent(frames)
	return content
}

// A dropped entry scripts the flaky endpoint: the frames written before the tear
// down really reach the client (writeSSE flush included) and the body then ends in
// a read error instead of a clean EOF, so a run reads a truncated stream rather
// than mistaking the cut short body for a finished answer.
func TestScriptedModelServerDropAfterFramesClosesTheConnectionMidBody(t *testing.T) {
	frames := sseFrame(textChunk("half an answer"))

	s := newScriptedModelServer(t, map[string][]modelEntry{
		"m-a": {dropAfterFrames(frames)},
	})

	status, contentType, body, err := postModelTruncated(t, s.url()+"/chat/completions", `{"model":"m-a","messages":[]}`)
	if status != http.StatusOK {
		t.Errorf("status = %d, want 200", status)
	}
	if contentType != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", contentType)
	}
	if err == nil {
		t.Errorf("body read error = nil, want the connection torn down mid-body; body = %q", body)
	}
	if body != frames {
		t.Errorf("bytes delivered before the drop = %q, want the flushed frames %q", body, frames)
	}
	if strings.Contains(body, "[DONE]") {
		t.Errorf("body = %q, want no done marker: a dropped stream does not complete", body)
	}
	if n := len(s.requestBodies(t)); n != 1 {
		t.Errorf("recorded %d request bodies, want 1", n)
	}
}

// postModelTruncated posts and reports whatever the body delivered together with
// the error that ended the read: for a dropped connection the error is the result
// under test, so it cannot be the t.Fatalf postModel treats it as.
func postModelTruncated(t *testing.T, url, body string) (int, string, string, error) {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	return resp.StatusCode, resp.Header.Get("Content-Type"), string(raw), err
}

// An exhausted queue is a scripting mistake, not an answer: the server replies 500
// naming the model and records the miss, so a run that asks for more than the test
// scripted fails the test instead of reading an empty stream as a finished reply.
func TestScriptedModelServerExhaustedQueueFailsLoudlyNamingTheModel(t *testing.T) {
	s := newScriptedModelServer(t, map[string][]modelEntry{
		"m-a": {replyFrames(sseStream(textChunk("only one")))},
	})

	_, _, firstBody := postModel(t, s.url()+"/chat/completions", `{"model":"m-a","messages":[]}`)
	if got := streamedText(firstBody); got != "only one" {
		t.Fatalf("first POST answered %q, want %q", got, "only one")
	}

	status, _, body := postModelResponse(t, s.url()+"/chat/completions", `{"model":"m-a","messages":[]}`)
	if status != http.StatusInternalServerError {
		t.Errorf("second POST status = %d, want 500", status)
	}
	if !strings.Contains(body, "m-a") {
		t.Errorf("second POST body = %q, want it to name the model m-a", body)
	}

	if got := s.unscriptedModels(); !slices.Equal(got, []string{"m-a"}) {
		t.Errorf("unscriptedModels = %q, want [m-a]", got)
	}
	if n := len(s.requestBodies(t)); n != 2 {
		t.Errorf("recorded %d request bodies, want 2: a miss is still a request", n)
	}
}

// A bare modelEntry is the one literal no constructor builds, and it carries no
// reply: it is refused naming the model, not answered with an empty stream.
func TestScriptedModelServerEntryWithNoReplyFailsLoudly(t *testing.T) {
	s := newScriptedModelServer(t, map[string][]modelEntry{
		"m-a": {{}},
	})

	status, _, body := postModelResponse(t, s.url()+"/chat/completions", `{"model":"m-a","messages":[]}`)
	if status != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", status)
	}
	if !strings.Contains(body, "m-a") {
		t.Errorf("body = %q, want it to name the model m-a", body)
	}
}
