package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
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
}

// replyFrames scripts one answer as raw SSE frames.
func replyFrames(frames string) modelEntry { return modelEntry{frames: frames} }

// replyCompletion scripts one answer as a whole completion body, served as the
// streaming frames the same answer arrives in.
func replyCompletion(body string) modelEntry { return modelEntry{completion: body} }

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
	bodies [][]byte
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
		s.bodies = append(s.bodies, body)
		var req struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(body, &req)
		queue := s.queues[req.Model]
		var entry modelEntry
		if len(queue) > 0 {
			entry = queue[0]
			s.queues[req.Model] = queue[1:]
		}
		s.mu.Unlock()

		writeSSE(w, entry.sseBody())
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

// requestBodies returns every recorded request body in arrival order.
func (s *scriptedModelServer) requestBodies(t *testing.T) [][]byte {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([][]byte, len(s.bodies))
	copy(out, s.bodies)
	return out
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
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read response body: %v", err)
	}
	return resp.StatusCode, resp.Header.Get("Content-Type"), string(raw)
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
}

// streamedText reads the first delta of a streamed reply — all these tests
// need to name. An unspent queue, and a body carrying no readable delta, both read
// as "", which the callers' non-empty want then turns into a failure.
func streamedText(frames string) string {
	content, _ := streamedContent(frames)
	return content
}
