package main

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

// normalizeMarkdown lowercases the README and collapses every whitespace run
// to one space, so the assertions below match the prose regardless of line
// wrapping, heading depth or code fences.
func normalizeMarkdown(t *testing.T, path string) string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return strings.ToLower(strings.Join(strings.Fields(string(raw)), " "))
}

// markdownSectionAround returns the text between the heading above needle and
// the next heading after it, so a claim can be pinned to the place that makes
// it ("goimports is offered by the dev-shell section", not "somewhere").
func markdownSectionAround(norm, needle string) string {
	i := strings.Index(norm, needle)
	if i < 0 {
		return ""
	}
	start := strings.LastIndex(norm[:i], "# ")
	end := strings.Index(norm[i+1:], "# ")
	if end < 0 {
		return norm[start:]
	}
	return norm[start : i+1+end]
}

// TestReadmeDocumentsNixWorkflowAndSmokeTest pins the documentation half of
// the flake issue: the README is the only place the dev-shell entry and the
// live-server smoke test are described, and the smoke test is manual — it
// never runs in CI or under `nix flake check` — so an unstated command is a
// workflow nobody can perform.
func TestReadmeDocumentsNixWorkflowAndSmokeTest(t *testing.T) {
	const file = "README.md"
	if _, err := os.Stat(file); err != nil {
		t.Fatalf("%s is missing and must document the nix workflow and the live smoke test: %v", file, err)
	}
	norm := normalizeMarkdown(t, file)

	for _, want := range []string{
		"nix build",
		"./result/bin/spindrift-driver",
		"nix develop",
		"nix flake check",
		"--model",
		"--base-url",
		"$spindrift_base_url",
		"$spindrift_api_key",
		"llama.cpp",
	} {
		if !strings.Contains(norm, want) {
			t.Errorf("%s does not mention %q", file, want)
		}
	}

	// The dev shell exists to hand out the Go toolchain, so its own section
	// must name all three tools the shell installs.
	devShell := markdownSectionAround(norm, "nix develop")
	if devShell == "" {
		t.Fatal("no section documents `nix develop`")
	}
	for _, tool := range []string{"go", "gopls", "goimports"} {
		if !regexp.MustCompile(`\b` + tool + `\b`).MatchString(devShell) {
			t.Errorf("the `nix develop` section does not state that the shell provides %s", tool)
		}
	}

	// `nix flake check` is the gate that runs the suite; documenting it without
	// saying what it runs would not distinguish it from a build.
	checks := markdownSectionAround(norm, "nix flake check")
	if checks == "" {
		t.Fatal("no section documents `nix flake check`")
	}
	if !strings.Contains(checks, "go test") {
		t.Error("the `nix flake check` section must say the check runs the go test suite")
	}

	// The live smoke test is the only end-to-end verification the project has,
	// and it needs all three knobs from main.go to point at a remote server.
	smoke := markdownSectionAround(norm, "smoke test")
	if smoke == "" {
		t.Fatal("no section documents the manual live smoke test")
	}
	for _, want := range []string{"--model", "--base-url", "$spindrift_base_url", "$spindrift_api_key", "manual"} {
		if !strings.Contains(smoke, want) {
			t.Errorf("the smoke-test section does not document %s", want)
		}
	}
	// The key is read from the environment only; a flag would put the secret in
	// argv, so the README must never show one.
	if strings.Contains(norm, "--api-key") {
		t.Error("README documents an --api-key flag; SPINDRIFT_API_KEY is env-only by design")
	}

	// llama.cpp is excluded on purpose; the exclusion has to travel with its
	// reason or a reader tries to add it back.
	llama := markdownSectionAround(norm, "llama.cpp")
	if llama == "" {
		t.Fatal("no section states that llama.cpp is excluded")
	}
	if !strings.Contains(llama, "remote") {
		t.Error("the llama.cpp exclusion must state that the Model server is always remote")
	}
	if !regexp.MustCompile(`not packaged|is not included|excluded|never packaged|not bundled`).MatchString(llama) {
		t.Error("the llama.cpp note must say it is deliberately not packaged")
	}
}
