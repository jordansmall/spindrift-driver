package main

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

// TestCIHasNixGateJob pins the acceptance criteria of the "nix is never
// evaluated in CI" issue: every PR must evaluate the flake, not just compile
// with a hand-installed Go. `go test` passing proves nothing about
// `nix build`/`nix flake check` — a flake.nix that fails to evaluate, or a
// flake.lock refreshed to a nixpkgs too old for go.mod's floor, sailed
// through green. The gate lives in a separate `nix` job (the Go matrix must
// stay nix-free so its runtime does not explode), so pin the job key at the
// jobs indentation rather than any step's text.
func TestCIHasNixGateJob(t *testing.T) {
	raw, err := os.ReadFile(".github/workflows/ci.yml")
	if err != nil {
		t.Fatalf("read .github/workflows/ci.yml: %v", err)
	}
	src := string(raw)

	if !regexp.MustCompile(`(?m)^  nix:\s*$`).MatchString(src) {
		t.Fatal("ci.yml must define a separate `nix` job; the Go-only `test` job never evaluates the flake")
	}

	// GitHub runners ship no usable flakes-enabled nix; the job must install
	// one and ask for the experimental features explicitly, because runner
	// images and nix releases have flipped those defaults before.
	if !strings.Contains(src, "install-nix-action") && !strings.Contains(src, "nix-installer-action") {
		t.Error("the nix job must install nix (install-nix-action or nix-installer-action); ubuntu runners do not provide flakes")
	}
	if !strings.Contains(src, "nix-command flakes") {
		t.Error("the nix job must enable experimental-features = nix-command flakes explicitly")
	}

	// Two flake-check shapes are load-bearing: a cheap --no-build eval of
	// every system on each PR, and a real build+test run. Split on lines so a
	// single `nix flake check` cannot satisfy both pins.
	var evalGate, buildGate bool
	for _, line := range strings.Split(src, "\n") {
		if !strings.Contains(line, "nix flake check") {
			continue
		}
		if strings.Contains(line, "--no-build") && strings.Contains(line, "--all-systems") {
			evalGate = true
		}
		if !strings.Contains(line, "--no-build") {
			buildGate = true
		}
	}
	if !evalGate {
		t.Error("ci.yml must run `nix flake check --no-build --all-systems` so every output of every system is evaluated per PR")
	}
	if !buildGate {
		t.Error("ci.yml must run `nix flake check` (a real build) so the flake gate covers more than evaluation")
	}

	// Lock drift: a PR that edits flake.nix without refreshing flake.lock
	// resolves against the stale lock silently. Neither Nix nor Lix ships a
	// --locked flag (verified against the 2.32.8 manual and runner nix), so
	// the portable gate is `nix flake lock` -- which adds missing inputs but
	// never updates existing ones -- followed by a git diff on the lock.
	if !(strings.Contains(src, "nix flake lock") && strings.Contains(src, "git diff")) {
		t.Error("ci.yml must detect a stale flake.lock by running `nix flake lock` and failing on a resulting git diff")
	}

	// GOTOOLCHAIN floor: buildGoModule sets GOTOOLCHAIN=local, so a lock
	// refreshed to a nixpkgs whose go predates go.mod's `go` line fails the
	// build with no download rescue. The job must compare the pinned nixpkgs'
	// go.version against the floor rather than trusting review.
	if !strings.Contains(src, "go.version") {
		t.Error("ci.yml must eval the pinned nixpkgs' go.version to assert the toolchain floor")
	}
	if !strings.Contains(src, "go.mod") {
		t.Error("ci.yml must read the toolchain floor from go.mod")
	}
}

// TestReadmeBuildDocumentsNixpkgsGoFloor: refreshing flake.lock to an old
// nixpkgs breaks every build (GOTOOLCHAIN=local cannot download go), so the
// Build section must state the floor next to the `nix flake update` prose.
// The floor is go.mod's requirement (1.26.8), never the currently-pinned
// nixpkgs' version (1.26.10) — pinning the latter in docs rots on the next
// refresh.
func TestReadmeBuildDocumentsNixpkgsGoFloor(t *testing.T) {
	norm := normalizeMarkdown(t, "README.md")

	section := markdownSectionAround(norm, "nix flake update")
	if section == "" {
		t.Fatal("no section documents `nix flake update`")
	}
	if !strings.Contains(section, "1.26.8") {
		t.Error("the flake.lock refresh prose must state the go >= 1.26.8 floor the refreshed nixpkgs must satisfy")
	}
	if !regexp.MustCompile(`1\.26\.8|go\.mod`).MatchString(section) {
		t.Error("the floor note must tie the requirement to go.mod's go line")
	}
	if strings.Contains(section, "1.26.10") {
		t.Error("README must document the go.mod floor (1.26.8), not the currently-pinned nixpkgs go version (1.26.10)")
	}
}
