package main

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// squash removes every whitespace run so flake assertions are insensitive to
// layout: a contributor reformatting flake.nix must not break these tests.
func squash(s string) string {
	return strings.Join(strings.Fields(s), "")
}

// stripNixComments drops '#' comment lines so counting assertions see nix
// code only, not prose that happens to name the same attributes.
func stripNixComments(s string) string {
	var keep []string
	for _, line := range strings.Split(s, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "#") {
			continue
		}
		keep = append(keep, line)
	}
	return strings.Join(keep, "\n")
}

func TestFlakeNixBuildsDriverWithBuildGoModule(t *testing.T) {
	raw, err := os.ReadFile("flake.nix")
	if err != nil {
		t.Fatalf("read flake.nix: %v", err)
	}
	flat := squash(string(raw))

	for _, want := range []string{
		"buildGoModule",
		`pname="spindrift-driver"`,
		`vendorHash=null`,
		`CGO_ENABLED="0"`,
	} {
		if !strings.Contains(flat, want) {
			t.Errorf("flake.nix package definition is missing %s", want)
		}
	}

	// A `packages` entry must be reachable as a per-system `default`, whether
	// written as a dotted attr path (packages.<system>.default /
	// packages.default inside eachSystem) or a nested attrset.
	packagesDefault := regexp.MustCompile(`packages(\.[-A-Za-z0-9_<>{}$+]+)*\.default=|packages=\{[^}]*\bdefault=`)
	if !packagesDefault.MatchString(flat) {
		t.Error(`flake.nix must expose packages.<system>.default`)
	}

	// The flake must name x86_64-linux as one of the systems it builds for.
	if !strings.Contains(flat, `"x86_64-linux"`) {
		t.Error(`flake.nix must expose a package for x86_64-linux`)
	}
}

// TestFlakeCheckRunsGoTestSuite pins the `nix flake check` half of the issue:
// flake evaluation never runs a package's checkPhase, so the suite only gates
// the flake once it is published as checks.<system>.go-test. That entry has to
// overrideAttrs the shared driver binding rather than re-declare the build, or
// the two definitions drift apart.
func TestFlakeCheckRunsGoTestSuite(t *testing.T) {
	raw, err := os.ReadFile("flake.nix")
	if err != nil {
		t.Fatalf("read flake.nix: %v", err)
	}
	flat := squash(stripNixComments(string(raw)))

	if !strings.Contains(flat, "checks") {
		t.Error("flake.nix must expose a checks output; nix flake check runs nothing otherwise")
	}

	// checks.<system>.go-test, written as a dotted attr path, a quoted key, or
	// a nested attrset -- but always an override of the driver binding, so
	// buildGoModule may be spelled exactly once in the file.
	goTest := regexp.MustCompile(`checks[.A-Za-z0-9_<>{}$+\[\]"-]*go-test"?\]?=[A-Za-z_][-A-Za-z0-9_]*\.overrideAttrs`)
	if !goTest.MatchString(flat) {
		t.Error("flake.nix must define checks.<system>.go-test as driver.overrideAttrs")
	}
	if n := strings.Count(flat, "buildGoModule"); n != 1 {
		t.Errorf("flake.nix declares %d buildGoModule derivations; the check must reuse the driver package", n)
	}

	for _, want := range []string{"doCheck=true", "doInstallCheck=false", "nativeCheckInputs"} {
		if !strings.Contains(flat, want) {
			t.Errorf("flake.nix go-test check is missing %s", want)
		}
	}

	// The suite looks `sleep` up on PATH and creates an exclusive probe file in
	// the process cwd, so the check needs coreutils/bash present and a writable
	// build dir; a checkPhase running only `go test` would fail on PATH.
	if i := strings.Index(flat, "nativeCheckInputs"); i >= 0 {
		scope := flat[i:min(i+160, len(flat))]
		for _, tool := range []string{"bash", "coreutils"} {
			if !strings.Contains(scope, tool) {
				t.Errorf("flake.nix nativeCheckInputs is missing %s, which the test suite shells out to", tool)
			}
		}
	}
}

// TestNoThirdPartyGoDependencies pins the second half of the flake acceptance
// criterion: flake.nix may only claim vendorHash = null while ADR-0001 holds.
// A new dependency has to be justified in a future ADR and re-flaked before
// this test can go green again. See docs/adr/0001-go-stdlib-only.md.
func TestNoThirdPartyGoDependencies(t *testing.T) {
	raw, err := os.ReadFile("go.mod")
	if err != nil {
		t.Fatalf("read go.mod: %v", err)
	}
	for i, line := range strings.Split(string(raw), "\n") {
		if trimmed := strings.TrimSpace(line); strings.HasPrefix(trimmed, "require") {
			t.Errorf("go.mod:%d declares a dependency (%q); flake.nix sets vendorHash = null", i+1, trimmed)
		}
	}

	err = filepath.WalkDir(".", func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if d.Name() == ".git" {
				return filepath.SkipDir
			}
			return nil
		}
		if d.Name() == "go.sum" {
			t.Errorf("%s exists; flake.nix sets vendorHash = null", path)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk repo: %v", err)
	}
}
