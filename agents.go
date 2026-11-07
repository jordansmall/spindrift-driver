package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// errNoAgentDir lets the caller tell "the directory is not there" from every
// other failure, the way errNoDriverConfig does for the Driver Config: an
// absent default directory means "no Agent Files" and is not fatal, a file that
// exists and is broken always is.
var errNoAgentDir = errors.New("directory not found")

// agentEntry is one Agent File. The frontmatter describes the subagent and the
// body is the system prompt its run starts from; the model id is stored
// verbatim, exactly as the file writes it.
type agentEntry struct {
	Name         string
	Description  string
	Model        string
	Tools        []string
	Spawns       []string
	SystemPrompt string
}

// parseAgentFile reads one Agent File. Frontmatter is hand-rolled rather than
// parsed with a YAML library (docs/adr/0001 keeps the driver stdlib-only): a
// "---" fence, "key: value" scalars, and for the two list fields either an
// inline comma-separated value or a block of "- item" lines under the key.
func parseAgentFile(path string) (agentEntry, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return agentEntry{}, fmt.Errorf("agent file %s: cannot read: %v", path, err)
	}
	lines := splitLines(string(raw))

	// The fence has to open the file: a leading blank line or prose makes the
	// whole file a body with nothing to describe, which is a broken Agent File,
	// not an Agent File with no frontmatter.
	if len(lines) == 0 || strings.TrimSpace(lines[0]) != "---" {
		return agentEntry{}, fmt.Errorf("agent file %s: missing opening --- frontmatter fence", path)
	}
	closeIdx := -1
	for i := 1; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == "---" {
			closeIdx = i
			break
		}
	}
	if closeIdx < 0 {
		return agentEntry{}, fmt.Errorf("agent file %s: unclosed --- frontmatter fence", path)
	}

	var entry agentEntry
	// pendingList points at the list field a block of "- item" lines feeds. Only
	// tools and spawns ever set it, and every "key: value" line clears it, so an
	// item can only follow the list field it belongs to — anywhere else it is an
	// error rather than something to guess about.
	var pendingList *[]string
	lastKey := ""
	// seen holds the fields this file has already written. An unknown key is
	// fatal, so a key that overwrites itself cannot quietly mean last-one-wins;
	// only "key: value" lines reach it, never the "- item" lines that continue a
	// list block, so a block is not read as a repeated field.
	seen := make(map[string]bool)
	for _, line := range lines[1:closeIdx] {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		if trimmed == "-" || strings.HasPrefix(trimmed, "- ") {
			switch {
			case pendingList != nil:
				*pendingList = append(*pendingList, strings.TrimSpace(trimmed[1:]))
			case lastKey != "":
				return agentEntry{}, fmt.Errorf("agent file %s: field %q does not accept a list", path, lastKey)
			default:
				return agentEntry{}, fmt.Errorf("agent file %s: list item with no field before it", path)
			}
			continue
		}
		key, value, ok := strings.Cut(trimmed, ":")
		if !ok {
			return agentEntry{}, fmt.Errorf("agent file %s: malformed frontmatter line %q", path, trimmed)
		}
		key, value = strings.TrimSpace(key), strings.TrimSpace(value)
		if seen[key] {
			return agentEntry{}, fmt.Errorf("agent file %s: field %q given twice", path, key)
		}
		seen[key] = true
		lastKey, pendingList = key, nil
		switch key {
		case "name":
			entry.Name = value
		case "description":
			entry.Description = value
		case "model":
			entry.Model = value
		case "tools":
			entry.Tools = splitInlineList(value)
			pendingList = &entry.Tools
		case "spawns":
			entry.Spawns = splitInlineList(value)
			pendingList = &entry.Spawns
		default:
			return agentEntry{}, fmt.Errorf("agent file %s: unknown field %q", path, key)
		}
	}
	// The name is the handle the parent spawns with, so it is stated in the file
	// and never inferred from the file name: two names for one subagent is one
	// too many to keep in step.
	if entry.Name == "" {
		return agentEntry{}, fmt.Errorf("agent file %s: field %q is required", path, "name")
	}
	if entry.Model == "" {
		return agentEntry{}, fmt.Errorf("agent file %s: field %q is required", path, "model")
	}
	entry.SystemPrompt = trimBlankEdges(strings.Join(lines[closeIdx+1:], "\n"))
	return entry, nil
}

// defaultAgentsDirPath is home-relative for the same reason the Driver Config
// path is: the caller joins it to $HOME, so resolution stays free of environment
// reads and the same code runs under any stub.
const defaultAgentsDirPath = ".config/spindrift-driver/agents"

// agentsDirPath reports where Agent Files live, or "" when there are none to
// look for. HOME goes through the injected lookupEnv, never os.UserHomeDir: that
// would bypass the seam every test stubs and make the byte-exact golden log
// depend on whose home directory the suite runs in. The only override is the
// Driver Config's agents_dir, applied by the caller, because a file the operator
// checked is the one place a directory may be redirected from; an environment
// variable would let any invocation swap in a different roster.
func agentsDirPath(lookupEnv func(string) (string, bool)) string {
	// A set-but-empty HOME reads as unset, as the Driver Config's does: an
	// unresolvable home means "no Agent Files", never an error.
	if home, ok := lookupEnv("HOME"); ok && home != "" {
		return filepath.Join(home, defaultAgentsDirPath)
	}
	return ""
}

// loadAgentDir reads every Agent File in dir as ordered entries. Ordering is
// what makes the spawn offer reproducible across runs, and os.ReadDir supplies
// it sorted by file name — nothing here reorders, so the entry order is the
// directory's lexical order.
func loadAgentDir(dir string) ([]agentEntry, error) {
	dirEntries, err := os.ReadDir(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("agent dir %s: %w", dir, errNoAgentDir)
		}
		return nil, fmt.Errorf("agent dir %s: cannot read: %v", dir, err)
	}
	var entries []agentEntry
	// claimed maps a name to the file that first stated it. findAgent answers
	// with the first match, so a second file under the same name would silently
	// hand every spawn the lexically-first one; the roster is only a roster if
	// each handle names one entry.
	claimed := make(map[string]string)
	for _, de := range dirEntries {
		// Only *.md is an Agent File: NOTES.md.bak and a stray README.txt sit in
		// the same directory without describing a subagent, and a directory of
		// that name is not a file to read.
		if de.IsDir() || !strings.HasSuffix(de.Name(), ".md") {
			continue
		}
		path := filepath.Join(dir, de.Name())
		entry, err := parseAgentFile(path)
		if err != nil {
			return nil, err
		}
		if prev, dup := claimed[entry.Name]; dup {
			return nil, fmt.Errorf("agent dir %s: agent %q is defined by both %s and %s", dir, entry.Name, prev, path)
		}
		claimed[entry.Name] = path
		entries = append(entries, entry)
	}
	return entries, nil
}

// offeredTools is the run's tool list, derived from the roster and nothing
// else: an empty roster means the Model has no subagent it could name, so
// offering the tool would only invite a call that cannot be honoured. bash is
// always there, roster or not.
func offeredTools(agents []agentEntry) []toolSpec {
	if len(agents) == 0 {
		return []toolSpec{bashTool}
	}
	return []toolSpec{bashTool, subagentTool}
}

// findAgent resolves a --agent name against the roster. The match is exact:
// a prefix or case-folded lookup would let a typo in argv run a different
// subagent than the operator named, silently.
func findAgent(agents []agentEntry, name string) (agentEntry, bool) {
	for _, entry := range agents {
		if entry.Name == name {
			return entry, true
		}
	}
	return agentEntry{}, false
}

// splitInlineList reads the comma-separated form of a list field. An empty
// value is no items at all — the block form below may still fill the field.
func splitInlineList(value string) []string {
	if value == "" {
		return nil
	}
	var items []string
	for _, item := range strings.Split(value, ",") {
		if item = strings.TrimSpace(item); item != "" {
			items = append(items, item)
		}
	}
	return items
}

// splitLines splits file contents into lines without a trailing empty element,
// and drops the CR of CRLF so a file written on another platform parses the
// same. Frontmatter matching is exact, so the CR cannot be left in.
func splitLines(s string) []string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	lines := strings.Split(s, "\n")
	if n := len(lines); n > 0 && lines[n-1] == "" {
		lines = lines[:n-1]
	}
	return lines
}

// trimBlankEdges strips blank lines from either end of the system prompt while
// keeping the body verbatim in between: indentation and internal spacing are
// part of what the file author wrote.
func trimBlankEdges(s string) string {
	lines := strings.Split(s, "\n")
	for len(lines) > 0 && strings.TrimSpace(lines[0]) == "" {
		lines = lines[1:]
	}
	for len(lines) > 0 && strings.TrimSpace(lines[len(lines)-1]) == "" {
		lines = lines[:len(lines)-1]
	}
	return strings.Join(lines, "\n")
}
