package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"time"
)

// envConfigPath names the override. The SPINDRIFT_* spelling matches the other
// environment variables the Driver reads: there is no SPINDRIFT_DRIVER_* or XDG
// layer to keep in step, so one variable and one relative default.
const (
	envConfigPath = "SPINDRIFT_CONFIG"
	// defaultDriverConfigPath is home-relative: the caller joins it to $HOME, so
	// this loader stays free of environment reads and can be handed any path.
	defaultDriverConfigPath = ".config/spindrift-driver/config.json"
)

// errNoDriverConfig lets the caller tell "the file is not there" from every
// other failure, which is the difference between a fatal error and nothing:
// the default path may be absent, an explicitly named file may not.
var errNoDriverConfig = errors.New("file not found")

// driverConfig is one Driver Config file, already typed. A zero field means the
// file said nothing about it — the same "empty reads as unset" rule the API key
// already follows in main.go — so a partial file never masks a flag.
type driverConfig struct {
	baseURL string
	apiKey  string
	timeout time.Duration // 0 = absent; non-positive values are rejected at load time
}

// knownConfigFields is the whole file schema: the contents list of the issue,
// and deliberately not the rest of the runtime knobs — model and max turns stay
// flags so a checked-in file cannot silently change what an agent run does.
var knownConfigFields = []string{"api_key", "base_url", "timeout"}

func isKnownConfigField(key string) bool {
	for _, k := range knownConfigFields {
		if k == key {
			return true
		}
	}
	return false
}

// loadDriverConfig reads exactly one path and never falls back. It writes
// nothing anywhere: stdout is the event stream, exclusively.
func loadDriverConfig(path string) (driverConfig, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return driverConfig{}, fmt.Errorf("driver config %s: %w", path, errNoDriverConfig)
		}
		return driverConfig{}, fmt.Errorf("driver config %s: cannot read: %v", path, err)
	}
	// Checked before decoding: unmarshalling into a map leaves the map nil for a
	// top-level null just as it does for an empty file, so the decode error alone
	// cannot tell "no file content" from "content of the wrong shape".
	if kind := jsonKind(raw); kind != "object" {
		return driverConfig{}, fmt.Errorf("driver config %s: malformed JSON: top-level value is %s, want an object", path, kind)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return driverConfig{}, fmt.Errorf("driver config %s: malformed JSON: %v", path, err)
	}
	// Sorted before the unknown-key scan so a file with several complaints always
	// reports the same one: map iteration order would otherwise pick at random.
	keys := make([]string, 0, len(fields))
	for k := range fields {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if !isKnownConfigField(k) {
			return driverConfig{}, fmt.Errorf("driver config %s: unknown field %q", path, k)
		}
	}
	var cfg driverConfig
	if raw, ok := fields["base_url"]; ok {
		s, err := configString(path, "base_url", raw)
		if err != nil {
			return driverConfig{}, err
		}
		cfg.baseURL = s
	}
	if raw, ok := fields["api_key"]; ok {
		s, err := configString(path, "api_key", raw)
		if err != nil {
			return driverConfig{}, err
		}
		cfg.apiKey = s
	}
	if raw, ok := fields["timeout"]; ok {
		s, err := configString(path, "timeout", raw)
		if err != nil {
			return driverConfig{}, err
		}
		d, err := time.ParseDuration(s)
		if err != nil {
			return driverConfig{}, fmt.Errorf("driver config %s: field %q must be a Go duration string such as \"90s\", got %q", path, "timeout", s)
		}
		if d <= 0 {
			return driverConfig{}, fmt.Errorf("driver config %s: field %q must be positive, got %q", path, "timeout", s)
		}
		cfg.timeout = d
	}
	return cfg, nil
}

// configString decodes a known key that has to hold a JSON string. The kind is
// decided from the raw bytes rather than from the decode error because json
// leaves its target untouched on null: a silent no-op there would read as "the
// file said nothing about this field", which is exactly the silent fallback this
// loader exists to rule out.
func configString(path, key string, raw json.RawMessage) (string, error) {
	if kind := jsonKind(raw); kind != "string" {
		return "", fmt.Errorf("driver config %s: field %q must be a string, got %s", path, key, kind)
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		// Unreachable for a value lifted out of a document that already decoded,
		// and kept rather than dropped so a caller handing in unvalidated bytes
		// gets an error instead of a silently empty field.
		return "", fmt.Errorf("driver config %s: field %q: %v", path, key, err)
	}
	return s, nil
}

// jsonKind names the JSON type of a raw value that json.Unmarshal has already
// accepted, by its first non-space byte — valid JSON admits no other reading of
// that byte. The vocabulary is the file author's ("boolean", not Go's "bool").
func jsonKind(raw json.RawMessage) string {
	for _, b := range raw {
		switch {
		case b == ' ' || b == '\t' || b == '\n' || b == '\r':
			continue
		case b == '{':
			return "object"
		case b == '[':
			return "array"
		case b == '"':
			return "string"
		case b == 't' || b == 'f':
			return "boolean"
		case b == 'n':
			return "null"
		default:
			return "number"
		}
	}
	return "empty"
}
