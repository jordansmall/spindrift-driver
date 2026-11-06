package main

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// assertErrMessage checks the two things every loader error has to carry: the
// path it was reading, and the problem in terms the operator can act on.
func assertErrMessage(t *testing.T, err error, path string, want ...string) {
	t.Helper()
	if err == nil {
		t.Fatalf("error = nil, want %q", want)
	}
	msg := err.Error()
	if !strings.Contains(msg, path) {
		t.Errorf("error %q does not name the file %q", msg, path)
	}
	for _, w := range want {
		if !strings.Contains(msg, w) {
			t.Errorf("error %q does not contain %q", msg, w)
		}
	}
}

// writeDriverConfig puts contents in a file of its own and hands back the path,
// which is the only thing loadDriverConfig is given.
func writeDriverConfig(t *testing.T, contents string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
		t.Fatalf("write config fixture: %v", err)
	}
	return path
}

func TestDriverConfigPath(t *testing.T) {
	for _, tc := range []struct {
		name      string
		lookupEnv func(string) (string, bool)
		wantPath  string
		wantFound bool
	}{
		{
			name:      "no env at all",
			lookupEnv: func(string) (string, bool) { return "", false },
			wantPath:  "",
			wantFound: false,
		},
		{
			// The default path is spelled out instead of built from
			// defaultDriverConfigPath, so renaming the default fails here.
			name: "HOME only",
			lookupEnv: func(name string) (string, bool) {
				if name == "HOME" {
					return "/home/dev", true
				}
				return "", false
			},
			wantPath:  filepath.Join("/home/dev", ".config/spindrift-driver/config.json"),
			wantFound: false,
		},
		{
			name: "override without HOME",
			lookupEnv: func(name string) (string, bool) {
				if name == "SPINDRIFT_CONFIG" {
					return "chosen.json", true
				}
				return "", false
			},
			wantPath:  "chosen.json",
			wantFound: true,
		},
		{
			name: "override beats HOME",
			lookupEnv: func(name string) (string, bool) {
				switch name {
				case "SPINDRIFT_CONFIG":
					return "/etc/spindrift/config.json", true
				case "HOME":
					return "/home/dev", true
				}
				return "", false
			},
			wantPath:  "/etc/spindrift/config.json",
			wantFound: true,
		},
		{
			name: "set-but-empty override reads as unset",
			lookupEnv: func(name string) (string, bool) {
				switch name {
				case "SPINDRIFT_CONFIG":
					return "", true
				case "HOME":
					return "/home/dev", true
				}
				return "", false
			},
			wantPath:  filepath.Join("/home/dev", ".config/spindrift-driver/config.json"),
			wantFound: false,
		},
		{
			name: "set-but-empty HOME is no default file",
			lookupEnv: func(name string) (string, bool) {
				if name == "HOME" {
					return "", true
				}
				return "", false
			},
			wantPath:  "",
			wantFound: false,
		},
		{
			// Neither segment nor tilde is touched: the operator's string is the
			// path, so a mistake in it stays visible in the error message.
			name: "absolute override is verbatim",
			lookupEnv: func(name string) (string, bool) {
				if name == "SPINDRIFT_CONFIG" {
					return "/opt/spindrift/./cfg.json", true
				}
				return "", false
			},
			wantPath:  "/opt/spindrift/./cfg.json",
			wantFound: true,
		},
		{
			name: "relative override is verbatim",
			lookupEnv: func(name string) (string, bool) {
				if name == "SPINDRIFT_CONFIG" {
					return "./cfg.json", true
				}
				return "", false
			},
			wantPath:  "./cfg.json",
			wantFound: true,
		},
		{
			name: "tilde in override is not expanded",
			lookupEnv: func(name string) (string, bool) {
				if name == "SPINDRIFT_CONFIG" {
					return "~/cfg.json", true
				}
				return "", false
			},
			wantPath:  "~/cfg.json",
			wantFound: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path, explicit := driverConfigPath(tc.lookupEnv)
			if path != tc.wantPath {
				t.Errorf("path = %q, want %q", path, tc.wantPath)
			}
			if explicit != tc.wantFound {
				t.Errorf("explicit = %v, want %v", explicit, tc.wantFound)
			}
		})
	}
}

func TestLoadDriverConfigWellFormed(t *testing.T) {
	path := writeDriverConfig(t, `{"base_url":"http://model.test:9/v1","api_key":"sk-test","timeout":"90s"}`)

	cfg, err := loadDriverConfig(path)
	if err != nil {
		t.Fatalf("loadDriverConfig(%q) error = %v, want nil", path, err)
	}
	if cfg.baseURL != "http://model.test:9/v1" {
		t.Errorf("baseURL = %q, want %q", cfg.baseURL, "http://model.test:9/v1")
	}
	if cfg.apiKey != "sk-test" {
		t.Errorf("apiKey = %q, want %q", cfg.apiKey, "sk-test")
	}
	if cfg.timeout != 90*time.Second {
		t.Errorf("timeout = %v, want %v", cfg.timeout, 90*time.Second)
	}
}

func TestLoadDriverConfigPartialFileLeavesRestZero(t *testing.T) {
	path := writeDriverConfig(t, `{"api_key":"sk-only"}`)

	cfg, err := loadDriverConfig(path)
	if err != nil {
		t.Fatalf("loadDriverConfig(%q) error = %v, want nil", path, err)
	}
	if cfg.baseURL != "" || cfg.timeout != 0 {
		t.Errorf("cfg = %+v, want only apiKey set", cfg)
	}
	if cfg.apiKey != "sk-only" {
		t.Errorf("apiKey = %q, want %q", cfg.apiKey, "sk-only")
	}
}

func TestLoadDriverConfigMissingFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "absent.json")

	cfg, err := loadDriverConfig(path)
	assertErrMessage(t, err, path, "file not found")
	if !errors.Is(err, errNoDriverConfig) {
		t.Errorf("errors.Is(err, errNoDriverConfig) = false, want true; err = %v", err)
	}
	if cfg != (driverConfig{}) {
		t.Errorf("cfg = %+v, want zero value on error", cfg)
	}
}

func TestLoadDriverConfigMalformedJSON(t *testing.T) {
	for _, tc := range []struct{ name, contents string }{
		{"truncated object", `{"base_url": `},
		{"empty file", ``},
		{"not json at all", `base_url = http://model.test`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := writeDriverConfig(t, tc.contents)

			_, err := loadDriverConfig(path)
			assertErrMessage(t, err, path, "malformed JSON")
			if errors.Is(err, errNoDriverConfig) {
				t.Errorf("errors.Is(err, errNoDriverConfig) = true, want false; err = %v", err)
			}
		})
	}
}

func TestLoadDriverConfigUnknownField(t *testing.T) {
	for _, tc := range []struct{ name, contents, want string }{
		{"stray key", `{"base_url":"http://model.test","region":"eu"}`, "region"},
		{"near-miss typo", `{"apiKEy":"sk-test"}`, "apiKEy"},
		{"model is not a config field", `{"model":"gpt-4o-mini"}`, "model"},
		{"max turns is not a config field", `{"max_turns":5}`, "max_turns"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := writeDriverConfig(t, tc.contents)

			_, err := loadDriverConfig(path)
			assertErrMessage(t, err, path, "unknown field", tc.want)
			if errors.Is(err, errNoDriverConfig) {
				t.Errorf("errors.Is(err, errNoDriverConfig) = true, want false; err = %v", err)
			}
		})
	}
}

func TestLoadDriverConfigUnknownFieldReportsLexicallyFirst(t *testing.T) {
	// The key order in the file must not decide which complaint is printed: the
	// same file always names the same key.
	path := writeDriverConfig(t, `{"zulu":1,"alpha":2,"Mango":3}`)

	_, err := loadDriverConfig(path)
	assertErrMessage(t, err, path, "unknown field", `"Mango"`)
	if strings.Contains(err.Error(), "alpha") || strings.Contains(err.Error(), "zulu") {
		t.Errorf("error %q names a key other than the lexically first", err.Error())
	}
}

func TestLoadDriverConfigWrongType(t *testing.T) {
	for _, tc := range []struct{ name, contents, want string }{
		{"base_url number", `{"base_url":8080}`, `field "base_url" must be a string, got number`},
		{"base_url null", `{"base_url":null}`, `field "base_url" must be a string, got null`},
		{"base_url object", `{"base_url":{"v":1}}`, `field "base_url" must be a string, got object`},
		{"base_url array", `{"base_url":["a"]}`, `field "base_url" must be a string, got array`},
		{"api_key boolean", `{"api_key":true}`, `field "api_key" must be a string, got boolean`},
		{"api_key null", `{"api_key":null}`, `field "api_key" must be a string, got null`},
		{"timeout number", `{"timeout":90}`, `field "timeout" must be a string, got number`},
		{"timeout null", `{"timeout":null}`, `field "timeout" must be a string, got null`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := writeDriverConfig(t, tc.contents)

			_, err := loadDriverConfig(path)
			assertErrMessage(t, err, path, tc.want)
			if errors.Is(err, errNoDriverConfig) {
				t.Errorf("errors.Is(err, errNoDriverConfig) = true, want false; err = %v", err)
			}
		})
	}
}

func TestLoadDriverConfigDuration(t *testing.T) {
	for _, tc := range []struct{ name, contents, want string }{
		{"not a duration", `{"timeout":"banana"}`, `field "timeout" must be a Go duration string such as "90s", got "banana"`},
		{"zero", `{"timeout":"0s"}`, `field "timeout" must be positive, got "0s"`},
		{"negative", `{"timeout":"-5s"}`, `field "timeout" must be positive, got "-5s"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := writeDriverConfig(t, tc.contents)

			_, err := loadDriverConfig(path)
			assertErrMessage(t, err, path, tc.want)
			if errors.Is(err, errNoDriverConfig) {
				t.Errorf("errors.Is(err, errNoDriverConfig) = true, want false; err = %v", err)
			}
		})
	}
}

func TestLoadDriverConfigTrailingBytes(t *testing.T) {
	path := writeDriverConfig(t, `{"base_url":"http://model.test"} {"base_url":"http://other"}`)

	_, err := loadDriverConfig(path)
	assertErrMessage(t, err, path, "malformed JSON")
}

func TestLoadDriverConfigTopLevelNotAnObject(t *testing.T) {
	for _, tc := range []struct{ name, contents string }{
		{"array", `[{"base_url":"http://model.test"}]`},
		{"string", `"http://model.test"`},
		{"number", `42`},
		{"null", `null`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := writeDriverConfig(t, tc.contents)

			_, err := loadDriverConfig(path)
			assertErrMessage(t, err, path, "malformed JSON")
		})
	}
}

func TestLoadDriverConfigUnreadablePath(t *testing.T) {
	path := t.TempDir() // a directory: openable, unreadable as a file

	_, err := loadDriverConfig(path)
	assertErrMessage(t, err, path, "cannot read")
	if errors.Is(err, errNoDriverConfig) {
		t.Errorf("errors.Is(err, errNoDriverConfig) = true, want false; err = %v", err)
	}
}

// TestLoadDriverConfigReadsOnlyItsPath: the loader takes the file's location as
// its only input, so nothing about the working directory — which for the Driver
// is the Target repo — can change what it reads. The default path is
// home-relative for the same reason.
func TestLoadDriverConfigReadsOnlyItsPath(t *testing.T) {
	path := writeDriverConfig(t, `{"base_url":"http://elsewhere.test/v1","timeout":"2m"}`)
	t.Chdir(t.TempDir())
	if filepath.IsAbs(defaultDriverConfigPath) {
		t.Errorf("defaultDriverConfigPath %q is absolute, want home-relative", defaultDriverConfigPath)
	}
	if strings.HasPrefix(defaultDriverConfigPath, string(os.PathSeparator)) {
		t.Errorf("defaultDriverConfigPath %q is cwd-relative, want home-relative", defaultDriverConfigPath)
	}

	cfg, err := loadDriverConfig(path)
	if err != nil {
		t.Fatalf("loadDriverConfig(%q) error = %v, want nil", path, err)
	}
	if cfg.baseURL != "http://elsewhere.test/v1" || cfg.timeout != 2*time.Minute {
		t.Errorf("cfg = %+v, want the values the file carries", cfg)
	}
}

func TestLoadDriverConfigPrettyPrinted(t *testing.T) {
	// The kind check reads the first non-space byte, so real-world indentation
	// and newlines inside the file must not be mistaken for a wrong type.
	path := writeDriverConfig(t, "{\n  \"base_url\":\n    \"http://model.test/v1\",\n  \"timeout\":\n\t\t\"45s\"\n}\n")

	cfg, err := loadDriverConfig(path)
	if err != nil {
		t.Fatalf("loadDriverConfig(%q) error = %v, want nil", path, err)
	}
	if cfg.baseURL != "http://model.test/v1" || cfg.timeout != 45*time.Second {
		t.Errorf("cfg = %+v, want the values the file carries", cfg)
	}
}

// TestDriverConfigPathDefaultFileFeedsTheLoader: the resolved default is not
// decoration, and its absence is a loader result rather than a resolution
// error — which is what lets the caller treat a missing default file as
// nothing at all and a named one as fatal.
func TestDriverConfigPathDefaultFileFeedsTheLoader(t *testing.T) {
	homeLookup := func(home string) func(string) (string, bool) {
		return func(name string) (string, bool) {
			if name == "HOME" {
				return home, true
			}
			return "", false
		}
	}

	t.Run("default file is read", func(t *testing.T) {
		home := t.TempDir()
		path := filepath.Join(home, ".config", "spindrift-driver", "config.json")
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatalf("mkdir default config dir: %v", err)
		}
		if err := os.WriteFile(path, []byte(`{"base_url":"http://from-file:1/v1"}`), 0o600); err != nil {
			t.Fatalf("write default config: %v", err)
		}

		got, explicit := driverConfigPath(homeLookup(home))
		if explicit {
			t.Errorf("explicit = true, want false for the default path")
		}
		cfg, err := loadDriverConfig(got)
		if err != nil {
			t.Fatalf("loadDriverConfig(%q) error = %v, want nil", got, err)
		}
		if cfg.baseURL != "http://from-file:1/v1" {
			t.Errorf("baseURL = %q, want %q", cfg.baseURL, "http://from-file:1/v1")
		}
	})

	t.Run("default file absent is the loader's complaint only", func(t *testing.T) {
		home := t.TempDir()

		got, explicit := driverConfigPath(homeLookup(home))
		if explicit {
			t.Errorf("explicit = true, want false for the default path")
		}
		if want := filepath.Join(home, ".config", "spindrift-driver", "config.json"); got != want {
			t.Fatalf("path = %q, want %q", got, want)
		}
		cfg, err := loadDriverConfig(got)
		assertErrMessage(t, err, got, "file not found")
		if !errors.Is(err, errNoDriverConfig) {
			t.Errorf("errors.Is(err, errNoDriverConfig) = false, want true; err = %v", err)
		}
		if cfg != (driverConfig{}) {
			t.Errorf("cfg = %+v, want zero value on error", cfg)
		}
	})
}
