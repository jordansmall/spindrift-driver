{
  description = "Spindrift agent-sandbox driver";

  # go.mod sets `go 1.26.8` and buildGoModule forces GOTOOLCHAIN=local, so this
  # input must pin a nixpkgs whose go is >= 1.26.8 or every build fails.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs, ... }:
    let
      # darwin deferred: CI is ubuntu-only and the darwin sandbox rejects the
      # test suite's loopback binds.
      systems = [ "x86_64-linux" "aarch64-linux" ];
    in
    nixpkgs.lib.eachSystem systems (system:
      let
        pkgs = import nixpkgs { inherit system; };

        driver = pkgs.buildGoModule {
          pname = "spindrift-driver";
          src = self;

          # stdlib-only per docs/adr/0001: nothing to hash, no vendor dir.
          vendorHash = null;

          # nixpkgs' go defaults CGO_ENABLED=1 on x86_64-linux, which would
          # link libc; the driver ships as a static binary.
          env.CGO_ENABLED = "0";
        };
      in
      {
        packages.default = driver;

        # `nix flake check` evaluates the `checks` attrset and never a
        # package's checkPhase, so the suite gates the flake only via this
        # binding. Override the shared driver derivation instead of a second
        # build, so package and check cannot drift apart.
        checks.go-test = driver.overrideAttrs (final: prev: {
          doCheck = true;
          doInstallCheck = false;

          # main_test.go looks `sleep` up with exec.LookPath and creates a
          # probe file in the process cwd, so the check phase needs these on
          # PATH and runs in a writable build dir, never the store.
          nativeCheckInputs = with pkgs; [ bash coreutils ];
        });
      });
}
