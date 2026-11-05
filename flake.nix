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

      # One eval of nixpkgs per system, shared by every output below.
      pkgsFor = nixpkgs.lib.genAttrs systems (system: import nixpkgs { inherit system; });

      # nixpkgs' lib provides no eachSystem/eachDefaultSystem/perSystem -- those
      # are flake-utils helpers -- so iterate with lib.genAttrs, which already
      # yields the { <system> = <attrset> } layout flakes resolve
      # (`packages.<system>.default`, never `<system>.packages`).
      forSystems = f: nixpkgs.lib.genAttrs systems (system: f pkgsFor.${system});

      driver = pkgs: pkgs.buildGoModule {
        pname = "spindrift-driver";

        # stdenv derives the derivation `name` from pname + version and has no
        # default version any more; the repo publishes no releases to take one
        # from, so the usual `unstable` stands for "built from source".
        version = "unstable";

        src = self;

        # stdlib-only per docs/adr/0001: nothing to hash, no vendor dir.
        vendorHash = null;

        # nixpkgs' go defaults CGO_ENABLED=1 on x86_64-linux, which would
        # link libc; the driver ships as a static binary.
        env.CGO_ENABLED = "0";
      };
    in
    {
      packages = forSystems (pkgs: { default = driver pkgs; });

      # `nix flake check` evaluates the `checks` attrset and never a
      # package's checkPhase, so the suite gates the flake only via this
      # binding. Override the shared driver derivation instead of a second
      # build, so package and check cannot drift apart.
      checks = forSystems (pkgs: {
        go-test = (driver pkgs).overrideAttrs (final: prev: {
          doCheck = true;
          doInstallCheck = false;

          # main_test.go looks `sleep` up with exec.LookPath and creates a
          # probe file in the process cwd, so the check phase needs these on
          # PATH and runs in a writable build dir, never the store.
          nativeCheckInputs = with pkgs; [ bash coreutils ];
        });
      });

      # nixpkgs has no top-level goimports attribute (`pkgs.goimports`
      # does not exist); the binary ships in gotools. `go` here is the same
      # pinned nixpkgs go that builds the package, so the shell clears
      # go.mod's `go 1.26.8` floor under GOTOOLCHAIN=local, and gopls speaks
      # the same toolchain's protocol.
      devShells = forSystems (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [ go gopls gotools ];
        };
      });
    };
}
