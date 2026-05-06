{
  description = "Python 3.11 + uv dev shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      lib = nixpkgs.lib;

      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      forAllSystems = lib.genAttrs systems;
    in {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};

          python = pkgs.python311;

          linuxRuntimeLibs = [
            pkgs.stdenv.cc.cc.lib
          ];
        in {
          default = pkgs.mkShell ({
            packages = [
              pkgs.uv
              python
              pkgs.gnumake
              pkgs.git
            ];

            UV_PYTHON = "${python}/bin/python";
            UV_PYTHON_DOWNLOADS = "never";

            shellHook = lib.optionalString pkgs.stdenv.isLinux ''
              if [ -d /usr/lib/wsl/lib ]; then
                export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
              fi
            '' + ''
              echo "Entered Nix shell"
              echo "uv: $(uv --version)"
              echo "python: $(${python}/bin/python --version)"
            '';
          } // lib.optionalAttrs pkgs.stdenv.isLinux {
            LD_LIBRARY_PATH = lib.makeLibraryPath linuxRuntimeLibs;
          });
        });
    };
}