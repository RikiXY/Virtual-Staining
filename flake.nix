{
  description = "Python 3.11 + uv dev shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      python = pkgs.python311;
      
      runtimeLibs = with pkgs; [
        stdenv.cc.cc.lib
      ];
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          uv
          python
          gnumake
          git
          ruff
          pyright
        ];
        
        UV_PYTHON = "${python}/bin/python";
        UV_PYTHON_DOWNLOADS = "never";
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
        ];

        shellHook = ''
          if [ -d /usr/lib/wsl/lib ]; then
            export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
          fi

          echo "Entered Nix shell"
          echo "uv: $(uv --version)"
          echo "python: $(${python}/bin/python3.11 --version)"
        '';
      };
    };
}
