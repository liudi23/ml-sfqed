#!/usr/bin/env bash
# Build Ptarmigan (pinned release) outside the Dropbox folder.
#   bash scripts/install_ptarmigan.sh [BUILD_DIR]
# Default BUILD_DIR: ~/build. Binary: $BUILD_DIR/ptarmigan/target/release/ptarmigan
#
# Rust: uses an existing `cargo` if found; otherwise installs Rust from conda-forge with micromamba
# (this route works where static.rust-lang.org / rustup is blocked, as in Claude's Linux shell).
# On macOS you can instead install Rust with `curl https://sh.rustup.rs -sSf | sh` or `brew install rust`.
set -euo pipefail
TAG=v1.6.1
BUILD_DIR=${1:-$HOME/build}
mkdir -p "$BUILD_DIR"

if ! command -v cargo >/dev/null 2>&1; then
  if [ -x "$HOME/rustenv/bin/cargo" ]; then
    export PATH="$HOME/rustenv/bin:$PATH"
  else
    case "$(uname -s)-$(uname -m)" in
      Linux-aarch64) SUB=linux-aarch64 ;; Linux-x86_64) SUB=linux-64 ;;
      Darwin-arm64) SUB=osx-arm64 ;; Darwin-x86_64) SUB=osx-64 ;;
      *) echo "unsupported platform"; exit 1 ;;
    esac
    tmp=$(mktemp -d)
    curl -sSL -o "$tmp/mm.tar.bz2" "https://conda.anaconda.org/conda-forge/$SUB/micromamba-2.10.0-0.tar.bz2"
    tar -xjf "$tmp/mm.tar.bz2" -C "$tmp"
    MAMBA_ROOT_PREFIX="$HOME/mamba" "$tmp/bin/micromamba" create -y -q -p "$HOME/rustenv" -c conda-forge rust
    export PATH="$HOME/rustenv/bin:$PATH"
  fi
fi
cargo --version

cd "$BUILD_DIR"
[ -d ptarmigan ] || git clone -q https://github.com/tgblackburn/ptarmigan.git
cd ptarmigan
git fetch -q --tags
git checkout -q "$TAG"
cargo build --release -q
echo "Built: $BUILD_DIR/ptarmigan/target/release/ptarmigan ($TAG, commit $(git rev-parse --short HEAD))"
