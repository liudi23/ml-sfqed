#!/usr/bin/env bash
# Fetch the public data of Los et al. (Nature Communications; HPLSE 2024) for the inference benchmark.
# Source: https://github.com/ELos385/RadiationReaction (no licence file: used locally, not redistributed).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D="$ROOT/data/raw/los2024"
TMP="$(mktemp -d)"
git clone -q --depth 1 https://github.com/ELos385/RadiationReaction.git "$TMP/RR"
mkdir -p "$D"
cp "$TMP/RR/data/Bayesian_analysis/mcmc_input_spectra.h5" "$TMP/RR/data/Bayesian_analysis/Model_evidences.csv" \
   "$TMP/RR/data/Frequentist_analysis/hits_and_nulls_selected.pkl" "$TMP/RR/results/Bayesian_analysis/Bayesian_inference_data.h5" "$D/"
cp "$TMP/RR/code/Frequentist analysis/mystats.py" "$D/mystats_los2024.py"
echo "Fetched into $D (commit $(git -C "$TMP/RR" rev-parse --short HEAD))"
