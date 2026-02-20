#!/usr/bin/env bash
set -euo pipefail
echo "Running pyright on static_tests/ and examples/..."
pyright static_tests/ examples/
echo "All static type checks passed."
