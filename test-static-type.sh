#!/usr/bin/env bash
set -euo pipefail

# ── Phase 1: Positive tests ──────────────────────────────────────────────────
echo "=== Phase 1: Positive type checks ==="
echo "Running pyright on static_tests/ and examples/..."
pyright static_tests/ examples/
echo "Phase 1 passed."

# ── Phase 2: Negative test verification ──────────────────────────────────────
echo ""
echo "=== Phase 2: Negative test verification ==="
echo "Verifying that every '# type: ignore' line is a real type error..."

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

# Collect all file:line pairs with '# type: ignore'
declare -A ignore_files  # tracks which files need temp copies
declare -a ignore_locations=()  # file:line pairs (original paths)

while IFS=: read -r file line _rest; do
    ignore_locations+=("$file:$line")
    ignore_files["$file"]=1
done < <(grep -rn '# type: ignore' static_tests/)

count=${#ignore_locations[@]}
echo "Found $count '# type: ignore' lines across ${#ignore_files[@]} files."

if [[ $count -eq 0 ]]; then
    echo "No negative tests found. Skipping phase 2."
    exit 0
fi

# Create temp copies with '# type: ignore' stripped
for file in "${!ignore_files[@]}"; do
    mkdir -p "$tmpdir/$(dirname "$file")"
    sed 's/  *# type: ignore.*//' "$file" > "$tmpdir/$file"
done

# Run pyright on the temp copies (expect errors)
echo "Running pyright on stripped copies..."
pyright_output=""
if pyright_output=$(pyright "$tmpdir/static_tests/" 2>&1); then
    echo "FAIL: pyright exited 0 on stripped files — expected type errors!"
    exit 1
fi

# Parse pyright errors into a set of file:line
declare -A error_locations=()
while IFS= read -r errline; do
    # Match lines like: /tmp/.../static_tests/test_basic.py:53:5 - error: ...
    if [[ $errline =~ $tmpdir/([^:]+):([0-9]+):[0-9]+\ -\ error: ]]; then
        rel_path="${BASH_REMATCH[1]}"
        err_line="${BASH_REMATCH[2]}"
        error_locations["$rel_path:$err_line"]=1
    fi
done <<< "$pyright_output"

# Verify every ignore location got a pyright error
missing=0
for loc in "${ignore_locations[@]}"; do
    # loc is like static_tests/test_basic.py:53
    if [[ -z "${error_locations[$loc]+x}" ]]; then
        echo "FAIL: No pyright error at $loc — the '# type: ignore' is not guarding a real error"
        missing=$((missing + 1))
    fi
done

if [[ $missing -gt 0 ]]; then
    echo "FAIL: $missing '# type: ignore' line(s) did not produce pyright errors."
    exit 1
fi

echo "All $count '# type: ignore' lines confirmed as real type errors."
echo "Phase 2 passed."
echo ""
echo "All static type checks passed."
