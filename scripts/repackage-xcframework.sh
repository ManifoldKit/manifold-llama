#!/usr/bin/env bash
#
# repackage-xcframework.sh — slim the upstream llama.cpp xcframework.
#
# The upstream `llama-b<NNNN>-xcframework.zip` asset from ggml-org ships seven
# platform slices and a fat dSYM bundled in every slice (~200 MB zipped /
# ~616 MB extracted). manifold-llama's `Package.swift` only declares
# `.iOS(.v18)` and `.macOS(.v15)`, so the tvOS and visionOS slices are dead
# weight, and dSYMs are never needed to build/link/run a binaryTarget.
#
# This script rebuilds a NEW xcframework containing ONLY:
#   - macos-arm64_x86_64
#   - ios-arm64
#   - ios-arm64_x86_64-simulator
# with all dSYMs dropped. The result is ~40-50 MB extracted.
#
# It then zips the slim framework, computes the SwiftPM package checksum, and
# prints the exact `url` + `checksum` lines to paste into Package.swift. The URL
# is a PLACEHOLDER — the maintainer must host the slim zip as a manifold-llama
# GitHub release asset and substitute the real download URL. See
# docs/LLAMA_CONTRACT.md ("Slimming the xcframework").
#
# Usage:
#   scripts/repackage-xcframework.sh                 # build b9744 (default)
#   BUILD=b9800 scripts/repackage-xcframework.sh     # override the upstream build
#   scripts/repackage-xcframework.sh b9800           # same, as a positional arg
#   WORK_DIR=/tmp/x scripts/repackage-xcframework.sh # override the work dir
#
# Env vars / args:
#   BUILD       upstream build tag (default: b9744). First positional arg wins.
#   WORK_DIR    working directory for download/unpack/output
#               (default: <repo>/tmp/repackage-xcframework).
#
# Idempotent: re-running reuses an already-downloaded zip and rebuilds the
# slim artifact in place.

set -euo pipefail

# --- configuration ----------------------------------------------------------

BUILD="${1:-${BUILD:-b9744}}"
UPSTREAM_URL="https://github.com/ggml-org/llama.cpp/releases/download/${BUILD}/llama-${BUILD}-xcframework.zip"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/tmp/repackage-xcframework}"

# The three slices we keep (must match Package.swift's declared platforms).
KEEP_SLICES=(
    "macos-arm64_x86_64"
    "ios-arm64"
    "ios-arm64_x86_64-simulator"
)

UPSTREAM_ZIP="${WORK_DIR}/llama-${BUILD}-xcframework.zip"
EXTRACT_DIR="${WORK_DIR}/extracted"
# UPSTREAM_XCFRAMEWORK is discovered after unzip (the archive nests it under
# build-apple/, and the layout has shifted between upstream releases).
UPSTREAM_XCFRAMEWORK=""
SLIM_XCFRAMEWORK="${WORK_DIR}/slim/llama.xcframework"
SLIM_ZIP="${WORK_DIR}/llama-${BUILD}-slim.xcframework.zip"

# --- helpers ----------------------------------------------------------------

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

human_size() {
    # du -h on the path; print just the size column.
    du -sh "$1" 2>/dev/null | cut -f1
}

# --- 1. tool check ----------------------------------------------------------

log "Checking required tools…"
for tool in curl unzip xcodebuild swift; do
    command -v "$tool" >/dev/null 2>&1 || fail "required tool not found on PATH: $tool"
done

# --- 2. download ------------------------------------------------------------

mkdir -p "${WORK_DIR}"

if [[ -f "${UPSTREAM_ZIP}" ]]; then
    log "Upstream zip already present, skipping download: ${UPSTREAM_ZIP}"
else
    log "Downloading ${UPSTREAM_URL}"
    curl --fail --location --progress-bar -o "${UPSTREAM_ZIP}.partial" "${UPSTREAM_URL}"
    mv "${UPSTREAM_ZIP}.partial" "${UPSTREAM_ZIP}"
fi

ORIGINAL_ZIP_SIZE="$(human_size "${UPSTREAM_ZIP}")"

# --- 3. unzip ---------------------------------------------------------------

log "Extracting upstream xcframework…"
rm -rf "${EXTRACT_DIR}"
mkdir -p "${EXTRACT_DIR}"
unzip -q "${UPSTREAM_ZIP}" -d "${EXTRACT_DIR}"

# Locate the xcframework. Upstream nests it under build-apple/, but be tolerant
# of the layout by searching.
UPSTREAM_XCFRAMEWORK="$(find "${EXTRACT_DIR}" -maxdepth 3 -type d -name 'llama.xcframework' | head -n1)"
[[ -n "${UPSTREAM_XCFRAMEWORK}" && -d "${UPSTREAM_XCFRAMEWORK}" ]] \
    || fail "could not find llama.xcframework after unzip; archive layout changed?"

ORIGINAL_EXTRACTED_SIZE="$(human_size "${UPSTREAM_XCFRAMEWORK}")"

# --- 4. build the slim xcframework -----------------------------------------

log "Building slim xcframework (keeping: ${KEEP_SLICES[*]})…"

rm -rf "$(dirname "${SLIM_XCFRAMEWORK}")"
mkdir -p "$(dirname "${SLIM_XCFRAMEWORK}")"

CREATE_ARGS=()
for slice in "${KEEP_SLICES[@]}"; do
    framework="${UPSTREAM_XCFRAMEWORK}/${slice}/llama.framework"
    [[ -d "${framework}" ]] || fail "expected slice missing: ${framework}"
    # Deliberately NOT passing -debug-symbols, so dSYMs are dropped.
    CREATE_ARGS+=(-framework "${framework}")
done

xcodebuild -create-xcframework "${CREATE_ARGS[@]}" -output "${SLIM_XCFRAMEWORK}"

# --- 5. zip the slim framework ----------------------------------------------

log "Zipping slim xcframework…"
rm -f "${SLIM_ZIP}"
# ditto preserves symlinks/framework bundle structure correctly (same tool the
# upstream release uses). -c -k --keepParent => a zip whose top entry is
# llama.xcframework, matching what SwiftPM expects.
ditto -c -k --sequesterRsrc --keepParent "${SLIM_XCFRAMEWORK}" "${SLIM_ZIP}"

SLIM_ZIP_SIZE="$(human_size "${SLIM_ZIP}")"
SLIM_EXTRACTED_SIZE="$(human_size "${SLIM_XCFRAMEWORK}")"

# --- 6. checksum ------------------------------------------------------------

log "Computing SwiftPM package checksum…"
CHECKSUM="$(cd "${WORK_DIR}" && swift package compute-checksum "$(basename "${SLIM_ZIP}")")"

# --- 7. verify + summary ----------------------------------------------------

DSYM_COUNT="$(find "${SLIM_XCFRAMEWORK}" -name 'dSYMs' -type d | wc -l | tr -d ' ')"
SLICE_LIST="$(find "${SLIM_XCFRAMEWORK}" -maxdepth 1 -mindepth 1 -type d -exec basename {} \; | sort | tr '\n' ' ')"

echo
echo "=========================================================================="
echo " repackage-xcframework summary  (build: ${BUILD})"
echo "=========================================================================="
echo "  Original zip:        ${ORIGINAL_ZIP_SIZE}   (${UPSTREAM_ZIP})"
echo "  Original extracted:  ${ORIGINAL_EXTRACTED_SIZE}   (7 slices + dSYMs)"
echo "  Slim zip:            ${SLIM_ZIP_SIZE}   (${SLIM_ZIP})"
echo "  Slim extracted:      ${SLIM_EXTRACTED_SIZE}"
echo "  Slices included:     ${SLICE_LIST}"
echo "  dSYM directories:    ${DSYM_COUNT}  $([[ "${DSYM_COUNT}" == "0" ]] && echo '(none — good)' || echo '(UNEXPECTED — dSYMs present!)')"
echo "  Slim checksum:       ${CHECKSUM}"
echo "--------------------------------------------------------------------------"
echo "  Paste into Package.swift (.binaryTarget name: \"llama-cpp\"):"
echo
echo "    url: \"https://github.com/roryford/manifold-llama/releases/download/<TAG>/llama-${BUILD}-slim.xcframework.zip\","
echo "    checksum: \"${CHECKSUM}\""
echo
echo "  NOTE: the url above is a PLACEHOLDER. Host ${SLIM_ZIP##*/} as a"
echo "  manifold-llama GitHub release asset, then substitute the real URL."
echo "=========================================================================="

[[ "${DSYM_COUNT}" == "0" ]] || fail "dSYMs unexpectedly present in slim framework"
