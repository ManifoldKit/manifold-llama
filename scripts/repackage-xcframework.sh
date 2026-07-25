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
# with all dSYMs dropped. Measured for b9859: ~769 MB → ~30 MB extracted,
# ~257 MB → ~11 MB zipped.
#
# It then zips the slim framework, computes the SwiftPM package checksum, and
# prints the exact `url` + `checksum` lines to paste into Package.swift. The URL
# is a PLACEHOLDER — the maintainer must host the slim zip as a manifold-llama
# GitHub release asset and substitute the real download URL. See
# docs/LLAMA_CONTRACT.md ("Slimming the xcframework").
#
# Usage:
#   scripts/repackage-xcframework.sh                 # build b9859 (default)
#   BUILD=b9900 scripts/repackage-xcframework.sh     # override the upstream build
#   scripts/repackage-xcframework.sh b9900           # same, as a positional arg
#   WORK_DIR=/tmp/x scripts/repackage-xcframework.sh # override the work dir
#
# Env vars / args:
#   BUILD       upstream build tag (default: b9859). First positional arg wins.
#   WORK_DIR    working directory for download/unpack/output
#               (default: <repo>/tmp/repackage-xcframework).
#
# Idempotent: re-running reuses an already-downloaded zip (only if it still
# passes an integrity check) and rebuilds the slim artifact in place.

set -euo pipefail

# --- configuration ----------------------------------------------------------

BUILD="${1:-${BUILD:-b9859}}"
UPSTREAM_URL="https://github.com/ggml-org/llama.cpp/releases/download/${BUILD}/llama-${BUILD}-xcframework.zip"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/tmp/repackage-xcframework}"
# Guard against an empty WORK_DIR (e.g. `WORK_DIR= scripts/...`): every `rm -rf`
# below is rooted at WORK_DIR, so an empty value could escalate to deleting the
# wrong tree. Require an absolute path.
[[ -n "${WORK_DIR}" && "${WORK_DIR}" = /* ]] \
    || { printf 'error: WORK_DIR must be a non-empty absolute path, got: %q\n' "${WORK_DIR}" >&2; exit 1; }

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
    command -v "${tool}" >/dev/null 2>&1 || fail "required tool not found on PATH: ${tool}"
done

# --- 1b. expected upstream checksums -----------------------------------------
#
# `unzip -tqq` below only catches corruption/truncation — it says nothing about
# authenticity. A tampered or substituted upstream asset with an intact zip
# structure sails straight through it. This table pins the SHA256 of the
# upstream `llama-<BUILD>-xcframework.zip` we have actually verified, and
# step 2b asserts the download matches it BEFORE any repackaging happens.
#
# Bash 3.2 (the macOS system /bin/bash this script targets — see the CI
# comment in .github/workflows/ci.yml) has no associative arrays, so this is a
# `case` lookup rather than `declare -A`.
#
# To add a new BUILD: download the upstream zip, `shasum -a 256` it, verify
# the value out-of-band if possible (ggml-org does not publish a checksums
# file per-release as of b9859 — cross-check via a second independent fetch,
# e.g. a different network path, if you want extra assurance), then add a
# case arm here. Record how you obtained it in the PR description.
expected_upstream_sha256() {
    case "$1" in
        b9859) printf '%s\n' "1fcf5b1ba2fd0890c5bbbc5932e1d1893f495e3de3a13331d05384f3c6e25620" ;;
        *) printf '' ;;
    esac
}

EXPECTED_UPSTREAM_SHA256="$(expected_upstream_sha256 "${BUILD}")"

if [[ -z "${EXPECTED_UPSTREAM_SHA256}" ]]; then
    # Fail-closed by default: an unrecorded BUILD has no authenticity check,
    # which is exactly the hole this script exists to close. Refuse unless the
    # caller explicitly opts out (e.g. while pinning the checksum for a brand
    # new BUILD for the first time).
    if [[ "${ALLOW_UNVERIFIED_UPSTREAM:-0}" == "1" ]]; then
        log "WARNING: no recorded upstream SHA256 for BUILD=${BUILD}; proceeding UNVERIFIED because ALLOW_UNVERIFIED_UPSTREAM=1. Do not host the resulting slim zip as a release asset without adding a checksum entry and re-running verified."
    else
        fail "no recorded upstream SHA256 for BUILD=${BUILD} in expected_upstream_sha256() — refusing to repackage an unverified download. Obtain the real SHA256 (see the comment above expected_upstream_sha256), add a case arm, or set ALLOW_UNVERIFIED_UPSTREAM=1 to explicitly bypass this for a one-off/dry run."
    fi
fi

# --- 2. download ------------------------------------------------------------

mkdir -p "${WORK_DIR}"

# Re-verify a cached download before trusting it: a previous run could have been
# interrupted (truncated zip) or the file could be corrupt. `unzip -t` confirms
# the central directory + CRCs are intact; a stale/corrupt cache is discarded and
# re-downloaded rather than silently reused.
if [[ -f "${UPSTREAM_ZIP}" ]] && unzip -tqq "${UPSTREAM_ZIP}" >/dev/null 2>&1; then
    log "Upstream zip already present and verified, skipping download: ${UPSTREAM_ZIP}"
else
    if [[ -f "${UPSTREAM_ZIP}" ]]; then
        log "Cached zip missing or failed integrity check — re-downloading: ${UPSTREAM_ZIP}"
        rm -f "${UPSTREAM_ZIP}"
    fi
    log "Downloading ${UPSTREAM_URL}"
    # Download to a .partial sidecar and only promote it on success, so an
    # interrupted curl never leaves a truncated file at the cache path.
    rm -f "${UPSTREAM_ZIP}.partial"
    curl --fail --location --progress-bar -o "${UPSTREAM_ZIP}.partial" "${UPSTREAM_URL}"
    unzip -tqq "${UPSTREAM_ZIP}.partial" >/dev/null 2>&1 \
        || fail "downloaded zip failed integrity check (truncated or corrupt): ${UPSTREAM_URL}"
    mv "${UPSTREAM_ZIP}.partial" "${UPSTREAM_ZIP}"
fi

ORIGINAL_ZIP_SIZE="$(human_size "${UPSTREAM_ZIP}")"

# --- 2b. verify upstream authenticity ----------------------------------------
#
# This is the actual threat closure: a compromised/substituted upstream
# release would faithfully pass `unzip -t` (intact zip structure) but fail
# this checksum comparison. Runs on BOTH freshly-downloaded and reused-cache
# zips (the cache-hit branch above never checks authenticity, only
# corruption), and fires BEFORE any repackaging (extraction/xcodebuild) work
# begins.
if [[ -n "${EXPECTED_UPSTREAM_SHA256}" ]]; then
    log "Verifying upstream zip SHA256 against pinned value for BUILD=${BUILD}…"
    ACTUAL_UPSTREAM_SHA256="$(shasum -a 256 "${UPSTREAM_ZIP}" | awk '{print $1}')"
    if [[ "${ACTUAL_UPSTREAM_SHA256}" != "${EXPECTED_UPSTREAM_SHA256}" ]]; then
        fail "upstream checksum mismatch for BUILD=${BUILD}: expected ${EXPECTED_UPSTREAM_SHA256}, got ${ACTUAL_UPSTREAM_SHA256}. The downloaded ${UPSTREAM_URL} does NOT match the pinned checksum in expected_upstream_sha256() — refusing to repackage. This could mean upstream re-published the asset (verify out-of-band before updating the pin) or the download was tampered with. The cached zip at ${UPSTREAM_ZIP} was left in place for inspection; delete it before re-running."
    fi
    log "Upstream checksum verified: ${ACTUAL_UPSTREAM_SHA256}"
else
    log "Skipping upstream checksum verification (ALLOW_UNVERIFIED_UPSTREAM=1, no pinned value for BUILD=${BUILD})"
fi

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

# Authoritative slice list comes from the xcframework's own Info.plist
# (LibraryIdentifier values), NOT from directory names — that is exactly what
# SwiftPM / Xcode consult when resolving a slice for a platform. Assert the slim
# output contains EXACTLY the slices we intended: no more (a stray upstream
# slice leaking through) and no fewer (a slice silently dropped, which would
# yield a short xcframework that fails to link on the missing platform).
SLIM_INFO_PLIST="${SLIM_XCFRAMEWORK}/Info.plist"
[[ -f "${SLIM_INFO_PLIST}" ]] || fail "slim xcframework has no Info.plist: ${SLIM_INFO_PLIST}"

ACTUAL_SLICES="$(/usr/libexec/PlistBuddy -c "Print" "${SLIM_INFO_PLIST}" 2>/dev/null \
    | grep 'LibraryIdentifier' | sed -E 's/.*= //' | sort)"
EXPECTED_SLICES="$(printf '%s\n' "${KEEP_SLICES[@]}" | sort)"

if [[ "${ACTUAL_SLICES}" != "${EXPECTED_SLICES}" ]]; then
    printf '%s\n' "expected slices:" "${EXPECTED_SLICES}" "but got:" "${ACTUAL_SLICES}" >&2
    fail "slim xcframework slice set does not match the intended ${#KEEP_SLICES[@]} slices"
fi

SLICE_LIST="$(printf '%s' "${ACTUAL_SLICES}" | tr '\n' ' ')"

# dSYMs (and any stray standalone *.dSYM bundles / BCSymbolMaps) must be absent
# from the OUTPUT tree. Check the slim framework, not the input.
DSYM_COUNT="$(find "${SLIM_XCFRAMEWORK}" \( -name 'dSYMs' -o -name '*.dSYM' -o -name 'BCSymbolMaps' \) -type d | wc -l | tr -d ' ')"

# --- 7b. provenance record ---------------------------------------------------
#
# Emitted as a file, not hand-written after the fact, so it can't drift from
# what this run actually did. Ship this alongside the release asset (see
# docs/LLAMA_CONTRACT.md "Trust chain") as PROVENANCE-<BUILD>.md, or fold its
# contents into the release notes.
REPACKAGE_SCRIPT_COMMIT="$(cd "${REPO_ROOT}" && git rev-parse HEAD 2>/dev/null || echo "unknown (not a git checkout or no commits yet)")"
PROVENANCE_FILE="${WORK_DIR}/PROVENANCE-${BUILD}.md"

cat > "${PROVENANCE_FILE}" <<EOF
# Provenance — vendor-llama-${BUILD}

Generated by scripts/repackage-xcframework.sh on $(date -u '+%Y-%m-%dT%H:%M:%SZ').

| Field | Value |
|---|---|
| Upstream tag | \`${BUILD}\` |
| Upstream asset URL | ${UPSTREAM_URL} |
| Upstream asset SHA256 (raw \`shasum -a 256\`) | \`${ACTUAL_UPSTREAM_SHA256:-UNVERIFIED — ALLOW_UNVERIFIED_UPSTREAM=1 was set, see docs/LLAMA_CONTRACT.md}\` |
| Slim asset filename | \`$(basename "${SLIM_ZIP}")\` |
| Slim asset SwiftPM checksum (\`swift package compute-checksum\`, NOT a raw sha256sum — this is the value \`Package.swift\`'s \`.binaryTarget(checksum:)\` verifies at resolve time) | \`${CHECKSUM}\` |
| Slices included | ${SLICE_LIST} |
| dSYMs present in slim output | ${DSYM_COUNT} |
| repackage-xcframework.sh commit | \`${REPACKAGE_SCRIPT_COMMIT}\` |

## What this proves / does not prove

- The upstream SHA256 above proves the bytes we downloaded from
  \`${UPSTREAM_URL}\` matched a value pinned in this script at
  \`${REPACKAGE_SCRIPT_COMMIT}\` — i.e. integrity-after-publication against
  ggml-org's CI-built release asset.
- It does **not** prove that asset corresponds to a reviewed llama.cpp source
  revision — that would require a full source rebuild, which is explicitly
  deferred (see docs/LLAMA_CONTRACT.md, "Trust chain").
- The slim SwiftPM checksum proves the slim zip hosted as the
  \`vendor-llama-${BUILD}\` release asset is byte-identical to the zip this
  run produced — nothing more. It is not comparable to the upstream SHA256
  (different algorithm output format, computed over a different, repackaged
  file).
EOF

log "Provenance record written: ${PROVENANCE_FILE}"

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
echo "  Provenance record:   ${PROVENANCE_FILE}"
echo "--------------------------------------------------------------------------"
echo "  Paste into Package.swift (.binaryTarget name: \"llama-cpp\"):"
echo
echo "    url: \"https://github.com/ManifoldKit/manifold-llama/releases/download/<TAG>/llama-${BUILD}-slim.xcframework.zip\","
echo "    checksum: \"${CHECKSUM}\""
echo
echo "  NOTE: the url above is a PLACEHOLDER. Host ${SLIM_ZIP##*/} as a"
echo "  manifold-llama GitHub release asset, then substitute the real URL."
echo "=========================================================================="

[[ "${DSYM_COUNT}" == "0" ]] || fail "dSYMs unexpectedly present in slim framework"
