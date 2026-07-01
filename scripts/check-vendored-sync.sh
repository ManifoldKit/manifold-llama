#!/bin/bash
#
# check-vendored-sync.sh — detect drift between this repo's vendored copies
# of ManifoldKit core fixtures/scenarios and the core source at the tag this
# repo's Package.resolved is pinned to.
#
# manifold-tools-llama hand-copies two trees from ManifoldKit core because
# `manifold-tools-llama` cannot resolve core's source-relative test paths at
# runtime:
#   Sources/manifold-tools-llama/Scenarios/*.json
#     <- Sources/ManifoldTools/Scenarios/built-in/*.json
#   Sources/manifold-tools-llama/Fixtures/manifold-tools/**
#     <- Tests/Fixtures/manifold-tools/**
# Nothing keeps these copies in sync automatically, so this script compares
# their SHA-256 against the same paths in core, fetched from
# raw.githubusercontent.com at the core tag matching this repo's resolved
# ManifoldKit pin (Package.resolved "manifoldkit".state.version -> "v<version>"
# tag, e.g. 0.63.0 -> v0.63.0).
#
# IMPORTANT: this script must stay Bash 3.2 compatible — it runs under
# macOS's /bin/bash in CI. No `declare -A`, no `mapfile`, no `${var,,}`.
#
# Usage:
#   scripts/check-vendored-sync.sh [--warn|--strict]
#     --warn    (default) always exit 0; prints DRIFT/MISSING-UPSTREAM as
#                warnings only.
#     --strict  exit 1 if any file shows DRIFT or MISSING-UPSTREAM. Network
#                failures never cause a non-zero exit in either mode.

set -uo pipefail

MODE="warn"
for arg in "$@"; do
  case "$arg" in
    --strict) MODE="strict" ;;
    --warn) MODE="warn" ;;
    *)
      echo "usage: $0 [--warn|--strict]" >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

PACKAGE_RESOLVED="Package.resolved"
if [ ! -f "$PACKAGE_RESOLVED" ]; then
  echo "warn: $PACKAGE_RESOLVED not found; skipping vendored-sync check" >&2
  exit 0
fi

CORE_VERSION=""
if command -v jq >/dev/null 2>&1; then
  CORE_VERSION="$(jq -r '.pins[] | select(.identity == "manifoldkit") | .state.version // empty' "$PACKAGE_RESOLVED" 2>/dev/null)"
fi
if [ -z "$CORE_VERSION" ]; then
  # Portable fallback if jq is unavailable: grep the pin block, then the
  # first "version" key inside it.
  CORE_VERSION="$(grep -A6 '"identity" *: *"manifoldkit"' "$PACKAGE_RESOLVED" | grep '"version"' | head -1 | sed -E 's/.*"version" *: *"([^"]+)".*/\1/')"
fi

if [ -z "$CORE_VERSION" ]; then
  echo "warn: could not determine ManifoldKit pin version from $PACKAGE_RESOLVED; skipping vendored-sync check" >&2
  exit 0
fi

CORE_TAG="v${CORE_VERSION}"
RAW_BASE="https://raw.githubusercontent.com/ManifoldKit/ManifoldKit/${CORE_TAG}"

OK_COUNT=0
DRIFT_COUNT=0
MISSING_UPSTREAM_COUNT=0
NETWORK_ERROR_COUNT=0

# Quick reachability probe. If we can't even reach raw.githubusercontent.com,
# don't spam 404/timeout rows for every file — warn once and exit 0.
if ! curl -fsSL --max-time 10 -o /dev/null "${RAW_BASE}/README.md" 2>/dev/null; then
  echo "warn: could not reach ${RAW_BASE} (network unavailable or tag missing); skipping vendored-sync check" >&2
  exit 0
fi

echo "Checking vendored files against ManifoldKit ${CORE_TAG} ..."
echo ""
printf '%-72s %s\n' "FILE" "STATUS"
printf '%-72s %s\n' "----" "------"

check_file() {
  # $1 = local path (relative to repo root), $2 = core-relative path
  local_path="$1"
  core_path="$2"

  if [ ! -f "$local_path" ]; then
    printf '%-72s %s\n' "$local_path" "MISSING-LOCAL"
    return
  fi

  local_sha="$(shasum -a 256 "$local_path" | awk '{print $1}')"

  tmpfile="$(mktemp)"
  http_code="$(curl -s --max-time 15 -o "$tmpfile" -w '%{http_code}' "${RAW_BASE}/${core_path}" 2>/dev/null)"
  curl_exit=$?

  if [ $curl_exit -ne 0 ]; then
    printf '%-72s %s\n' "$local_path" "NETWORK-ERROR"
    NETWORK_ERROR_COUNT=$((NETWORK_ERROR_COUNT + 1))
  elif [ "$http_code" = "404" ]; then
    printf '%-72s %s\n' "$local_path" "MISSING-UPSTREAM"
    MISSING_UPSTREAM_COUNT=$((MISSING_UPSTREAM_COUNT + 1))
  elif [ "$http_code" != "200" ]; then
    printf '%-72s %s\n' "$local_path" "NETWORK-ERROR (HTTP $http_code)"
    NETWORK_ERROR_COUNT=$((NETWORK_ERROR_COUNT + 1))
  else
    remote_sha="$(shasum -a 256 "$tmpfile" | awk '{print $1}')"
    if [ "$remote_sha" = "$local_sha" ]; then
      printf '%-72s %s\n' "$local_path" "OK"
      OK_COUNT=$((OK_COUNT + 1))
    else
      printf '%-72s %s\n' "$local_path" "DRIFT"
      DRIFT_COUNT=$((DRIFT_COUNT + 1))
    fi
  fi
  rm -f "$tmpfile"
}

# 1. Scenarios/*.json <- Sources/ManifoldTools/Scenarios/built-in/*.json
SCENARIOS_DIR="Sources/manifold-tools-llama/Scenarios"
if [ -d "$SCENARIOS_DIR" ]; then
  for f in "$SCENARIOS_DIR"/*.json; do
    [ -f "$f" ] || continue
    name="$(basename "$f")"
    check_file "$f" "Sources/ManifoldTools/Scenarios/built-in/${name}"
  done
fi

# 2. Fixtures/manifold-tools/** <- Tests/Fixtures/manifold-tools/**
FIXTURES_DIR="Sources/manifold-tools-llama/Fixtures/manifold-tools"
if [ -d "$FIXTURES_DIR" ]; then
  while IFS= read -r f; do
    rel="${f#"$FIXTURES_DIR"/}"
    check_file "$f" "Tests/Fixtures/manifold-tools/${rel}"
  done < <(find "$FIXTURES_DIR" -type f | sort)
fi

TOTAL=$((OK_COUNT + DRIFT_COUNT + MISSING_UPSTREAM_COUNT + NETWORK_ERROR_COUNT))
echo ""
echo "Summary: ${TOTAL} checked, ${OK_COUNT} OK, ${DRIFT_COUNT} DRIFT, ${MISSING_UPSTREAM_COUNT} MISSING-UPSTREAM, ${NETWORK_ERROR_COUNT} NETWORK-ERROR (core ${CORE_TAG})"

if [ "$DRIFT_COUNT" -gt 0 ] || [ "$MISSING_UPSTREAM_COUNT" -gt 0 ]; then
  if [ "$MODE" = "strict" ]; then
    echo "error: vendored-sync drift detected (--strict)" >&2
    exit 1
  else
    echo "warn: vendored-sync drift detected (run with --strict to fail CI on this)" >&2
  fi
fi

exit 0
