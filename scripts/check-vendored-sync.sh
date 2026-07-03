#!/bin/bash
#
# check-vendored-sync.sh — detect orphaned scenario overrides in this repo
# vs. ManifoldKit core at the tag this repo's Package.resolved is pinned to.
#
# Post-D1 refactor (MK 0.64+): manifold-tools-llama no longer vendors full
# copies of core's bundled scenario corpus or fixture tree — those are
# consumed live from the `ManifoldTools` product (`ScenarioLoader.loadBuiltIn()`
# / `ToolFixtures.bundledRoot()` via `ScenarioCLIHarness`). The only vendored
# content left is:
#   Sources/manifold-tools-llama/ScenarioOverrides/*.json
# — four scenarios (shopping-list-budget, parallel-readme-comparison,
# oversize-tool-output, structured-json-extraction) whose assertion wording is
# DELIBERATELY tuned for llama/gemma soak behaviour and diverges from core's
# copy at the same id. Byte-identical comparison against core would therefore
# always report false-positive "drift" for these four files by design.
#
# What this script actually checks instead: that every override still
# corresponds to a scenario id core still ships (Sources/ManifoldTools/Scenarios
# /built-in/<name>.json exists upstream at the resolved tag). An override
# whose upstream counterpart has vanished (core renamed/retired the scenario)
# is an ORPHAN — it silently stops being spliced by id in `loadScenarios()`
# and should be cleaned up or re-targeted.
#
# IMPORTANT: this script must stay Bash 3.2 compatible — it runs under
# macOS's /bin/bash in CI. No `declare -A`, no `mapfile`, no `${var,,}`.
#
# Usage:
#   scripts/check-vendored-sync.sh [--warn|--strict]
#     --strict  (default) exit 1 if any override is orphaned. Network failures
#                never cause a non-zero exit in either mode.
#     --warn    always exit 0; prints ORPHAN as a warning only.

set -uo pipefail

MODE="strict"
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
ORPHAN_COUNT=0
NETWORK_ERROR_COUNT=0

# Quick reachability probe. If we can't even reach raw.githubusercontent.com,
# don't spam 404/timeout rows for every file — warn once and exit 0.
if ! curl -fsSL --max-time 10 -o /dev/null "${RAW_BASE}/README.md" 2>/dev/null; then
  echo "warn: could not reach ${RAW_BASE} (network unavailable or tag missing); skipping vendored-sync check" >&2
  exit 0
fi

echo "Checking scenario overrides against ManifoldKit ${CORE_TAG} built-in corpus ..."
echo ""
printf '%-72s %s\n' "FILE" "STATUS"
printf '%-72s %s\n' "----" "------"

check_override() {
  # $1 = local override path (relative to repo root)
  local_path="$1"
  name="$(basename "$local_path")"
  core_path="Sources/ManifoldTools/Scenarios/built-in/${name}"

  http_code="$(curl -s --max-time 15 -o /dev/null -w '%{http_code}' "${RAW_BASE}/${core_path}" 2>/dev/null)"
  curl_exit=$?

  if [ $curl_exit -ne 0 ]; then
    printf '%-72s %s\n' "$local_path" "NETWORK-ERROR"
    NETWORK_ERROR_COUNT=$((NETWORK_ERROR_COUNT + 1))
  elif [ "$http_code" = "404" ]; then
    printf '%-72s %s\n' "$local_path" "ORPHAN (no upstream scenario at this id)"
    ORPHAN_COUNT=$((ORPHAN_COUNT + 1))
  elif [ "$http_code" != "200" ]; then
    printf '%-72s %s\n' "$local_path" "NETWORK-ERROR (HTTP $http_code)"
    NETWORK_ERROR_COUNT=$((NETWORK_ERROR_COUNT + 1))
  else
    printf '%-72s %s\n' "$local_path" "OK"
    OK_COUNT=$((OK_COUNT + 1))
  fi
}

OVERRIDES_DIR="Sources/manifold-tools-llama/ScenarioOverrides"
if [ -d "$OVERRIDES_DIR" ]; then
  for f in "$OVERRIDES_DIR"/*.json; do
    [ -f "$f" ] || continue
    check_override "$f"
  done
fi

TOTAL=$((OK_COUNT + ORPHAN_COUNT + NETWORK_ERROR_COUNT))
echo ""
echo "Summary: ${TOTAL} checked, ${OK_COUNT} OK, ${ORPHAN_COUNT} ORPHAN, ${NETWORK_ERROR_COUNT} NETWORK-ERROR (core ${CORE_TAG})"

if [ "$ORPHAN_COUNT" -gt 0 ]; then
  if [ "$MODE" = "strict" ]; then
    echo "error: orphaned scenario override(s) detected (--strict)" >&2
    exit 1
  else
    echo "warn: orphaned scenario override(s) detected (run with --strict to fail CI on this)" >&2
  fi
fi

exit 0
