# Provenance — `vendor-llama-b9859`

This is the committed provenance record for the llama.cpp xcframework
currently pinned in `Package.swift` (`.binaryTarget(name: "llama-cpp", …)`).
It lets anyone walk the trust chain — ggml-org's CI-built binary → our
download of it → our repackage (drops-only) → our checksum pin — without
re-running the repackage. See `docs/LLAMA_CONTRACT.md` ("Trust chain — what
the checksum pin does and does not prove") for the full explanation of what
each link in this chain establishes.

`scripts/repackage-xcframework.sh` now emits a record in this same shape as
an output artifact (`${WORK_DIR}/PROVENANCE-<BUILD>.md`) on every run, so
future version bumps don't have to hand-author this file from scratch — copy
the emitted one in here and adjust wording as needed.

| Field | Value |
|---|---|
| Upstream tag | `b9859` |
| Upstream asset URL | `https://github.com/ggml-org/llama.cpp/releases/download/b9859/llama-b9859-xcframework.zip` |
| Upstream asset SHA256 (raw `shasum -a 256`) | `1fcf5b1ba2fd0890c5bbbc5932e1d1893f495e3de3a13331d05384f3c6e25620` |
| How/when verified | Downloaded directly from the URL above and hashed with `shasum -a 256` on 2026-07-25. ggml-org does not publish a checksums manifest for this release as of b9859 (checked via `gh release view b9859 --repo ggml-org/llama.cpp`), so this is a direct-download hash, not a cross-check against an upstream-published value. This is the value now pinned in `expected_upstream_sha256()` in `scripts/repackage-xcframework.sh`, asserted before every repackage of this `BUILD`. |
| Slim asset filename | `llama-b9859-slim.xcframework.zip` |
| Slim asset checksum (`swift package compute-checksum` of the slim zip — a SHA-256 hex digest, same algorithm/output as `shasum -a 256` on the same file; the value `Package.swift`'s `.binaryTarget(checksum:)` verifies at resolve time) | `88e382d47d12e41c786fcf36d5c829c24004d0ee7e3a11483bf4fb63e1d7b190` (copied from `Package.swift`, the value SwiftPM currently resolves against — independently reproduced by both `shasum -a 256` and `swift package compute-checksum` against the live hosted asset while preparing this record) |
| Slices included | `macos-arm64_x86_64`, `ios-arm64`, `ios-arm64_x86_64-simulator` |
| dSYMs present in slim output | 0 (asserted by the script) |
| repackage-xcframework.sh commit (at time the currently-hosted `vendor-llama-b9859` asset was produced) | [`6ff3f3f`](https://github.com/ManifoldKit/manifold-llama/commit/6ff3f3f) (#127, "bump vendored llama.cpp xcframework pin to b9859") |

## What this proves / does not prove

- The upstream SHA256 above proves the bytes downloaded from the upstream URL
  matched what was hashed at verification time — i.e. **integrity-after-publication**
  against ggml-org's CI-built release asset. It does **not** prove that asset
  corresponds to a reviewed llama.cpp source revision; that would require a
  full source rebuild, which is explicitly deferred (see
  `docs/LLAMA_CONTRACT.md`, "Full source-rebuild / byte-reproducibility —
  deferred").
- The slim checksum is the value SwiftPM asserts on every `swift package
  resolve` against the hosted `vendor-llama-b9859` release asset, **and it is
  directly independently re-verifiable**: `swift package compute-checksum`
  and `shasum -a 256` are the same algorithm and produce the same hex digest
  for the same file. Anyone can confirm the pin without SwiftPM by running
  `curl -L https://github.com/ManifoldKit/manifold-llama/releases/download/vendor-llama-b9859/llama-b9859-slim.xcframework.zip
  | shasum -a 256` and comparing the result against both the value above and
  `Package.swift`'s pinned `checksum:` — this record was prepared by doing
  exactly that (see below). It will never equal the upstream SHA256 above:
  not because the algorithms differ, but because the two hashes cover
  different files — the upstream 7-slice zip vs. this repackaged 3-slice
  zip. Mislabeling either field, or implying they *should* match, defeats
  the audit this document exists for.
- **Independently reproduced.** While preparing this record,
  `curl -L <the release asset URL above> | shasum -a 256` and a local `swift
  package compute-checksum` on the same downloaded file both returned
  `88e382d47d12e41c786fcf36d5c829c24004d0ee7e3a11483bf4fb63e1d7b190` —
  matching the `Package.swift` pin exactly. So the hosted asset, the pinned
  checksum, and this record all agree.
- **The repackage step itself is not byte-reproducible.** Re-running
  `scripts/repackage-xcframework.sh` for `BUILD=b9859` today produces a
  *different* slim-zip checksum than the one recorded here, because
  `xcodebuild -create-xcframework` + `ditto` commonly embed timestamps in
  the output. This was verified directly while preparing this record (a
  fresh local repackage run produced
  `3bb5b5a040a81e958058f7f53c1c273ab0b0118b417480ecf2fe5bb156b32ca4`, not
  the pinned `88e382d4…`). This does **not** undermine the checksum pin
  above — the *hosted asset* is fixed and directly verifiable regardless of
  whether re-running the script reproduces it; it only means "re-run the
  script and expect the same checksum" is not a valid verification method.
  See `docs/LLAMA_CONTRACT.md`'s "Determinism caveat."
