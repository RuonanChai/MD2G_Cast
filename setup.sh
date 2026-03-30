#!/usr/bin/env bash
set -euo pipefail

echo "[setup.sh] Preparing MoQ binaries (moq-relay/moq-sub/hang)..."

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
MOQ_DIR="${THIRD_PARTY_DIR}/moq"

mkdir -p "${THIRD_PARTY_DIR}"

# 1) Clone MoQ
if [ ! -d "${MOQ_DIR}/.git" ]; then
  git clone https://github.com/moq-dev/moq.git "${MOQ_DIR}"
fi

cd "${MOQ_DIR}"

# 2) Checkout a specific commit for reproducibility.
#    You can re-run with a newer commit, but the Review Artifact defaults to this tested SHA:
MOQ_TEST_COMMIT="aae43f824b9b8d93b68dd6a765f8bec691f60c0b"
git fetch --all --prune
git checkout "${MOQ_TEST_COMMIT}"

# 3) Build binaries
cargo build --release

BIN_DIR="${MOQ_DIR}/target/release"
echo "[setup.sh] Built binaries should be available under: ${BIN_DIR}"
for b in moq-relay moq-sub hang moq-token; do
  if [ -f "${BIN_DIR}/${b}" ]; then
    echo "[setup.sh] OK: ${b}"
  else
    echo "[setup.sh] NOTE: binary not found (optional depending on MoQ layout): ${b}"
  fi
done

echo "[setup.sh] Done."

cat <<'EOF'

V-PCC layer generation guidance (for full artifact / paper replication)
-----------------------------------------------------------------------
1) Download MPEG V-PCC official test content (e.g., the official "bbb" sequence)
   from the MPEG V-PCC website / official distribution.

2) Generate fragmented Base + Enhanced MP4 layers for MoQ streaming.
   Then place them into:
     - video/base.mp4
     - video/enhanced.mp4

3) Review Artifact quick start:
   this repository already ships the two required fragmented placeholders
   (`video/base.mp4` and `video/enhanced.mp4`), so no FFmpeg generation is needed
   for end-to-end sanity testing.
EOF

