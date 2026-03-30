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
1) Download MPEG V-PCC official test content (e.g., "bbb.mp4") from the MPEG
   V-PCC website / official distribution.

2) Generate Base + Enhanced layers into this repository's "video/" folder.
   This codebase expects fragmented MP4 outputs for MoQ streaming.

   In moq_mininet_eval.py, the default preparation does:
     - Base  : re-encode from OFFICIAL_VIDEO with bitrate ~10 Mbps
     - Enhanced: grayscale/enhanced layer with bitrate ~1 Mbps
     - then outputs to:
         video/redandblack_2/base/redandblack_base_fragmented.mp4
         video/redandblack_2/enhanced/redandblack_enhanced_fragmented.mp4

   If you want to bypass the default generation, replace video/bbb.mp4 with
   your official "bbb.mp4" and keep OFFICIAL_VIDEO pointing to it.

3) For the Review Artifact quick start, we ship a tiny placeholder video
   (video/bbb.mp4) so the end-to-end pipeline can run without large uploads.
EOF

