#!/usr/bin/env bash
# Download the CountsDiff dataset, checkpoints, and experimental results from Zenodo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"

# Zenodo record for the CountsDiff (ICML 2026) artifact.
# TODO(maintainer): set these once the Zenodo record has been created and uploaded.
ZENODO_RECORD_ID="REPLACE_WITH_ZENODO_RECORD_ID"
ARCHIVE="countsdiff_data.tar.gz"
URL="https://zenodo.org/records/${ZENODO_RECORD_ID}/files/${ARCHIVE}?download=1"
# Integrity check (sha256 of countsdiff_data.tar.gz built 2026-06-25).
SHA256="e77fcec6303eb97fdfa95f320c3766d091f8b82f42f94a6969d617cb6cb34711"

if [[ "$ZENODO_RECORD_ID" == REPLACE_* ]]; then
  echo "ERROR: scripts/download_data.sh is not configured yet." >&2
  echo "       Set ZENODO_RECORD_ID (and optionally SHA256) to the published" >&2
  echo "       Zenodo record for the CountsDiff artifact, then re-run." >&2
  exit 1
fi

mkdir -p "$DATA_DIR"
echo "Downloading ${ARCHIVE} from Zenodo record ${ZENODO_RECORD_ID} -> ${DATA_DIR}"
wget -O "${DATA_DIR}/${ARCHIVE}" "$URL"

if [[ -n "$SHA256" ]]; then
  echo "Verifying checksum..."
  echo "${SHA256}  ${DATA_DIR}/${ARCHIVE}" | sha256sum -c -
fi

echo "Extracting..."
tar -xzf "${DATA_DIR}/${ARCHIVE}" -C "$DATA_DIR"
rm "${DATA_DIR}/${ARCHIVE}"

echo "Done. Data is now in: ${DATA_DIR}"
