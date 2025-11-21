#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"

mkdir -p "$DATA_DIR"

echo "Downloading dataset to: $DATA_DIR"
wget -O "$DATA_DIR/data.tar.gz" \
  "https://countsdiff-iclr-artifacts.s3.amazonaws.com/countsdiff_data.tar.gz"

echo "Extracting..."
tar -xvzf "$DATA_DIR/data.tar.gz" -C "$DATA_DIR"
rm "$DATA_DIR/data.tar.gz"

echo "Done. Data is now in: $DATA_DIR"
