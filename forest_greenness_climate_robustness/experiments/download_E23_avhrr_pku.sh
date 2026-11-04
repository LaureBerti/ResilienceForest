#!/usr/bin/env bash
set -euo pipefail
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw/avhrr_pku"
mkdir -p "$DEST"
BASE="https://zenodo.org/records/8253971/files"
for f in 1982_1990 1991_2000 2001_2010 2011_2015; do
  out="$DEST/PKU_GIMMS_NDVI_AVHRR_solely_$f.zip"
  [ -s "$out" ] && { echo "have $out"; continue; }
  curl -fSL --retry 3 -o "$out" "$BASE/PKU_GIMMS_NDVI_AVHRR_solely_$f.zip?download=1"
done
if [ "${1:-}" = "--modis" ]; then
  for f in 2001_2010 2011_2022; do
    out="$DEST/PKU_GIMMS_NDVI_AVHRR_MODIS_consolidated_$f.zip"
    [ -s "$out" ] && { echo "have $out"; continue; }
    curl -fSL --retry 3 -o "$out" "$BASE/PKU_GIMMS_NDVI_AVHRR_MODIS_consolidated_$f.zip?download=1"
  done
fi
curl -fSL -o "$DEST/Readme_PKU_GIMMS_NDVI.pdf" \
  "$BASE/Readme_for_PKU_GIMMS_NDVI_Product_updated_20230817.pdf?download=1"
ls -la "$DEST"
