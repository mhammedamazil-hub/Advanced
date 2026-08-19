#!/usr/bin/env bash
# Downloads the built-in detection models into app assets.
# Run once before building (CI runs this automatically).
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)/app/src/main/assets/models"
mkdir -p "$DIR"

BASE="https://storage.googleapis.com/mediapipe-models/object_detector"

# download <output-name> <model-dir> — tries float16, then float32, then int8
download() {
  local name="$1" model="$2"
  if [ -s "$DIR/$name" ]; then
    echo "OK (cached): $name"
    return
  fi
  for fmt in float16 float32 int8; do
    echo "Trying $model ($fmt) ..."
    if curl -fSL --retry 3 -o "$DIR/$name" "$BASE/$model/$fmt/1/$name"; then
      echo "OK: $name [$fmt] ($(du -h "$DIR/$name" | cut -f1))"
      return
    fi
  done
  echo "ERROR: could not download $name" >&2
  exit 1
}

download efficientdet_lite0.tflite efficientdet_lite0   # fast model
download efficientdet_lite2.tflite efficientdet_lite2   # accurate model

echo "All models ready in $DIR"
