#!/usr/bin/env bash
# Downloads the built-in detection models into app assets.
# Run once before building (CI runs this automatically).
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)/app/src/main/assets/models"
mkdir -p "$DIR"

BASE="https://storage.googleapis.com/mediapipe-models/object_detector"

download() {
  local name="$1" url="$2"
  if [ -s "$DIR/$name" ]; then
    echo "OK (cached): $name"
    return
  fi
  echo "Downloading $name ..."
  curl -fSL --retry 3 -o "$DIR/$name" "$url"
  echo "OK: $name ($(du -h "$DIR/$name" | cut -f1))"
}

# Fast model (float16 for GPU delegate)
download efficientdet_lite0.tflite \
  "$BASE/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"

# Accurate model
download efficientdet_lite2.tflite \
  "$BASE/efficientdet_lite2/float16/1/efficientdet_lite2.tflite"

echo "All models ready in $DIR"
