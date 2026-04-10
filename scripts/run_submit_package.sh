#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-/workspace/configs/submit.yaml}"
OUT_ROOT="/workspace/logs/submit"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[error] config not found: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

echo "[info] config: $CONFIG_PATH"
echo "[info] running full submission generation..."
PYTHONPATH=/workspace:/workspace/src /usr/bin/python /workspace/src/run_video_editor_v2.py --config "$CONFIG_PATH"

latest_dir="$(find "$OUT_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'all_*' -printf '%T@ %p\n' | sort -nr | head -n1 | awk '{print $2}')"
if [[ -z "${latest_dir:-}" ]]; then
  echo "[error] no output directory found under $OUT_ROOT" >&2
  exit 1
fi

zip_path="$latest_dir/submission_mp4.zip"
echo "[info] latest output: $latest_dir"
echo "[info] creating zip: $zip_path"
(
  cd "$latest_dir"
  find . -maxdepth 1 -type f -name '*.mp4' -print | sed 's#^./##' | sort > mp4_list.txt
  zip -q -@ "$zip_path" < mp4_list.txt
)

mp4_count="$(find "$latest_dir" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')"
zip_size="$(du -h "$zip_path" | awk '{print $1}')"

echo "[done] mp4_count=$mp4_count"
echo "[done] zip=$zip_path ($zip_size)"
