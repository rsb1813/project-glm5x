#!/bin/sh
# 전체 GLM5X 변환 완료 후 번들 조립과 CUDA reference 게이트를 자동 실행합니다.
set -eu

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
output_dir="$repo_root/build-glm5x-full-k3x"
source_dir="$repo_root/build-glm5x-full-source"
bundle_path="$output_dir/glm5x-experts-full.json"
stream_report="$repo_root/build-glm5x-full-stream-final.json"
cold_cuda_report="$repo_root/build-glm5x-full-reference-cuda-cold.json"
cached_cuda_report="$repo_root/build-glm5x-full-reference-cuda-cached.json"
expected_shards="${EXPECTED_SHARDS:-282}"
poll_seconds="${POLL_SECONDS:-60}"
expert_load_workers="${EXPERT_LOAD_WORKERS:-16}"
if [ -n "${K3X_PYTHON:-}" ]; then
  python_bin="$K3X_PYTHON"
elif [ -x /home/jolib/.venvs/k3x-m1/bin/python ]; then
  python_bin="/home/jolib/.venvs/k3x-m1/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="$(command -v python3)"
else
  python_bin="$(command -v python)"
fi

if [ ! -d "$output_dir" ] || [ ! -d "$source_dir" ]; then
  echo "GLM5X_FULL_GATE_INPUT_DIRECTORY_MISSING" >&2
  exit 2
fi

while :; do
  completed="$(find "$output_dir" -maxdepth 1 -type f -name '*.source-deleted.json' -printf '.' | wc -c)"
  printf '%s markers=%s/%s\n' "$(date --iso-8601=seconds)" "$completed" "$expected_shards"
  if [ "$completed" -ge "$expected_shards" ]; then
    break
  fi
  sleep "$poll_seconds"
done

cd "$repo_root"
export PYTHONPATH="reference:converter:."

stream_partial="${stream_report}.partial"
"$python_bin" tools/stream_glm5x_checkpoint.py \
  --source-dir "$source_dir" \
  --output-dir "$output_dir" \
  --bundle "$bundle_path" \
  > "$stream_partial"
mv -f "$stream_partial" "$stream_report"

cuda_partial="${cold_cuda_report}.partial"
"$python_bin" tools/benchmark_glm5x_reference.py \
  --bundle "$bundle_path" \
  --config "$source_dir/config.json" \
  --prompt 0 \
  --new-tokens 1 \
  --device cuda \
  --expert-load-workers "$expert_load_workers" \
  --expert-cache-bytes 0 \
  --expert-device-cache-bytes 0 \
  --lazy-bundle \
  > "$cuda_partial"
mv -f "$cuda_partial" "$cold_cuda_report"

cuda_partial="${cached_cuda_report}.partial"
"$python_bin" tools/benchmark_glm5x_reference.py \
  --bundle "$bundle_path" \
  --config "$source_dir/config.json" \
  --prompt 0 \
  --new-tokens 2 \
  --device cuda \
  --expert-load-workers "$expert_load_workers" \
  --expert-cache-bytes 8589934592 \
  --expert-device-cache-bytes 4294967296 \
  --lazy-bundle \
  > "$cuda_partial"
mv -f "$cuda_partial" "$cached_cuda_report"

printf '%s full gate completed\n' "$(date --iso-8601=seconds)"
