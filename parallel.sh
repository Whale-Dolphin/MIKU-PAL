#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 input_list_file [stage] [stop_stage]"
    exit 1
fi

input_list="$1"
stage="${2:-0}"
stop_stage="${3:-3}"

if [ ! -f "$input_list" ]; then
    echo "Error: $input_list not found"
    exit 1
fi

files=()
while IFS= read -r input_dir; do
    [ -z "$input_dir" ] && continue
    if [ ! -d "$input_dir" ]; then
        echo "Error: Input directory $input_dir not found"
        exit 1
    fi
    while IFS= read -r -d '' file; do
        files+=("$file")
    done < <(find "$input_dir" -type f -print0)
done < "$input_list"

total_files=${#files[@]}
if [ $total_files -eq 0 ]; then
    echo "Error: No files found in input directories"
    exit 1
fi

num_groups=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
per_group=$(( (total_files + num_groups - 1) / num_groups ))

temp_dirs=()
cleanup() {
    for dir in "${temp_dirs[@]}"; do
        rm -rf "$dir"
    done
}
trap cleanup EXIT

for ((group=0; group<num_groups; group++)); do
    start=$((group * per_group))
    end=$((start + per_group))
    (( end > total_files )) && end=$total_files
    (( start >= end )) && continue

    temp_dir=$(mktemp -d)
    temp_dirs+=("$temp_dir")

    for ((i=start; i<end; i++)); do
        file="${files[i]}"
        ln -s "$file" "$temp_dir/$(basename "$file")"
    done

    CUDA_VISIBLE_DEVICES=$group python pipeline.py \
        --input-dir "$temp_dir" \
        --output-dir output \
        --stage "$stage" \
        --stop-stage "$stop_stage" &
done

wait