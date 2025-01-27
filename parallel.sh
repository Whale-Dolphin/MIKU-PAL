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

line_num=0
while IFS= read -r input_dir; do
    [ -z "$input_dir" ] && continue
    GPU_ID=$((line_num % 8))
    CUDA_VISIBLE_DEVICES=$GPU_ID python pipeline.py --input_dir "$input_dir" --output_dir output --stage "$stage" --stop-stage "$stop_stage"&
    ((line_num++))
    sleep 10
done < "$input_list"

wait
