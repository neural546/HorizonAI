#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: ./generate.sh \"<prompt>\" [--cfg 7.5] [--frames 8] [--seed 42] [--steps 50]"
    exit 1
fi

PROMPT="$1"; shift
EXTRA=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cfg) EXTRA="$EXTRA --cfg-scale $2"; shift 2 ;;
        --frames) EXTRA="$EXTRA --num-frames $2"; shift 2 ;;
        --seed) EXTRA="$EXTRA --seed $2"; shift 2 ;;
        --steps) EXTRA="$EXTRA --num-inference-steps $2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

python AI/inference.py --prompt "$PROMPT" $EXTRA
