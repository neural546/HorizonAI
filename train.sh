#!/usr/bin/env bash
set -euo pipefail

PROFILE="low"
AUTO_PROFILE=""
RESUME=""
SEED="42"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) PROFILE="$2"; shift 2 ;;
        --auto-profile) AUTO_PROFILE="--auto-profile"; shift ;;
        --resume) RESUME="--resume $2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ARGS="--seed $SEED"
if [ -n "$AUTO_PROFILE" ]; then
    ARGS="$ARGS $AUTO_PROFILE"
else
    ARGS="$ARGS --profile $PROFILE"
fi
[ -n "$RESUME" ] && ARGS="$ARGS $RESUME"

echo "Starting training: python AI/train.py $ARGS"
python AI/train.py $ARGS
