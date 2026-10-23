#!/usr/bin/env bash
set -uo pipefail
cd /root/euler_original/lanl_experiments
echo "[$(date -u +%FT%TZ)] start split.py"
zcat /root/AuthBench/data/raw/auth.txt.gz > /root/euler_original_data/auth.fifo &
feeder=$!
../.venv/bin/python loaders/split.py
rc=$?
wait $feeder
echo "[$(date -u +%FT%TZ)] split.py exit $rc, feeder exit $?"
