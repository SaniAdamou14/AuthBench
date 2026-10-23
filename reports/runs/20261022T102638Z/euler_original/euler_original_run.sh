#!/usr/bin/env bash
set -uo pipefail
cd /root/euler_original/lanl_experiments
cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
( while sleep 15; do echo "$(date -u +%T) $(($(cat $cg/memory.current) / 1048576)) MiB"; done ) > /root/euler_original_data/memory.log &
sampler=$!
echo "[$(date -u +%FT%TZ)] start run.py at commit $(git rev-parse --short HEAD), paths diff in /root/euler_original_data/paths.diff"
../.venv/bin/python -u run.py -w 4 -t 5 -d 0.5 -e GCN -i PRED
echo "[$(date -u +%FT%TZ)] run.py exit $?"
kill $sampler
