#!/usr/bin/env bash
# Waits for both AuthBench evaluations to finish, then runs the authors code alone.
while pgrep -f "[a]uthbench.pipeline.train_eval" > /dev/null; do sleep 60; done
echo "[$(date -u +%FT%TZ)] AuthBench evaluations finished; starting the authors code"
exec bash /root/euler_original_run.sh
