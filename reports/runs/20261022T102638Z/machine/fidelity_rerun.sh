#!/usr/bin/env bash
cd /root/AuthBench
. /root/.local/bin/env
R=reports/runs/20261022T102638Z
echo "[$(date -u +%FT%TZ)] start euler_fidelity configs after commit $(git rev-parse --short HEAD)" >> $R/run.log
began=$(date +%s)
uv run python scripts/euler_fidelity.py data/interim/auth data/raw/redteam.txt.gz $R ntlm ntlm_vocabulary ntlm_code all --seeds 42,43,44,45,46 > $R/euler_fidelity_configs.log 2>&1
rc=$?
echo "[$(date -u +%FT%TZ)] euler_fidelity configs exit $rc in $(($(date +%s) - began)) s" >> $R/run.log
