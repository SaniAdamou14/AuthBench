#!/usr/bin/env bash
# One reproducible LANL run on a rented machine (S6-15), from a clean checkout to an archive.
#
# Prerequisites on the machine: git, curl, and uv (https://docs.astral.sh/uv/). The download
# happens on the machine itself. Required: AUTHBENCH_LANL_EMAIL, the address LANL's data-use
# form asks for. Optional: HOURLY_EUR, the machine's price per hour, for the cost line (default
# 0.4431, Hetzner CCX43 as quoted in docs/sprints/S6_decembre_2026.md).
#
#   AUTHBENCH_LANL_EMAIL=you@example.org bash scripts/run_lanl.sh
#
# Steps: locked install, preflight, download, conversion of all 58 days, then for each protocol
# (`temporal`, the published one, and `harmonised`, Larroche's) the feature store and the
# evaluation with the autoencoders and Euler, then Euler's fidelity check. Every step's output,
# the machine, the commit, the durations and the cost go under reports/runs/<run id>/, which is
# archived at the end.

set -euo pipefail

: "${AUTHBENCH_LANL_EMAIL:?set AUTHBENCH_LANL_EMAIL to the address the LANL data-use form requires}"
HOURLY_EUR="${HOURLY_EUR:-0.4431}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="reports/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
START="$(date +%s)"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "${RUN_DIR}/run.log"; }

step() {
    local name="$1"
    shift
    local began
    began="$(date +%s)"
    log "start ${name}: $*"
    "$@" 2>&1 | tee "${RUN_DIR}/${name}.log"
    log "done ${name} in $(($(date +%s) - began)) s"
}

{
    echo "run_id: ${RUN_ID}"
    echo "commit: $(git rev-parse HEAD)"
    echo "dirty: $(git status --porcelain | wc -l) changed paths"
    uname -a
    echo "cpus: $(nproc)"
    free -g
    df -h .
} > "${RUN_DIR}/machine.txt"

step install uv sync --extra classical --extra deep --extra tracking --extra dev --locked --python 3.11
step preflight uv run authbench preflight
step download uv run authbench data download --dataset lanl
step convert uv run authbench data to-parquet data/raw/auth.txt.gz --out-dir data/interim/auth --dataset lanl

for split in temporal harmonised; do
    step "features_${split}" uv run python -m authbench.pipeline.build_features "split=${split}"
    step "evaluate_${split}" uv run python -m authbench.pipeline.train_eval "split=${split}" \
        runtime.deep_models=true runtime.euler=true \
        "paths.reports_dir=${RUN_DIR}/${split}" \
        "paths.tables_dir=${RUN_DIR}/${split}/tables" \
        "paths.figures_dir=${RUN_DIR}/${split}/figures"
done

step euler_fidelity uv run python scripts/euler_fidelity.py data/interim/auth data/raw/redteam.txt.gz "${RUN_DIR}"

SECONDS_TOTAL="$(($(date +%s) - START))"
HOURS="$(awk "BEGIN { printf \"%.2f\", ${SECONDS_TOTAL} / 3600 }")"
COST="$(awk "BEGIN { printf \"%.2f\", ${SECONDS_TOTAL} / 3600 * ${HOURLY_EUR} }")"
cat > "${RUN_DIR}/run.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "commit": "$(git rev-parse HEAD)",
  "seconds": ${SECONDS_TOTAL},
  "hours": ${HOURS},
  "hourly_eur": ${HOURLY_EUR},
  "cost_eur": ${COST}
}
EOF
log "run finished in ${HOURS} h, about ${COST} EUR at ${HOURLY_EUR} EUR/h"
tar -czf "reports/runs/${RUN_ID}.tar.gz" -C reports/runs "${RUN_ID}"
log "archived to reports/runs/${RUN_ID}.tar.gz"
