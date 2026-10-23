set -euo pipefail
. $HOME/.local/bin/env
cd /root/AuthBench
O="split=temporal split.train_days=[0,0] split.val_days=[1,1] split.test_days=[2,2] features.universe=remote_logon paths.processed_dir=/root/smoke/processed paths.tables_dir=/root/smoke/tables paths.figures_dir=/root/smoke/figures paths.reports_dir=/root/smoke eval.fit_variance.seeds=[42,7]"
date -u
uv run python -m authbench.pipeline.build_features $O
date -u
uv run python -m authbench.pipeline.train_eval $O runtime.deep_models=true runtime.euler=true
date -u
