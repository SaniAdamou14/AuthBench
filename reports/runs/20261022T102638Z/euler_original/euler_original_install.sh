#!/usr/bin/env bash
set -euxo pipefail
. /root/.local/bin/env
cd /root/euler_original
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch==1.13.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python torch-scatter==2.1.0 torch-sparse==0.6.16 torch-cluster==1.6.0 --find-links https://data.pyg.org/whl/torch-1.13.0+cpu.html --no-build-isolation
uv pip install --python .venv/bin/python torch-geometric==2.2.0 numpy==1.23.5 pandas==1.4.3 scikit-learn==1.1.1 scipy==1.9.3 joblib==1.1.0 tqdm==4.64.1 psutil==5.9.0
.venv/bin/python -c "import torch, torch_geometric, torch_scatter, torch_sparse; print(torch.__version__, torch_geometric.__version__, torch_scatter.__version__, torch_sparse.__version__)"
