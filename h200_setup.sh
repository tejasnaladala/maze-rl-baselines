#!/bin/bash
# H200 one-shot setup + launch script.
# Run this on a freshly-rented 4x H200 vast.ai instance after SSH'ing in.
#
# Usage:
#   bash h200_setup.sh

set -euo pipefail

echo "================================================================"
echo "ENGRAM H200 SETUP — one-shot install + parallel launch"
echo "================================================================"
date
echo

# 1. Sanity check GPUs
echo "[1/6] GPU check"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo

# 2. Clone repo
echo "[2/6] Clone repo"
if [ ! -d engram ]; then
    git clone https://github.com/tejasnaladala/engram.git
fi
cd engram
git pull
echo "Code at commit: $(git rev-parse HEAD)"
echo

# 3. Install dependencies
echo "[3/6] Install Python dependencies"
pip install --quiet --upgrade pip
pip install --quiet \
    'torch>=2.0' \
    'numpy>=1.24' \
    'gymnasium>=0.29' \
    'stable-baselines3>=2.0' \
    'minigrid>=3.0' \
    'procgen' \
    'matplotlib' \
    'scipy' \
    'tqdm'
echo "Installed."
echo

# 4. Smoke test (5 sec)
echo "[4/6] Smoke test"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  [{i}] {torch.cuda.get_device_name(i)}')
import stable_baselines3
print(f'SB3: {stable_baselines3.__version__}')
import minigrid
print(f'MiniGrid: {minigrid.__version__}')
import procgen
print(f'Procgen: imported OK')
from experiment_lib_v2 import code_hash
print(f'Code hash: {code_hash()}')
"
echo

# 5. Launch 4 parallel jobs (one per GPU)
echo "[5/6] Launch 4 parallel jobs (one per GPU)"
mkdir -p logs

# GPU 0: Procgen Maze (heaviest, gets dedicated GPU)
echo "  GPU 0 -> Procgen Maze (5 agents x 20 seeds = 100 runs)"
CUDA_VISIBLE_DEVICES=0 nohup python launch_procgen_maze.py \
    > logs/h200_gpu0_procgen.log 2>&1 &
echo "  -> PID $!"

# GPU 1: SB3 budget-matched (closes A1)
echo "  GPU 1 -> Budget-matched SB3 (3 agents x 3 budgets x 3 sizes x 20 seeds = 540 runs)"
CUDA_VISIBLE_DEVICES=1 nohup python launch_budget_matched_sb3.py \
    > logs/h200_gpu1_sb3.log 2>&1 &
echo "  -> PID $!"

# GPU 2: MiniGrid 4-env (env diversity)
echo "  GPU 2 -> MiniGrid 4-env (4 envs x 5 agents x 20 seeds = 400 runs)"
CUDA_VISIBLE_DEVICES=2 nohup python launch_minigrid.py \
    > logs/h200_gpu2_minigrid.log 2>&1 &
echo "  -> PID $!"

# GPU 3: DRQN multi-scale + LR sweep + Decision Transformer + RND/ICM (light loads chained)
echo "  GPU 3 -> DRQN multi-scale + DT + RND/ICM"
CUDA_VISIBLE_DEVICES=3 nohup bash -c "
    python launch_drqn_multiscale.py 2>&1
    python launch_lr_sweep.py 2>&1
    python launch_decision_transformer.py 2>&1
    python launch_rnd_icm.py 2>&1
" > logs/h200_gpu3_chain.log 2>&1 &
echo "  -> PID $!"

echo
echo "[6/6] Setup complete. Use these to monitor:"
echo "  watch -n 5 'ls raw_results/*/checkpoint.json 2>/dev/null | xargs -I {} bash -c \"echo \\$(basename \\$(dirname {})): \\$(jq length {} 2>/dev/null)\"'"
echo "  tail -f logs/h200_gpu0_procgen.log"
echo "  nvidia-smi"
echo
echo "Estimated wall time: 4-6 hours. Checkpoints survive process death."
echo "After completion, run: python finalize.py"
date
