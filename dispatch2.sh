#!/bin/bash
# Auto-dispatch chain for H200. Only WORKING (post-fix) launchers.
cd /workspace/engram

dispatch() {
  local gpu=$1
  local script=$2
  local logname=$3
  local python_bin=${4:-python3}
  echo "$(date) GPU$gpu -> $script"
  CUDA_VISIBLE_DEVICES=$gpu nohup $python_bin $script > logs/gpu${gpu}_${logname}.log 2>&1 &
  echo "  PID: $!"
}

# Wait for extra_seeds (PID 4608) -> policy_distillation on GPU 3
echo "Waiting for extra_seeds to finish..."
while pgrep -f extra_seeds > /dev/null; do sleep 30; done
dispatch 3 launch_policy_distillation.py policy_distill

# Wait for MiniGrid -> count_based on GPU 1
echo "Waiting for MiniGrid to finish..."
while pgrep -f launch_minigrid > /dev/null; do sleep 30; done
dispatch 1 launch_count_based_exploration.py count_explore

# Wait for DRQN multiscale + kill broken DT/RND chain -> procgen on GPU 2
echo "Waiting for DRQN multi-scale to finish..."
while pgrep -f launch_drqn_multiscale > /dev/null; do sleep 30; done
pkill -f launch_decision_transformer 2>/dev/null || true
pkill -f launch_rnd_icm 2>/dev/null || true
sleep 5
dispatch 2 launch_procgen_p310.py procgen /venv/p310/bin/python

# Wait for policy_distillation -> cross_env on GPU 3
echo "Waiting for policy_distillation to finish..."
while pgrep -f launch_policy_distillation > /dev/null; do sleep 30; done
dispatch 3 launch_cross_env_transfer.py cross_env

# Wait for count_based -> reward sensitivity on GPU 1
echo "Waiting for count_based to finish..."
while pgrep -f launch_count_based > /dev/null; do sleep 30; done
dispatch 1 launch_reward_sensitivity.py reward_sens

# Wait for SB3 (long-running) -> wall_following on GPU 0
echo "Waiting for SB3 to finish..."
while pgrep -f launch_budget_matched > /dev/null; do sleep 60; done
dispatch 0 launch_wall_following.py wall_follow

echo "$(date) Dispatch chain complete."
