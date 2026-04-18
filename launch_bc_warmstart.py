"""BC WARM-START RL — disambiguating "exploration vs representation" failure.

==========================================================================
EXPERIMENTAL MOTIVATION (responding to adversarial reviewer R8)
==========================================================================

The headline finding from launch_policy_distillation.py is that an MLP with
the same architecture as MLP_DQN reaches ~97.4% test success when trained
via supervised cross-entropy on BFS oracle action labels — vs ~19.3% when
the same MLP is trained from scratch via DQN on the shaped reward.

A reviewer correctly observed that this only proves the high-performing
policy is a GLOBAL MINIMUM of supervised loss. It does NOT prove that the
distilled basin of attraction is REACHABLE under reward-driven gradient
descent from random initialization, nor whether it is even STABLE under
the DQN loss landscape.

The clean experiment that disambiguates these two hypotheses:

  1. Train an MLP via BC distillation from BFSOracle. (Gives ~97% policy.)
  2. Initialize MLP_DQN's online + target networks with those weights.
  3. Fine-tune via standard DQN (epsilon-greedy, replay, target net) on the
     same shaped reward used by the from-scratch DQN baseline.
  4. Measure final test success rate.

Two possible outcomes, each with a distinct scientific claim:

  - POST-FT PERF STAYS NEAR ~97%:
      The distilled representation is REACHABLE in principle by gradient
      descent on the reward, but standard exploration cannot FIND it from
      random init. The failure is a basin-of-attraction / initialization
      problem, not a representational or reward-incompatibility problem.
      Conclusion: shape priors matter, but the function class is fine.

  - POST-FT PERF COLLAPSES BACK TOWARD ~19%:
      The shaped reward landscape ACTIVELY DESTROYS the distilled
      representation. This is a much stronger negative result: the reward
      itself is misaligned with the supervised optimum, so DQN gradients
      pull the policy *away* from the BFS-optimal manifold. This would
      indicate a reward-design bug or a fundamental incompatibility between
      bootstrapped value learning and the dense shaping signal.

  - INTERMEDIATE / HIGH VARIANCE:
      Mixed signal — the basin is metastable but not robust. Plot
      per-seed trajectories.

==========================================================================
DESIGN DECISIONS
==========================================================================

  * BC distillation is RE-RUN each seed rather than loaded from
    raw_results/exp_policy_distillation/ because the existing distillation
    runs only saved JSON metrics (success_rate, n_demos, etc.) — the model
    state_dicts were not persisted. Retraining is cheap (~30s/seed on GPU)
    and guarantees seed-reproducibility.

  * The BC MLP architecture
        Linear(OBS_DIM, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 4)
    is BIT-IDENTICAL to the MLPDQNAgent.net architecture in
    experiment_lib_v2.py. Warm-start is therefore a direct
    `load_state_dict` — no reshape, no projection, no architectural drift.

  * Both the online net AND the target net are warm-started from the
    distilled weights. (Initializing only the online net would let the
    target's random Q-values dominate the bootstrapped TD target during
    early steps and immediately rip the policy off-manifold.)

  * Fine-tune budget is 200K env steps (vs the ~32K env steps that the
    main sweep uses for MLP_DQN: 100 train eps x ~324 max steps). The
    deliberately larger budget is to give the reward landscape every
    chance to either reinforce the distilled policy (good) or destroy it
    (also informative). The original main-sweep budget of 32K is too short
    to confidently observe collapse.

  * Replay buffer is initialized empty — we do NOT pre-fill it with BC
    demos, because the question is "does ON-POLICY DQN preserve the
    distilled basin", not "does offline-to-online RL with demos work". We
    keep the test as a clean isolation of the reward-vs-supervised
    optimum question.

  * Test phase uses main_sweep_test_seeds() so the success rate is
    directly comparable to MLP_DQN (19.3%) and DistilledMLP (97.4%) from
    prior experiments.

==========================================================================
USAGE
==========================================================================

    python launch_bc_warmstart.py

Outputs to raw_results/exp_bc_warmstart/. Resumable via checkpoint.json.

Per-seed runtime estimate (RTX-class GPU):
  - BC demo collection (BFS, 500 eps):           ~10s
  - BC training (50 epochs on ~9k samples):      ~25s
  - DQN fine-tune (200K env steps):              ~6-9 min
  - Test phase (50 episodes, greedy):            ~2s
  Total per seed:                                ~7-10 min
Total for 5 seeds:                               ~35-50 min
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from experiment_lib_v2 import (  # noqa: E402
    MLPDQNAgent,
    BFSOracleAgent,
    make_maze,
    OBS_DIM,
    NUM_ACTIONS,
    ACTIONS,
    WALL,
    HAZARD,
    load_checkpoint,
    save_checkpoint,
    run_key,
    atomic_save,
    set_all_seeds,
    code_hash,
)
from maze_env_helpers import (  # noqa: E402
    get_obs,
    step_env,
    main_sweep_test_seeds,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 456, 789, 1024]
MAZE_SIZE = 9

# BC distillation hyperparameters (mirror launch_policy_distillation.py)
NUM_DEMO_EPISODES = 500
BC_EPOCHS = 50
BC_BATCH = 256
BC_LR = 1e-3
BC_WD = 1e-4

# DQN fine-tune hyperparameters
FT_BUDGET_STEPS = 200_000          # env steps (smaller than main sweep)
FT_LR = 5e-4                       # matches MLPDQNAgent default
FT_GAMMA = 0.99
FT_EPS_START = 0.20                # lower than usual: distilled policy is
FT_EPS_END = 0.05                  #   already useful, don't trash it
FT_EPS_DECAY = 50_000              # in env steps
FT_TARGET_UPDATE = 300
FT_BUFFER_SIZE = 20_000
FT_BATCH = 64
FT_HIDDEN = 64

# Reward shaping (matches run_experiment defaults)
WALL_BUMP_COST = -0.3
HAZARD_COST = -1.0
GOAL_REWARD = 10.0
STEP_COST = -0.02
SHAPE_COEF_NEAR = 0.08
SHAPE_COEF_FAR = -0.04
VISIT_PENALTY = -0.1

NUM_TEST_EPS = 50
MAX_STEPS_PER_EP = max(300, 4 * MAZE_SIZE * MAZE_SIZE)

OUT_DIR = Path(__file__).parent / "raw_results" / "exp_bc_warmstart"
CHECKPOINT_FILE = OUT_DIR / "checkpoint.json"


# ---------------------------------------------------------------------------
# BC POLICY (architecture is BIT-IDENTICAL to MLPDQNAgent.net)
# ---------------------------------------------------------------------------


class MLPPolicy(nn.Module):
    """Same arch as MLPDQNAgent.net so state_dict transfers directly."""

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = NUM_ACTIONS,
                 hidden: int = FT_HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# BC DEMO COLLECTION + TRAINING
# ---------------------------------------------------------------------------


def collect_bfs_demos(num_episodes: int, maze_size: int,
                      seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Roll BFSOracle on the train-distribution mazes and emit (states, actions)."""
    import random as _random
    train_rng = _random.Random(seed)
    teacher = BFSOracleAgent()

    all_states: list[np.ndarray] = []
    all_actions: list[int] = []
    collected = 0
    attempts = 0
    max_attempts = num_episodes * 10

    while collected < num_episodes and attempts < max_attempts:
        attempts += 1
        s = train_rng.randint(0, 10_000_000)        # train-phase seed_offset=0
        maze = make_maze(maze_size, seed=s)
        ax, ay = 1, 1
        gx, gy = maze_size - 2, maze_size - 2
        teacher.set_env(maze, maze_size, gx, gy)
        if hasattr(teacher, "reset_for_new_maze"):
            teacher.reset_for_new_maze()

        action_hist: list[int] = []
        ep_states: list[np.ndarray] = []
        ep_actions: list[int] = []

        for step in range(MAX_STEPS_PER_EP):
            obs = get_obs(maze, ax, ay, gx, gy, maze_size, action_hist)
            ep_states.append(obs.copy())
            if hasattr(teacher, "eval_action"):
                action = teacher.eval_action(obs)
            else:
                action = teacher.act(obs, step)
            ep_actions.append(int(action))
            new_ax, new_ay, _, _, _ = step_env(maze, ax, ay, action, maze_size)
            ax, ay = new_ax, new_ay
            action_hist.append(int(action))
            if (ax, ay) == (gx, gy):
                all_states.extend(ep_states)
                all_actions.extend(ep_actions)
                collected += 1
                break

    if not all_states:
        return (np.zeros((0, OBS_DIM), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))
    return (np.array(all_states, dtype=np.float32),
            np.array(all_actions, dtype=np.int64))


def train_bc(states: np.ndarray, actions: np.ndarray) -> MLPPolicy:
    """Supervised cross-entropy on BFS demos. Mirrors launch_policy_distillation.train_mlp."""
    model = MLPPolicy().to(DEVICE)
    if len(states) == 0:
        return model
    opt = torch.optim.AdamW(model.parameters(), lr=BC_LR, weight_decay=BC_WD)
    s_t = torch.from_numpy(states).to(DEVICE)
    a_t = torch.from_numpy(actions).to(DEVICE)
    n = len(states)
    for _ in range(BC_EPOCHS):
        idx = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BC_BATCH):
            b = idx[i:i + BC_BATCH]
            logits = model(s_t[b])
            loss = F.cross_entropy(logits, a_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


# ---------------------------------------------------------------------------
# DQN FINE-TUNE WITH STEP BUDGET
# ---------------------------------------------------------------------------


def make_warmstarted_agent(distilled: MLPPolicy) -> MLPDQNAgent:
    """Build an MLPDQNAgent and copy distilled weights into BOTH online and target."""
    agent = MLPDQNAgent(
        hidden=FT_HIDDEN,
        lr=FT_LR,
        gamma=FT_GAMMA,
        eps_start=FT_EPS_START,
        eps_end=FT_EPS_END,
        eps_decay=FT_EPS_DECAY,
        target_update=FT_TARGET_UPDATE,
        buffer_size=FT_BUFFER_SIZE,
        batch_size=FT_BATCH,
        device=DEVICE,
    )
    sd = distilled.net.state_dict()
    agent.net.load_state_dict(sd)
    agent.target.load_state_dict(sd)
    agent.target.requires_grad_(False)
    return agent


def fine_tune_dqn(agent: MLPDQNAgent, budget_steps: int,
                  maze_size: int, seed: int) -> dict:
    """Run epsilon-greedy DQN against the same shaped reward as run_experiment,
    bounded by total env-step budget instead of episode count.

    Returns training-phase telemetry: number of episodes consumed, mean
    train success in the LAST 100 eps, total env steps actually taken.
    """
    import random as _random
    rng = _random.Random(seed)

    total_steps = 0
    episodes = 0
    recent_solved: list[int] = []
    train_returns: list[float] = []

    while total_steps < budget_steps:
        maze_seed = rng.randint(0, 10_000_000)        # train-phase seed_offset=0
        grid = make_maze(maze_size, maze_seed)
        ax, ay = 1, 1
        gx, gy = maze_size - 2, maze_size - 2
        action_hist: list[int] = []
        visited: set[tuple[int, int]] = {(1, 1)}
        ep_reward = 0.0
        solved = False

        for step in range(MAX_STEPS_PER_EP):
            obs = get_obs(grid, ax, ay, gx, gy, maze_size, action_hist)
            action = agent.act(obs, step)
            dx, dy = ACTIONS[action]
            nx, ny = ax + dx, ay + dy
            reward = STEP_COST
            done = False
            prev_dist = abs(ax - gx) + abs(ay - gy)

            if 0 <= nx < maze_size and 0 <= ny < maze_size and grid[ny][nx] != WALL:
                ax, ay = nx, ny
                new_dist = abs(ax - gx) + abs(ay - gy)
                if new_dist < prev_dist:
                    reward += SHAPE_COEF_NEAR
                elif new_dist > prev_dist:
                    reward += SHAPE_COEF_FAR
                if (ax, ay) in visited:
                    reward += VISIT_PENALTY
                visited.add((ax, ay))
                if grid[ay][ax] == HAZARD:
                    reward = HAZARD_COST
                if ax == gx and ay == gy:
                    reward = GOAL_REWARD
                    done = True
                    solved = True
            else:
                reward = WALL_BUMP_COST

            next_obs = get_obs(grid, ax, ay, gx, gy, maze_size,
                               action_hist + [action])
            agent.learn(obs, action, reward, next_obs, done)

            ep_reward += reward
            action_hist.append(int(action))
            total_steps += 1

            if done or total_steps >= budget_steps:
                break

        episodes += 1
        train_returns.append(ep_reward)
        recent_solved.append(int(solved))
        if len(recent_solved) > 100:
            recent_solved.pop(0)

    last_100_success = (sum(recent_solved) / len(recent_solved)
                        if recent_solved else 0.0)
    return {
        "ft_episodes": episodes,
        "ft_env_steps": total_steps,
        "ft_last100_success": last_100_success,
        "ft_mean_return": float(np.mean(train_returns)) if train_returns else 0.0,
    }


def evaluate_greedy(agent: MLPDQNAgent, n_test: int, seed: int) -> dict:
    """Greedy eval on main_sweep_test_seeds — directly comparable to MLP_DQN
    (19.3%) and DistilledMLP (97.4%)."""
    agent.net.train(False)
    solved = 0
    total_steps = 0
    test_seeds = main_sweep_test_seeds(seed, n_test)

    for s in test_seeds:
        maze = make_maze(MAZE_SIZE, seed=s)
        ax, ay = 1, 1
        gx, gy = MAZE_SIZE - 2, MAZE_SIZE - 2
        action_hist: list[int] = []
        step = 0
        for step in range(MAX_STEPS_PER_EP):
            obs = get_obs(maze, ax, ay, gx, gy, MAZE_SIZE, action_hist)
            action = agent.eval_action(obs)
            new_ax, new_ay, _, _, _ = step_env(maze, ax, ay, action, MAZE_SIZE)
            ax, ay = new_ax, new_ay
            action_hist.append(int(action))
            if (ax, ay) == (gx, gy):
                solved += 1
                break
        total_steps += step + 1

    agent.net.train(True)
    return {
        "test_n_eps": n_test,
        "test_solved": solved,
        "test_success_rate": solved / n_test,
        "test_mean_steps": total_steps / n_test,
    }


def evaluate_distilled_only(distilled: MLPPolicy, n_test: int,
                            seed: int) -> dict:
    """Sanity-check the BC policy BEFORE fine-tune (should be ~97%)."""
    distilled.train(False)
    solved = 0
    total_steps = 0
    test_seeds = main_sweep_test_seeds(seed, n_test)
    for s in test_seeds:
        maze = make_maze(MAZE_SIZE, seed=s)
        ax, ay = 1, 1
        gx, gy = MAZE_SIZE - 2, MAZE_SIZE - 2
        action_hist: list[int] = []
        step = 0
        for step in range(MAX_STEPS_PER_EP):
            obs = get_obs(maze, ax, ay, gx, gy, MAZE_SIZE, action_hist)
            with torch.no_grad():
                logits = distilled(
                    torch.from_numpy(obs).float().to(DEVICE).unsqueeze(0))
                action = int(logits.argmax(dim=-1).item())
            new_ax, new_ay, _, _, _ = step_env(maze, ax, ay, action, MAZE_SIZE)
            ax, ay = new_ax, new_ay
            action_hist.append(int(action))
            if (ax, ay) == (gx, gy):
                solved += 1
                break
        total_steps += step + 1
    return {
        "bc_n_eps": n_test,
        "bc_solved": solved,
        "bc_success_rate": solved / n_test,
        "bc_mean_steps": total_steps / n_test,
    }


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_checkpoint(CHECKPOINT_FILE)

    agent_name = "BCWarmStart_MLP_DQN"
    total = len(SEEDS)
    done = len(completed)
    print(f"\nBC Warm-Start RL: {total} runs, {done} done")
    print(f"Seeds: {SEEDS}")
    print(f"Maze size: {MAZE_SIZE}x{MAZE_SIZE}")
    print(f"BC: {NUM_DEMO_EPISODES} demos, {BC_EPOCHS} epochs")
    print(f"Fine-tune: {FT_BUDGET_STEPS:,} env steps "
          f"(eps {FT_EPS_START}->{FT_EPS_END} over {FT_EPS_DECAY:,} steps)")
    print(f"Test: {NUM_TEST_EPS} eps on main_sweep_test_seeds")
    print(f"Code hash: {code_hash()}\n")

    for seed in SEEDS:
        key = run_key(agent_name, MAZE_SIZE, seed)
        if key in completed:
            print(f"  [skip] {agent_name} s={seed} (already complete)")
            continue

        print(f"  [{done}/{total}] {agent_name} s={seed} ...", flush=True)
        t0 = time.time()
        set_all_seeds(seed, deterministic=False)

        # 1. Collect BFS demos
        t_demo = time.time()
        states, actions = collect_bfs_demos(NUM_DEMO_EPISODES, MAZE_SIZE, seed)
        n_demos = len(states)
        print(f"      demos: {n_demos:,} state-action pairs "
              f"({time.time() - t_demo:.1f}s)", flush=True)

        # 2. BC train the distilled policy
        t_bc = time.time()
        distilled = train_bc(states, actions)
        bc_eval = evaluate_distilled_only(distilled, NUM_TEST_EPS, seed)
        print(f"      BC done ({time.time() - t_bc:.1f}s)  "
              f"BC test={100 * bc_eval['bc_success_rate']:.1f}%", flush=True)

        # 3. Warm-start MLP_DQN with distilled weights
        agent = make_warmstarted_agent(distilled)

        # 4. Fine-tune via DQN with the shaped reward
        t_ft = time.time()
        ft_stats = fine_tune_dqn(agent, FT_BUDGET_STEPS, MAZE_SIZE, seed)
        print(f"      FT done ({time.time() - t_ft:.0f}s)  "
              f"eps={ft_stats['ft_episodes']:,} "
              f"steps={ft_stats['ft_env_steps']:,} "
              f"last100={100 * ft_stats['ft_last100_success']:.1f}%",
              flush=True)

        # 5. Final greedy evaluation on main-sweep test seeds
        t_test = time.time()
        test_eval = evaluate_greedy(agent, NUM_TEST_EPS, seed)
        print(f"      TEST ({time.time() - t_test:.1f}s)  "
              f"final={100 * test_eval['test_success_rate']:.1f}%",
              flush=True)

        elapsed = time.time() - t0
        run_file = OUT_DIR / f"{agent_name}_{MAZE_SIZE}_{seed}.json"
        atomic_save([{
            "agent_name": agent_name,
            "experiment": "bc_warmstart",
            "maze_size": MAZE_SIZE,
            "seed": seed,
            "phase": "test",
            "wall_time_s": elapsed,
            "code_hash": code_hash(),
            "n_demonstrations": n_demos,
            "n_demo_eps_target": NUM_DEMO_EPISODES,
            "config": {
                "bc_epochs": BC_EPOCHS,
                "bc_batch": BC_BATCH,
                "bc_lr": BC_LR,
                "ft_budget_steps": FT_BUDGET_STEPS,
                "ft_lr": FT_LR,
                "ft_gamma": FT_GAMMA,
                "ft_eps_start": FT_EPS_START,
                "ft_eps_end": FT_EPS_END,
                "ft_eps_decay": FT_EPS_DECAY,
                "ft_target_update": FT_TARGET_UPDATE,
                "ft_buffer_size": FT_BUFFER_SIZE,
                "ft_batch": FT_BATCH,
                "ft_hidden": FT_HIDDEN,
                "wall_bump_cost": WALL_BUMP_COST,
                "hazard_cost": HAZARD_COST,
                "goal_reward": GOAL_REWARD,
            },
            **bc_eval,
            **ft_stats,
            **test_eval,
            # convenience aliases for downstream analysis
            "n_eps": test_eval["test_n_eps"],
            "solved": test_eval["test_solved"],
            "success_rate": test_eval["test_success_rate"],
            "mean_steps": test_eval["test_mean_steps"],
        }], run_file)

        completed.add(key)
        save_checkpoint(CHECKPOINT_FILE, completed)
        done += 1

        delta = test_eval["test_success_rate"] - bc_eval["bc_success_rate"]
        verdict = ("PRESERVED" if delta >= -0.05
                   else "DEGRADED" if delta >= -0.30
                   else "COLLAPSED")
        print(f"      seed {seed} TOTAL {elapsed:.0f}s   "
              f"BC={100 * bc_eval['bc_success_rate']:.1f}% -> "
              f"FT={100 * test_eval['test_success_rate']:.1f}%   "
              f"[{verdict}]\n", flush=True)

    print(f"\nBC warm-start sweep complete. Results in {OUT_DIR}")
    print("Aggregate post-hoc with: python analyze_*.py exp_bc_warmstart")


if __name__ == "__main__":
    main()
