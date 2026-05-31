"""
dqn_nitrogen_v2.py — Improved DQN Nitrogen Management
Based on: "Optimizing Nitrogen Management with Deep Reinforcement Learning and Crop Simulations"
Wu et al., CVPRW 2022

Improvements over v1 (all changes marked with ★):
  1. ★ Double DQN      — decoupled action selection/evaluation; kills Q-value overestimation
  2. ★ Huber loss      — smoother gradients near the optimum; more robust than raw MSE
  3. ★ Gradient clip   — prevents catastrophic weight updates from reward spikes
  4. ★ Soft target upd — Polyak averaging (τ=0.005) instead of hard copy every 10 eps;
                          smoother, more stable target Q values
  5. ★ Warmup phase    — skip learning for the first WARMUP_STEPS to fill the buffer
                          with diverse experiences before any gradient step

Run inside the Docker container (same as before):
    python dqn_nitrogen_v2.py
Outputs:
    dqn_model_v2.pth          — improved model weights (drop into nitrogen_app/ for the website)
    dqn_training_log_v2.json  — training history
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gym
import gym_dssat_pdi  # noqa: F401
import json
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Hyperparameters ───────────────────────────────────────────────────────────
LR             = 5e-5          # Adam learning rate (same as paper)
BATCH_SIZE     = 64
GAMMA          = 0.99
EPS_START      = 1.0
EPS_END        = 0.0
EPS_DECAY      = 0.992         # Iowa decay
# ★ Soft update coefficient (replaces hard TARGET_UPDATE every 10 eps)
TAU            = 0.005         # Polyak: target ← τ·online + (1-τ)·target each step
BUFFER_SIZE    = 50_000
N_EPISODES     = 1200
HIDDEN_SIZE    = 256
# ★ Warmup: collect this many transitions before any gradient update
WARMUP_STEPS   = 1_000
# ★ Gradient clipping max norm
GRAD_CLIP      = 10.0

# Reward weights (paper eq.1)
W1, W2, W3, W4 = 0.1, 0.1, 0.1, 1.0
N_THRESHOLD    = 300.0

ACTIONS        = [0, 40, 80, 120, 160]
N_ACTIONS      = len(ACTIONS)
LOG_FILE       = "dqn_training_log_v2.json"
MODEL_FILE     = "dqn_model_v2.pth"


# ── Observation flattener ─────────────────────────────────────────────────────
def flatten_obs(obs: dict) -> np.ndarray:
    parts = []
    for k, v in sorted(obs.items()):
        if isinstance(v, (int, float, np.integer, np.floating)):
            parts.append(float(v))
        elif isinstance(v, np.ndarray):
            parts.extend(v.flatten().tolist())
        elif isinstance(v, (list, tuple)):
            parts.extend([float(x) for x in v])
        else:
            parts.append(float(v))
    return np.array(parts, dtype=np.float32)


def get_obs_dim(env) -> int:
    obs = env.reset()
    return len(flatten_obs(obs))


# ── Reward shaper (unchanged from v1) ────────────────────────────────────────
class RewardShaper:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_n = 0.0

    def shape(self, raw_reward, obs: dict, action_kg: float, done: bool) -> float:
        if raw_reward is None:
            raw_reward = 0.0
        self.total_n += action_kg
        nl = float(obs.get("tleachd", 0.0)) if "tleachd" in obs else 0.0
        if action_kg > 0 and self.total_n > N_THRESHOLD:
            pt = self.total_n - N_THRESHOLD
        else:
            pt = 0.0
        step_r = -W2 * action_kg - W3 * nl - W4 * pt
        if done:
            yield_val = float(obs.get("topwt", 0.0))
            return W1 * yield_val + step_r
        return step_r


# ── Q-Network (unchanged architecture from v1) ────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ── Replay Buffer (unchanged) ─────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_, done):
        self.buf.append((s, a, r, s_, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s_, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(s_)),
            torch.FloatTensor(d),
        )

    def __len__(self):
        return len(self.buf)


# ── DQN Agent (improved) ──────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, obs_dim: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.q_net      = QNetwork(obs_dim, N_ACTIONS).to(self.device)
        self.target_net = QNetwork(obs_dim, N_ACTIONS).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.buffer    = ReplayBuffer(BUFFER_SIZE)
        self.epsilon   = EPS_START
        self.total_steps = 0   # ★ track total env steps for warmup

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_net(s).argmax(dim=1).item()

    def update(self):
        # ★ Skip gradient updates during warmup
        if len(self.buffer) < BATCH_SIZE or self.total_steps < WARMUP_STEPS:
            return None

        s, a, r, s_, d = self.buffer.sample(BATCH_SIZE)
        s, a, r, s_, d = (t.to(self.device) for t in (s, a, r, s_, d))

        # Current Q values (online network)
        q_vals = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # ★ DOUBLE DQN target computation:
        #   Step 1 — use the ONLINE network to select the best next action
        #   Step 2 — use the TARGET network to EVALUATE that action's Q-value
        #   (v1 used target_net for both selection AND evaluation → overestimates)
        with torch.no_grad():
            # Step 1: online net picks action
            next_actions = self.q_net(s_).argmax(dim=1, keepdim=True)   # ★
            # Step 2: target net evaluates it
            next_q = self.target_net(s_).gather(1, next_actions).squeeze(1)  # ★
            target = r + GAMMA * next_q * (1 - d)

        # ★ Huber loss (SmoothL1) instead of MSE — less sensitive to reward outliers
        loss = nn.SmoothL1Loss()(q_vals, target)   # ★ was: nn.MSELoss()

        self.optimizer.zero_grad()
        loss.backward()
        # ★ Gradient clipping — prevents exploding gradients from large reward spikes
        nn.utils.clip_grad_norm_(self.q_net.parameters(), GRAD_CLIP)   # ★
        self.optimizer.step()

        # ★ Soft target update (Polyak averaging) — replaces hard copy every 10 eps
        #   Runs every training step → smoother, more stable target network
        for target_p, online_p in zip(self.target_net.parameters(),
                                       self.q_net.parameters()):
            target_p.data.copy_(TAU * online_p.data + (1.0 - TAU) * target_p.data)  # ★

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)


# ── Training Loop ─────────────────────────────────────────────────────────────
def train():
    env = gym.make(
        "gym_dssat_pdi:GymDssatPdi-v0",
        mode="fertilization",
        seed=SEED,
    )

    obs_dim = get_obs_dim(env)
    print(f"Observation dim: {obs_dim}  |  Actions: {ACTIONS}")
    print(f"Double DQN: ON | Huber loss: ON | Grad clip: {GRAD_CLIP} | "
          f"Soft τ: {TAU} | Warmup: {WARMUP_STEPS} steps")

    agent  = DQNAgent(obs_dim)
    shaper = RewardShaper()

    log = {
        "episode_rewards":        [],
        "episode_shaped_rewards": [],
        "episode_total_n":        [],
        "episode_yield":          [],
        "epsilon":                [],
    }

    for ep in range(1, N_EPISODES + 1):
        obs            = env.reset()
        last_valid_obs = obs
        state          = flatten_obs(obs)
        shaper.reset()

        ep_raw_r    = 0.0
        ep_shaped_r = 0.0
        ep_yield    = 0.0
        done        = False

        while not done:
            action_idx  = agent.select_action(state)
            action_kg   = float(ACTIONS[action_idx])
            action_dict = {"anfer": action_kg}

            next_obs, raw_r, done, info = env.step(action_dict)
            if next_obs is not None:
                last_valid_obs = next_obs
            next_state = flatten_obs(last_valid_obs)

            shaped_r = shaper.shape(raw_r, last_valid_obs, action_kg, done)
            if raw_r is None:
                raw_r = 0.0

            agent.buffer.push(state, action_idx, shaped_r, next_state, float(done))
            agent.total_steps += 1   # ★
            agent.update()

            ep_raw_r    += raw_r
            ep_shaped_r += shaped_r
            state        = next_state

            if done:
                ep_yield = float(last_valid_obs.get("topwt", 0.0))

        agent.decay_epsilon()
        # ★ No more hard target update here — soft update runs inside agent.update()

        log["episode_rewards"].append(ep_raw_r)
        log["episode_shaped_rewards"].append(ep_shaped_r)
        log["episode_total_n"].append(shaper.total_n)
        log["episode_yield"].append(ep_yield)
        log["epsilon"].append(agent.epsilon)

        if ep % 50 == 0 or ep == 1:
            recent = log["episode_shaped_rewards"][-50:]
            warmup_note = " [WARMUP]" if agent.total_steps < WARMUP_STEPS else ""
            print(
                f"Ep {ep:4d}/{N_EPISODES} | "
                f"ε={agent.epsilon:.3f} | "
                f"ShapedR(last50)={np.mean(recent):8.1f} | "
                f"RawR={ep_raw_r:8.1f} | "
                f"TotalN={shaper.total_n:.0f} kg/ha | "
                f"Yield={ep_yield:.0f} kg/ha{warmup_note}"
            )

    env.close()

    torch.save(agent.q_net.state_dict(), MODEL_FILE)
    with open(LOG_FILE, "w") as f:
        json.dump(log, f)
    print(f"\nModel saved  → {MODEL_FILE}")
    print(f"Training log → {LOG_FILE}")

    return agent, log


# ── Evaluation (unchanged logic) ─────────────────────────────────────────────
def evaluate(agent: DQNAgent, n_eval: int = 5):
    env = gym.make(
        "gym_dssat_pdi:GymDssatPdi-v0",
        mode="fertilization",
        seed=SEED + 100,
    )
    obs_dim  = get_obs_dim(env)
    agent.epsilon = 0.0

    results = []
    shaper  = RewardShaper()

    for ep in range(n_eval):
        obs            = env.reset()
        last_valid_obs = obs
        state          = flatten_obs(obs)
        shaper.reset()
        done     = False
        ep_raw_r = 0.0
        fertilization_schedule = []

        while not done:
            action_idx  = agent.select_action(state)
            action_kg   = float(ACTIONS[action_idx])
            next_obs, raw_r, done, _ = env.step({"anfer": action_kg})
            if next_obs is not None:
                last_valid_obs = next_obs
            if raw_r is None:
                raw_r = 0.0
            ep_raw_r += raw_r
            if action_kg > 0:
                dap = int(last_valid_obs.get("dap", -1))
                fertilization_schedule.append((dap, action_kg))
            state = flatten_obs(last_valid_obs)

        yield_val = float(last_valid_obs.get("topwt", 0.0))
        results.append({
            "ep":         ep + 1,
            "raw_reward": ep_raw_r,
            "total_n":    shaper.total_n,
            "yield_kgha": yield_val,
            "schedule":   fertilization_schedule,
        })

    env.close()

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (greedy Double-DQN policy)")
    print("=" * 60)
    print(f"{'Ep':>4} | {'RawReward':>10} | {'TotalN(kg/ha)':>14} | {'Yield(kg/ha)':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['ep']:>4} | {r['raw_reward']:>10.1f} | "
              f"{r['total_n']:>14.1f} | {r['yield_kgha']:>12.1f}")
    print("-" * 60)
    avg_r = np.mean([r["raw_reward"]  for r in results])
    avg_n = np.mean([r["total_n"]     for r in results])
    avg_y = np.mean([r["yield_kgha"]  for r in results])
    print(f"{'AVG':>4} | {avg_r:>10.1f} | {avg_n:>14.1f} | {avg_y:>12.1f}")
    print("=" * 60)
    print(f"\nBaseline (naive 5 kg/ha every day): ~-263")
    print(f"Double-DQN average raw reward:       {avg_r:.1f}")
    delta = avg_r - (-263)
    print(f"Improvement over baseline:           {delta:+.1f}")

    return results


# ── Baselines ────────────────────────────────────────────────────────────────
def run_baseline(n_kg: float, n_eval: int = 3):
    env = gym.make(
        "gym_dssat_pdi:GymDssatPdi-v0",
        mode="fertilization",
        seed=SEED + 200,
    )
    shaper  = RewardShaper()
    rewards = []

    for _ in range(n_eval):
        obs     = env.reset()
        shaper.reset()
        done    = False
        ep_raw_r = 0.0
        applied = False

        while not done:
            vstage    = float(obs.get("vstage", 0.0))
            action_kg = n_kg if (vstage >= 5.0 and not applied) else 0.0
            if vstage >= 5.0 and not applied:
                applied = True
            next_obs, raw_r, done, _ = env.step({"anfer": action_kg})
            if raw_r is None:
                raw_r = 0.0
            ep_raw_r += raw_r
            obs = next_obs

        rewards.append(ep_raw_r)

    env.close()
    avg = np.mean(rewards)
    print(f"Baseline ({n_kg:.0f} kg/ha at v5): avg raw reward = {avg:.1f}")
    return avg


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Double-DQN Nitrogen Management (improved v2)")
    print("=" * 60)
    print("\nKey improvements over v1:")
    print("  ★ Double DQN     — decoupled selection/evaluation")
    print("  ★ Huber loss     — robust SmoothL1 instead of MSE")
    print("  ★ Grad clipping  — max norm =", GRAD_CLIP)
    print(f"  ★ Soft updates   — Polyak τ = {TAU} every step")
    print(f"  ★ Warmup         — {WARMUP_STEPS} steps before learning")
    print()

    print("--- Paper-style baselines ---")
    for n in [160, 240, 280]:
        run_baseline(float(n), n_eval=3)

    print("\n--- Training Double-DQN ---")
    agent, log = train()

    print("\n--- Evaluating ---")
    evaluate(agent, n_eval=5)
