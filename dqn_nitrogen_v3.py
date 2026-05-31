"""
dqn_nitrogen_v3.py — Rainbow-style DQN Nitrogen Management
Based on: "Optimizing Nitrogen Management with Deep Reinforcement Learning and Crop Simulations"
Wu et al., CVPRW 2022

Improvements stacked on v2:
  v1  — Baseline DQN (paper implementation)
  v2  — ★ Double DQN  + Huber loss + Grad clipping + Soft target updates + Warmup
  v3  — ★★ Dueling Architecture  (value + advantage streams)
        ★★ N-step Returns        (n=5, faster credit assignment from harvest reward)

Run inside Docker container:
    python dqn_nitrogen_v3.py
Outputs:
    dqn_model_v3.pth
    dqn_training_log_v3.json
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

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Hyperparameters ───────────────────────────────────────────────────────────
LR            = 5e-5
BATCH_SIZE    = 64
GAMMA         = 0.99
EPS_START     = 1.0
EPS_END       = 0.0
EPS_DECAY     = 0.992
TAU           = 0.005          # soft target update (from v2)
BUFFER_SIZE   = 50_000
N_EPISODES    = 1200
HIDDEN_SIZE   = 256
WARMUP_STEPS  = 1_000          # (from v2)
GRAD_CLIP     = 10.0           # (from v2)
# ★★ N-step return horizon
N_STEPS       = 5
GAMMA_N       = GAMMA ** N_STEPS   # γ^5 — discount for bootstrap term

# Reward weights (paper eq.1)
W1, W2, W3, W4 = 0.1, 0.1, 0.1, 1.0
N_THRESHOLD   = 300.0

ACTIONS       = [0, 40, 80, 120, 160]
N_ACTIONS     = len(ACTIONS)
LOG_FILE      = "dqn_training_log_v3.json"
MODEL_FILE    = "dqn_model_v3.pth"


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


# ── Reward shaper (unchanged) ─────────────────────────────────────────────────
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
        pt = max(0.0, self.total_n - N_THRESHOLD) if action_kg > 0 else 0.0
        step_r = -W2 * action_kg - W3 * nl - W4 * pt
        if done:
            yield_val = float(obs.get("topwt", 0.0))
            return W1 * yield_val + step_r
        return step_r


# ══════════════════════════════════════════════════════════════════════════════
# ★★ DUELING Q-NETWORK
# ══════════════════════════════════════════════════════════════════════════════
class DuelingQNetwork(nn.Module):
    """
    Dueling architecture (Wang et al., 2016).

    Instead of directly estimating Q(s,a), the network learns:
        V(s)    — scalar: how valuable is this crop state?
        A(s,a)  — vector: how much better is each action relative to average?

    Combined as:  Q(s,a) = V(s) + [ A(s,a) - mean_a(A(s,a)) ]

    Why subtracting the mean?  Identifiability — without it, any constant could
    be absorbed between V and A. Subtracting the mean forces V to represent the
    true average value of the state.

    Why does this help for nitrogen management?
      - On ~130/158 days the optimal action is 0 kg/ha regardless of exact state.
      - V(s) learns "crop health" independently; A(s,a) activates only when
        the action choice actually matters (N-stress dropping, tasseling, etc.)
      - The model stops wasting capacity ranking 5 actions when 1 is always right.
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = HIDDEN_SIZE):
        super().__init__()

        # ── Shared feature extractor (same depth as v1/v2) ───────────────────
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
        )

        # ── Value stream: estimates V(s) ─────────────────────────────────────
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),           # scalar output
        )

        # ── Advantage stream: estimates A(s,a) for each action ───────────────
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),   # one output per action
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        v    = self.value_stream(feat)                          # (B, 1)
        a    = self.advantage_stream(feat)                      # (B, n_actions)
        # Combine: subtract mean advantage for stability
        q    = v + (a - a.mean(dim=1, keepdim=True))           # (B, n_actions)
        return q


# ══════════════════════════════════════════════════════════════════════════════
# ★★ N-STEP REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════════════
class NStepReplayBuffer:
    """
    Replay buffer with built-in n-step return computation.

    Standard 1-step buffer stores:  (s, a, r, s', done)
    This buffer stores:             (s, a, G_n, s_n, done_n)

    where:
        G_n    = r_t + γ·r_{t+1} + γ²·r_{t+2} + … + γ^{n-1}·r_{t+n-1}
        s_n    = state n steps later  (or last state if episode ended early)
        done_n = whether a terminal was hit within the n steps

    During learning the target becomes:
        target = G_n  +  γⁿ · Q_target(s_n) · (1 − done_n)

    Why does this help?
        Yield arrives only on Day 158. With 1-step TD, the reward must hop
        backwards through 158 bootstrapping steps, getting diluted each time.
        With n=5, the harvest signal reaches earlier states 5× faster, and
        mid-season N-stress signals connect to consequences much more clearly.
    """

    def __init__(self, capacity: int, n_steps: int, gamma: float):
        self.main_buf = deque(maxlen=capacity)  # stores ready n-step transitions
        self.n        = n_steps
        self.gamma    = gamma
        self._window  = deque()                 # rolling window of raw transitions

    # ── Public API ────────────────────────────────────────────────────────────

    def push(self, s, a, r, s_next, done: bool):
        """Add a raw transition. Internally converts to n-step and stores."""
        self._window.append((s, a, r, s_next, done))

        # Once we have a full n-step window, emit the oldest transition
        if len(self._window) == self.n:
            self._emit_front()

        # At episode end, flush whatever remains in the window
        if done:
            while self._window:
                self._emit_front()

    def sample(self, batch_size: int):
        batch = random.sample(self.main_buf, batch_size)
        s, a, g, s_n, d_n = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(a),
            torch.FloatTensor(g),       # n-step return G_n
            torch.FloatTensor(np.array(s_n)),
            torch.FloatTensor(d_n),
        )

    def __len__(self):
        return len(self.main_buf)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _emit_front(self):
        """Compute n-step return for the oldest transition and store it."""
        G            = 0.0
        bootstrap_s  = self._window[-1][3]   # default: last next_state in window
        bootstrap_done = False

        for i, (_, _, ri, si_next, di) in enumerate(self._window):
            G += (self.gamma ** i) * ri
            # If we hit a terminal, stop accumulating and use that next_state
            if di:
                bootstrap_s    = si_next
                bootstrap_done = True
                break

        s0, a0 = self._window[0][0], self._window[0][1]
        self.main_buf.append((s0, a0, G, bootstrap_s, float(bootstrap_done)))
        self._window.popleft()


# ══════════════════════════════════════════════════════════════════════════════
# DQN Agent (Double DQN + Dueling + N-step)
# ══════════════════════════════════════════════════════════════════════════════
class DQNAgent:
    def __init__(self, obs_dim: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # ★★ Swap plain QNetwork → DuelingQNetwork
        self.q_net      = DuelingQNetwork(obs_dim, N_ACTIONS).to(self.device)
        self.target_net = DuelingQNetwork(obs_dim, N_ACTIONS).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer   = optim.Adam(self.q_net.parameters(), lr=LR)
        # ★★ Swap plain ReplayBuffer → NStepReplayBuffer
        self.buffer      = NStepReplayBuffer(BUFFER_SIZE, N_STEPS, GAMMA)
        self.epsilon     = EPS_START
        self.total_steps = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_net(s).argmax(dim=1).item()

    def update(self):
        if len(self.buffer) < BATCH_SIZE or self.total_steps < WARMUP_STEPS:
            return None

        s, a, g, s_n, d_n = self.buffer.sample(BATCH_SIZE)
        s, a, g, s_n, d_n = (t.to(self.device) for t in (s, a, g, s_n, d_n))

        # Current Q values from online network
        q_vals = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # ★ Double DQN  +  ★★ N-step bootstrap
        with torch.no_grad():
            # Online net selects best action at s_n (Double DQN)
            next_actions = self.q_net(s_n).argmax(dim=1, keepdim=True)
            # Target net evaluates that action
            next_q = self.target_net(s_n).gather(1, next_actions).squeeze(1)
            # ★★ Use GAMMA_N (γ^n) instead of γ^1 for the bootstrap term
            target = g + GAMMA_N * next_q * (1 - d_n)

        # ★ Huber loss
        loss = nn.SmoothL1Loss()(q_vals, target)

        self.optimizer.zero_grad()
        loss.backward()
        # ★ Gradient clipping
        nn.utils.clip_grad_norm_(self.q_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

        # ★ Soft target update (Polyak)
        for tp, op in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(TAU * op.data + (1.0 - TAU) * tp.data)

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
    print(f"Dueling DQN: ON | Double DQN: ON | N-step: {N_STEPS} | "
          f"γⁿ={GAMMA_N:.4f} | Warmup: {WARMUP_STEPS} | τ={TAU}")

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

            next_obs, raw_r, done, _ = env.step({"anfer": action_kg})
            if next_obs is not None:
                last_valid_obs = next_obs
            next_state = flatten_obs(last_valid_obs)

            shaped_r = shaper.shape(raw_r, last_valid_obs, action_kg, done)
            if raw_r is None:
                raw_r = 0.0

            # ★★ Push to n-step buffer (handles windowing internally)
            agent.buffer.push(state, action_idx, shaped_r, next_state, done)
            agent.total_steps += 1
            agent.update()

            ep_raw_r    += raw_r
            ep_shaped_r += shaped_r
            state        = next_state

            if done:
                ep_yield = float(last_valid_obs.get("topwt", 0.0))

        agent.decay_epsilon()

        log["episode_rewards"].append(ep_raw_r)
        log["episode_shaped_rewards"].append(ep_shaped_r)
        log["episode_total_n"].append(shaper.total_n)
        log["episode_yield"].append(ep_yield)
        log["epsilon"].append(agent.epsilon)

        if ep % 50 == 0 or ep == 1:
            recent      = log["episode_shaped_rewards"][-50:]
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
    import os
    torch.save(agent.q_net.state_dict(), MODEL_FILE)
    with open(LOG_FILE, "w") as f:
        json.dump(log, f)
    print(f"\nModel saved  → {MODEL_FILE}")
    print(f"Training log → {LOG_FILE}")
    return agent, log


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(agent: DQNAgent, n_eval: int = 5):
    env = gym.make(
        "gym_dssat_pdi:GymDssatPdi-v0",
        mode="fertilization",
        seed=SEED + 100,
    )
    get_obs_dim(env)
    agent.epsilon = 0.0
    shaper  = RewardShaper()
    results = []

    for ep in range(n_eval):
        obs            = env.reset()
        last_valid_obs = obs
        state          = flatten_obs(obs)
        shaper.reset()
        done     = False
        ep_raw_r = 0.0
        schedule = []

        while not done:
            action_idx = agent.select_action(state)
            action_kg  = float(ACTIONS[action_idx])
            next_obs, raw_r, done, _ = env.step({"anfer": action_kg})
            if next_obs is not None:
                last_valid_obs = next_obs
            if raw_r is None:
                raw_r = 0.0
            ep_raw_r += raw_r
            if action_kg > 0:
                schedule.append((int(last_valid_obs.get("dap", -1)), action_kg))
            state = flatten_obs(last_valid_obs)

        results.append({
            "ep":         ep + 1,
            "raw_reward": ep_raw_r,
            "total_n":    shaper.total_n,
            "yield_kgha": float(last_valid_obs.get("topwt", 0.0)),
            "schedule":   schedule,
        })

    env.close()

    print("\n" + "=" * 65)
    print("EVALUATION — Dueling Double DQN + N-step (v3)")
    print("=" * 65)
    print(f"{'Ep':>4} | {'RawReward':>10} | {'TotalN(kg/ha)':>14} | {'Yield(kg/ha)':>12}")
    print("-" * 65)
    for r in results:
        print(f"{r['ep']:>4} | {r['raw_reward']:>10.1f} | "
              f"{r['total_n']:>14.1f} | {r['yield_kgha']:>12.1f}")
    print("-" * 65)
    avg_r = np.mean([r["raw_reward"]  for r in results])
    avg_n = np.mean([r["total_n"]     for r in results])
    avg_y = np.mean([r["yield_kgha"]  for r in results])
    print(f"{'AVG':>4} | {avg_r:>10.1f} | {avg_n:>14.1f} | {avg_y:>12.1f}")
    print("=" * 65)
    print(f"\nBaseline (naive):        ~-263")
    print(f"v3 avg raw reward:        {avg_r:.1f}")
    print(f"Improvement over baseline:{avg_r-(-263):+.1f}")
    return results


# ── Baselines ─────────────────────────────────────────────────────────────────
def run_baseline(n_kg: float, n_eval: int = 3):
    env     = gym.make("gym_dssat_pdi:GymDssatPdi-v0", mode="fertilization", seed=SEED+200)
    shaper  = RewardShaper()
    rewards = []
    for _ in range(n_eval):
        obs = env.reset(); shaper.reset(); done = False; ep_r = 0.0; applied = False
        while not done:
            vstage    = float(obs.get("vstage", 0.0))
            action_kg = n_kg if (vstage >= 5.0 and not applied) else 0.0
            if vstage >= 5.0 and not applied: applied = True
            next_obs, raw_r, done, _ = env.step({"anfer": action_kg})
            ep_r += raw_r or 0.0; obs = next_obs
        rewards.append(ep_r)
    env.close()
    avg = np.mean(rewards)
    print(f"Baseline ({n_kg:.0f} kg/ha at V5): avg = {avg:.1f}")
    return avg


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  Dueling Double DQN + N-step Returns  (v3)")
    print("=" * 65)
    print("\nAll active improvements:")
    print("  ★  Double DQN     — decoupled action selection/evaluation")
    print("  ★  Huber loss     — robust SmoothL1")
    print("  ★  Grad clipping  — max norm =", GRAD_CLIP)
    print(f"  ★  Soft updates   — Polyak τ = {TAU}")
    print(f"  ★  Warmup         — {WARMUP_STEPS} steps")
    print(f"  ★★ Dueling net    — V(s) + A(s,a) streams")
    print(f"  ★★ N-step returns — n = {N_STEPS},  γⁿ = {GAMMA_N:.5f}")
    print()

    print("--- Baselines ---")
    for n in [160, 240, 280]:
        run_baseline(float(n))

    print("\n--- Training v3 ---")
    agent, log = train()

    print("\n--- Evaluation ---")
    evaluate(agent, n_eval=5)
