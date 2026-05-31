"""
dqn_nitrogen_transfer.py
Cross-Climate Transfer Learning: Iowa → Northern Algeria (Tell Region)

Compares three training conditions on the Algeria maize environment:

  Condition A — Transfer (Full Fine-tune):
      Load pre-trained Iowa DQN v3 weights → fine-tune ALL layers on Algeria
      with a reduced learning rate (1e-5). Tests whether the Iowa policy
      provides a warm-start that accelerates convergence in a drier climate.

  Condition B — Transfer (Frozen Backbone):
      Load Iowa weights → FREEZE the first shared FC layer → train only the
      second FC layer and output head on Algeria (lr = 5e-5). Tests whether
      the low-level crop features learned in Iowa transfer without modification.

  Condition C — Scratch Baseline:
      Random weight initialisation → train from scratch on Algeria
      (lr = 5e-5, same as original Iowa training). Establishes the baseline
      convergence rate for the target environment.

Outputs (saved to ./nitrogen_app/):
  dqn_model_algeria_transfer.pth    — best Condition A weights
  dqn_model_algeria_frozen.pth      — best Condition B weights
  dqn_model_algeria_scratch.pth     — best Condition C weights
  transfer_comparison_log.json      — per-episode metrics for all 3 conditions

Usage:
  python dqn_nitrogen_transfer.py [--iowa-weights PATH] [--episodes N]
  python dqn_nitrogen_transfer.py                          # uses defaults
"""

import argparse
import collections
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
import sys
sys.path.insert(0, str(ROOT))
from mock_env_algeria import AlgeriaCropEnv, ACTIONS, flatten_obs


# ══════════════════════════════════════════════════════════════════════════════
#  Architecture  (Dueling — identical to v3 so weights load cleanly)
# ══════════════════════════════════════════════════════════════════════════════
class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim: int = 11, n_actions: int = 5, hidden: int = 256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.value_stream     = nn.Linear(hidden, 1)
        self.advantage_stream = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f  = self.feature(x)
        V  = self.value_stream(f)
        A  = self.advantage_stream(f)
        return V + (A - A.mean(dim=1, keepdim=True))

    def freeze_backbone(self):
        """Freeze the shared feature extractor (first two FC layers)."""
        for param in self.feature.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


# ══════════════════════════════════════════════════════════════════════════════
#  N-step Replay Buffer (identical to v3)
# ══════════════════════════════════════════════════════════════════════════════
class NStepReplayBuffer:
    def __init__(self, capacity: int = 50_000, n: int = 5, gamma: float = 0.99):
        self.buffer   = collections.deque(maxlen=capacity)
        self.n_buf    = collections.deque(maxlen=n)
        self.n        = n
        self.gamma    = gamma
        self.gamma_n  = gamma ** n

    def push(self, s, a, r, s2, done):
        self.n_buf.append((s, a, r, s2, done))
        if len(self.n_buf) == self.n:
            self._emit()
        if done:
            while len(self.n_buf) > 0:
                self._emit()

    def _emit(self):
        s0, a0, _, _, _ = self.n_buf[0]
        G = 0.0
        for i, (_, _, r, _, _) in enumerate(self.n_buf):
            G += (self.gamma ** i) * r
        sn, _, _, s_boot, d_boot = self.n_buf[-1]
        self.buffer.append((s0, a0, G, s_boot, d_boot))
        self.n_buf.popleft()

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, g, s2, d = zip(*batch)
        return (torch.FloatTensor(np.array(s)),
                torch.LongTensor(a),
                torch.FloatTensor(g),
                torch.FloatTensor(np.array(s2)),
                torch.FloatTensor(d))

    def __len__(self):
        return len(self.buffer)


# ══════════════════════════════════════════════════════════════════════════════
#  Single-condition trainer
# ══════════════════════════════════════════════════════════════════════════════
def train_condition(
    condition_name: str,
    iowa_weights:   str | None,
    freeze_backbone: bool,
    lr:             float,
    n_episodes:     int,
    seed:           int,
    convergence_threshold: float = -50.0,
) -> dict:
    """
    Train one condition; return per-episode log dict.
    iowa_weights=None → train from scratch.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env     = AlgeriaCropEnv(seed=seed)
    q_net   = DuelingQNetwork()
    t_net   = DuelingQNetwork()
    buf     = NStepReplayBuffer()

    # ── Weight initialisation ─────────────────────────────────────────────────
    if iowa_weights and Path(iowa_weights).exists():
        state_dict = torch.load(iowa_weights, map_location="cpu",
                                weights_only=False)
        # Handle both plain and wrapped checkpoints
        if isinstance(state_dict, dict) and "q_net" in state_dict:
            q_net.load_state_dict(state_dict["q_net"])
        else:
            q_net.load_state_dict(state_dict)
        t_net.load_state_dict(q_net.state_dict())
        print(f"  [{condition_name}] Loaded Iowa weights from {iowa_weights}")
    else:
        if iowa_weights:
            print(f"  [{condition_name}] Iowa weights not found at {iowa_weights} "
                  f"— training from scratch")
        else:
            print(f"  [{condition_name}] Training from scratch")

    if freeze_backbone:
        q_net.freeze_backbone()
        trainable = [p for p in q_net.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable, lr=lr)
        print(f"  [{condition_name}] Backbone frozen; training head only")
    else:
        optimizer = optim.Adam(q_net.parameters(), lr=lr)

    t_net.load_state_dict(q_net.state_dict())

    GAMMA_N       = buf.gamma_n
    BATCH         = 64
    WARMUP        = 500
    TAU           = 0.005
    GRAD_CLIP     = 10.0
    TARGET_SYNC   = 10
    eps           = 1.0
    EPS_DECAY     = 0.993 if iowa_weights else 0.992   # faster decay when transferring
    EPS_MIN       = 0.01
    loss_fn       = nn.SmoothL1Loss()

    log = {
        "condition":    condition_name,
        "episodes":     [],
        "raw_rewards":  [],
        "total_n":      [],
        "ma50":         [],
        "converged_ep": None,
    }
    reward_window = collections.deque(maxlen=50)
    converged     = False
    global_step   = 0
    t0            = time.time()

    for ep in range(1, n_episodes + 1):
        obs  = env.reset()
        s    = flatten_obs(obs)
        done = False
        ep_reward = 0.0
        ep_n      = 0.0

        while not done:
            # ε-greedy
            if global_step < WARMUP or random.random() < eps:
                act_idx = random.randrange(len(ACTIONS))
            else:
                with torch.no_grad():
                    act_idx = int(q_net(torch.FloatTensor(s).unsqueeze(0))
                                  .argmax(dim=1).item())

            action_kg = ACTIONS[act_idx]
            obs2, raw_r, done, _ = env.step({"anfer": float(action_kg)})
            s2 = flatten_obs(obs2)
            r  = raw_r if raw_r is not None else 0.0

            buf.push(s, act_idx, r, s2, float(done))
            ep_reward += r
            ep_n      += action_kg
            s          = s2
            global_step += 1

            # Training step
            if len(buf) >= WARMUP and len(buf) >= BATCH:
                S, A, G, S2, D = buf.sample(BATCH)
                with torch.no_grad():
                    next_acts = q_net(S2).argmax(dim=1, keepdim=True)
                    next_q    = t_net(S2).gather(1, next_acts).squeeze(1)
                    target    = G + GAMMA_N * next_q * (1 - D)
                pred   = q_net(S).gather(1, A.unsqueeze(1)).squeeze(1)
                loss   = loss_fn(pred, target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), GRAD_CLIP)
                optimizer.step()

                # Soft target update
                for tp, op in zip(t_net.parameters(), q_net.parameters()):
                    tp.data.copy_(TAU * op.data + (1 - TAU) * tp.data)

        # Hard target sync every TARGET_SYNC eps
        if ep % TARGET_SYNC == 0:
            t_net.load_state_dict(q_net.state_dict())

        eps = max(EPS_MIN, eps * EPS_DECAY)
        reward_window.append(ep_reward)
        ma50 = float(np.mean(reward_window))

        # Convergence detection
        if not converged and len(reward_window) == 50 and ma50 > convergence_threshold:
            log["converged_ep"] = ep
            converged = True
            print(f"  [{condition_name}] *** Converged at episode {ep} "
                  f"(MA-50 = {ma50:.1f}) ***")

        log["episodes"].append(ep)
        log["raw_rewards"].append(round(ep_reward, 2))
        log["total_n"].append(round(ep_n, 1))
        log["ma50"].append(round(ma50, 2))

        if ep % 100 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / ep * (n_episodes - ep)
            print(f"  [{condition_name}] Ep {ep:4d}/{n_episodes}  "
                  f"MA-50={ma50:7.1f}  ε={eps:.3f}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    return log, q_net


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Iowa→Algeria transfer experiment")
    parser.add_argument("--iowa-weights", default=str(ROOT / "dqn_model_v3.pth"),
                        help="Path to pre-trained Iowa DQN v3 weights")
    parser.add_argument("--episodes",     type=int, default=800,
                        help="Training episodes per condition (default 800)")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    N_EP = args.episodes
    print("\n" + "=" * 60)
    print("Cross-Climate Transfer Learning: Iowa → Northern Algeria")
    print("=" * 60)
    print(f"Iowa weights : {args.iowa_weights}")
    print(f"Episodes     : {N_EP} per condition")
    print(f"Seed         : {args.seed}")
    print("=" * 60 + "\n")

    results = {}

    # ── Condition A: Full fine-tune ───────────────────────────────────────────
    print("\n[ Condition A ] Transfer — Full Fine-tune (lr=1e-5)")
    log_a, net_a = train_condition(
        "A-FullFT", iowa_weights=args.iowa_weights,
        freeze_backbone=False, lr=1e-5,
        n_episodes=N_EP, seed=args.seed,
    )
    results["A_full_finetune"] = log_a
    torch.save(net_a.state_dict(), ROOT / "dqn_model_algeria_transfer.pth")
    print(f"  Saved → dqn_model_algeria_transfer.pth")

    # ── Condition B: Frozen backbone ──────────────────────────────────────────
    print("\n[ Condition B ] Transfer — Frozen Backbone (lr=5e-5)")
    log_b, net_b = train_condition(
        "B-Frozen", iowa_weights=args.iowa_weights,
        freeze_backbone=True, lr=5e-5,
        n_episodes=N_EP, seed=args.seed,
    )
    results["B_frozen_backbone"] = log_b
    torch.save(net_b.state_dict(), ROOT / "dqn_model_algeria_frozen.pth")
    print(f"  Saved → dqn_model_algeria_frozen.pth")

    # ── Condition C: Scratch ──────────────────────────────────────────────────
    print("\n[ Condition C ] Scratch Baseline (lr=5e-5)")
    log_c, net_c = train_condition(
        "C-Scratch", iowa_weights=None,
        freeze_backbone=False, lr=5e-5,
        n_episodes=N_EP, seed=args.seed,
    )
    results["C_scratch"] = log_c
    torch.save(net_c.state_dict(), ROOT / "dqn_model_algeria_scratch.pth")
    print(f"  Saved → dqn_model_algeria_scratch.pth")

    # ── Save comparison log ───────────────────────────────────────────────────
    log_path = ROOT / "transfer_comparison_log.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nComparison log saved → {log_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    header = f"{'Condition':<25} {'Conv.Ep':>8} {'Final MA-50':>12} {'Δ vs Scratch':>12}"
    print(header)
    print("-" * 60)
    scratch_final = float(np.mean(log_c["ma50"][-50:]))
    for key, label in [
        ("A_full_finetune", "A: Full Fine-tune"),
        ("B_frozen_backbone", "B: Frozen Backbone"),
        ("C_scratch", "C: Scratch"),
    ]:
        lg     = results[key]
        conv   = lg["converged_ep"] or "—"
        final  = float(np.mean(lg["ma50"][-50:]))
        delta  = final - scratch_final if key != "C_scratch" else 0.0
        delta_s = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        print(f"{label:<25} {str(conv):>8} {final:>12.1f} {delta_s:>12}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
