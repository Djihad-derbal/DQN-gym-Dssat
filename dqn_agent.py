"""
dqn_agent.py — Load the trained DQN weights for pure inference.

Architecture matches dqn_nitrogen.py exactly:
  QNetwork: Linear(11→256) → ReLU → Linear(256→256) → ReLU → Linear(256→5)

Usage:
    agent = DQNInference("dqn_model.pth")
    q_vals  = agent.q_values(obs_flat)   # np.ndarray shape (5,)
    action  = agent.select_action(obs_flat)  # int 0-4
    action_kg = ACTIONS[action]
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

ACTIONS    = [0, 40, 80, 120, 160]   # kg N/ha
N_ACTIONS  = len(ACTIONS)
OBS_DIM    = 11
HIDDEN     = 256


class QNetwork(nn.Module):
    """Exact replica of the network in dqn_nitrogen.py."""
    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNInference:
    """
    Wraps the trained QNetwork for CPU inference.

    If the model file is not found, falls back to a randomly initialised
    network (useful for UI demo without the weights).
    """

    def __init__(self, model_path: str = "dqn_model.pth"):
        self.device = torch.device("cpu")   # always CPU for web serving
        self.network = QNetwork().to(self.device)
        self.loaded  = False
        self.model_path = model_path
        self._load(model_path)

    def _load(self, path: str):
        if os.path.exists(path):
            try:
                state = torch.load(path, map_location=self.device)
                self.network.load_state_dict(state)
                self.network.eval()
                self.loaded = True
                print(f"[DQN] Loaded weights from {path}")
            except Exception as e:
                print(f"[DQN] Warning: could not load weights ({e}). Using random init.")
        else:
            print(f"[DQN] Warning: {path} not found. Using random init.")
        self.network.eval()

    def q_values(self, obs_flat: np.ndarray) -> np.ndarray:
        """
        Return Q-values for all 5 actions given a flattened observation.

        Args:
            obs_flat: np.ndarray of shape (11,)
        Returns:
            np.ndarray of shape (5,)
        """
        with torch.no_grad():
            t = torch.FloatTensor(obs_flat).unsqueeze(0).to(self.device)
            q = self.network(t).squeeze(0).cpu().numpy()
        return q

    def select_action(self, obs_flat: np.ndarray) -> int:
        """Return greedy action index (0-4)."""
        return int(np.argmax(self.q_values(obs_flat)))

    def action_kg(self, obs_flat: np.ndarray) -> float:
        """Return the kg/ha amount the greedy policy would apply."""
        return float(ACTIONS[self.select_action(obs_flat)])

    def status(self) -> dict:
        return {
            "loaded": self.loaded,
            "model_path": self.model_path,
            "obs_dim": OBS_DIM,
            "n_actions": N_ACTIONS,
            "actions_kg": ACTIONS,
            "hidden_size": HIDDEN,
            "device": str(self.device),
        }
