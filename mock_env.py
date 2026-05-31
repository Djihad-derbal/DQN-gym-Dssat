"""
mock_env.py — Realistic mock of the gym-dssat-pdi fertilization environment.

Simulates a ~158-day corn growing season WITHOUT requiring Docker or DSSAT.
The 11 observation keys match the real environment (alphabetical order):
  cumsumfert, dap, dtt, istage, nstres, pcngrn, rain, tleachd, topwt, vstage, xlai

The DQN model trained on the real env will still produce meaningful Q-values
because the obs space shape and variable ranges are carefully matched.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any


ACTIONS = [0, 40, 80, 120, 160]   # kg N / ha  (same as training)
OBS_KEYS = ["cumsumfert", "dap", "dtt", "istage", "nstres",
            "pcngrn", "rain", "tleachd", "topwt", "vstage", "xlai"]
SEASON_LENGTH = 158               # days (matches quick_test output)

# Optimal total-N for the season (shapes n-stress curve)
OPTIMAL_N = 200.0   # kg/ha

# Reward weights  (paper eq.1)
W1, W2, W3, W4 = 0.1, 0.1, 0.1, 1.0
N_THRESHOLD = 300.0  # kg/ha — penalty threshold


def flatten_obs(obs: dict) -> np.ndarray:
    """Alphabetical flatten — must match dqn_nitrogen.py exactly."""
    parts = []
    for k in sorted(obs.keys()):
        v = obs[k]
        if isinstance(v, (int, float, np.integer, np.floating)):
            parts.append(float(v))
        elif isinstance(v, np.ndarray):
            parts.extend(v.flatten().tolist())
        else:
            parts.extend([float(x) for x in v])
    return np.array(parts, dtype=np.float32)


class MockCropEnv:
    """
    Mock DSSAT corn fertilization environment.

    Crop growth model (simplified):
    - Thermal-time drives vegetative stages (V1–V18) then reproductive
    - Leaf area index peaks around V12, then senesces
    - Biomass accumulates from leaf area × radiation-use efficiency
    - Nitrogen stress reduces RUE and grain filling
    - Nitrate leaching occurs on high-rain days when N has been applied
    - Yield = aboveground biomass × 0.48 grain-index at harvest
    """

    rng: np.random.Generator

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._state: dict = {}
        self._step_count = 0
        self._done = False

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> dict:
        self.rng = np.random.default_rng(self.seed + self._step_count)
        self._step_count = 0
        self._done = False

        # Internal continuous state
        self._cumN       = 0.0    # kg N/ha applied so far
        self._cumleach   = 0.0    # kg N/ha leached
        self._biomass    = 0.0    # kg/ha aboveground dry matter (topwt)
        self._lai        = 0.0    # leaf area index (m²/m²)
        self._vstage     = 0.0    # vegetative stage (leaf count)
        self._istage     = 1      # DSSAT growth-stage integer (1-9)
        self._n_pool     = 0.0    # kg N/ha available in root zone
        self._cumtt      = 0.0    # cumulative thermal time (°C·d)
        self._pcngrn     = 0.0    # grain N (%)
        self._grain_wt   = 0.0    # grain dry weight (kg/ha)

        return self._make_obs()

    def step(self, action_dict: dict) -> Tuple[Optional[dict], Optional[float], bool, dict]:
        """
        Returns (obs, raw_reward, done, info).
        raw_reward is None every step except the terminal one.
        """
        if self._done:
            return self._make_obs(), 0.0, True, {}

        action_kg = float(action_dict.get("anfer", 0.0))

        # Apply fertilizer
        self._cumN    += action_kg
        self._n_pool  += action_kg

        # Simulate one day
        self._step_count += 1
        dap = self._step_count

        # --- Weather ---
        dtt  = self._daily_thermal_time(dap)
        rain = self._daily_rain(dap)
        self._cumtt += dtt

        # --- Growth stages ---
        self._advance_growth_stage(dtt)

        # --- N dynamics ---
        nstres = self._nitrogen_stress()
        leach  = self._leaching(rain, action_kg)
        self._cumleach += leach
        self._n_pool    = max(0.0, self._n_pool - leach - 0.5)   # uptake + loss

        # --- Canopy / biomass ---
        self._grow_canopy(dtt, nstres)

        # --- Terminal ---
        done = dap >= SEASON_LENGTH
        self._done = done

        # --- Grain filling (last 30 days) ---
        if dap > SEASON_LENGTH - 30:
            fill_rate = 30.0 * nstres * (0.8 + 0.2 * self.rng.random())
            self._grain_wt += fill_rate
            self._pcngrn    = 1.5 * nstres + 0.2 * self.rng.standard_normal()
            self._pcngrn    = max(0.0, min(4.0, self._pcngrn))

        obs = self._make_obs()

        # raw_reward only at harvest
        raw_reward = None
        if done:
            yield_val = self._grain_wt
            obs["topwt"] = yield_val       # final yield reported as topwt
            raw_reward   = self._compute_raw_reward(yield_val)

        return obs, raw_reward, done, {}

    # ── Obs / reward helpers ─────────────────────────────────────────────────

    def _make_obs(self) -> dict:
        rain = self._daily_rain(self._step_count)
        return {
            "cumsumfert": float(self._cumN),
            "dap":        float(self._step_count),
            "dtt":        self._daily_thermal_time(self._step_count),
            "istage":     float(self._istage),
            "nstres":     float(self._nitrogen_stress()),
            "pcngrn":     float(self._pcngrn),
            "rain":       float(rain),
            "tleachd":    float(self._cumleach),
            "topwt":      float(self._biomass),
            "vstage":     float(self._vstage),
            "xlai":       float(self._lai),
        }

    def _compute_raw_reward(self, yield_val: float) -> float:
        """Simplified paper reward at harvest."""
        penalty = max(0.0, self._cumN - N_THRESHOLD)
        return W1 * yield_val - W3 * self._cumleach - W4 * penalty

    # ── Internal growth dynamics ─────────────────────────────────────────────

    def _daily_thermal_time(self, dap: int) -> float:
        """
        Iowa corn season thermal time: ~15–25°C·d/day early, tapering late.
        """
        base  = 18.0
        noise = self.rng.standard_normal() * 1.5
        # Slightly cooler at start and end of season
        seasonal = 1.0 - 0.3 * abs(dap - 80) / 80
        return max(2.0, base * seasonal + noise)

    def _daily_rain(self, dap: int) -> float:
        """Stochastic rainfall — higher probability mid-season."""
        prob = 0.25 + 0.15 * np.exp(-((dap - 70) ** 2) / 1200)
        if self.rng.random() < prob:
            return float(self.rng.exponential(8.0))
        return 0.0

    def _advance_growth_stage(self, dtt: float):
        """Map cumulative thermal time to DSSAT istage / vstage."""
        tt = self._cumtt
        # Rough thermal-time thresholds for Iowa corn
        if tt < 100:
            self._istage = 1   # germination / emergence
            self._vstage = max(0.0, (tt - 50) / 50)
        elif tt < 800:
            self._istage = 2   # vegetative
            self._vstage = 1.0 + (tt - 100) / 46.7     # ~V1 to V15
            self._vstage = min(15.0, self._vstage)
        elif tt < 1000:
            self._istage = 3   # tasseling / silking
            self._vstage = min(18.0, self._vstage + 0.05)
        elif tt < 1400:
            self._istage = 4   # grain fill
        elif tt < 1700:
            self._istage = 5   # dough
        else:
            self._istage = 6   # maturity

    def _nitrogen_stress(self) -> float:
        """
        N stress factor: 0 = maximum stress, 1 = no stress.
        Driven by N pool relative to crop demand.
        """
        demand = max(1.0, self._biomass * 0.015)   # ~1.5% N in biomass
        supply = self._cumN
        frac   = min(1.0, supply / (demand + 1e-6))
        # Add some noise to simulate weather/soil variability
        noise = self.rng.standard_normal() * 0.03
        return float(np.clip(0.35 + 0.65 * frac + noise, 0.0, 1.0))

    def _leaching(self, rain: float, applied_today: float) -> float:
        """N leached today: only significant on rainy days with N in pool."""
        if rain < 5.0 or self._n_pool < 1.0:
            return 0.0
        leach_frac = 0.08 * (rain / 20.0)
        return float(min(self._n_pool * leach_frac, applied_today * 0.3))

    def _grow_canopy(self, dtt: float, nstres: float):
        """Update LAI and biomass accumulation."""
        # LAI expansion (vegetative) — peaks at V12~V14
        if self._istage <= 3 and self._lai < 6.5:
            potential_lai = 0.06 * dtt / 18.0  * (1 + self._vstage / 15.0)
            self._lai += potential_lai * nstres
            self._lai  = min(6.5, self._lai)

        # LAI senescence (reproductive onward)
        elif self._istage >= 4:
            sene_rate = 0.03 + 0.02 * (1.0 - nstres)
            self._lai = max(0.0, self._lai - sene_rate)

        # Biomass accumulation via radiation-use efficiency
        rue = 1.6 * nstres                          # g/MJ PAR intercepted
        par = 8.0 * (1 - np.exp(-0.65 * self._lai)) # simplified PAR interception
        delta_bm = rue * par * dtt / 18.0 * 8.5    # rough conversion
        noise = self.rng.standard_normal() * 5.0
        self._biomass += max(0.0, delta_bm + noise)


# ── Standalone smoke test ────────────────────────────────────────────────────
if __name__ == "__main__":
    env = MockCropEnv(seed=42)
    for ep in range(3):
        obs = env.reset()
        done = False
        total_n = 0.0
        raw_sum = 0.0
        steps   = 0
        while not done:
            action_kg = float(np.random.choice(ACTIONS))
            next_obs, raw_r, done, _ = env.step({"anfer": action_kg})
            total_n += action_kg
            raw_sum += raw_r or 0.0
            steps   += 1
            obs = next_obs
        print(f"Episode {ep+1}: steps={steps}, raw_reward={raw_sum:.1f}, "
              f"total_N={total_n:.0f} kg/ha, yield={obs['topwt']:.0f} kg/ha")
    print("\nObs keys:", sorted(obs.keys()))
    print("Obs dim:", len(flatten_obs(obs)))
    print("✓ Mock env OK")
