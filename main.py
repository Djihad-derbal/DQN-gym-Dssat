"""
main.py — FastAPI backend for the DQN + LLM Nitrogen Management demo.

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000
Then open http://localhost:8000
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mock_env import MockCropEnv, flatten_obs, ACTIONS
from dqn_agent import DQNInference
from llm_service import LLMService

# ── App init ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Nitrogen Management — DQN + LLM",
    description="Visual demo of a trained DQN with an LLM co-pilot for precision agriculture.",
    version="1.0.0",
)

BASE_DIR   = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Global singletons ─────────────────────────────────────────────────────────

env  = MockCropEnv(seed=42)
dqn  = DQNInference(model_path=str(BASE_DIR / "dqn_model.pth"))
llm  = LLMService()

# ── Per-session state (single-user demo; extend with sessions for multi-user) ─

_episode: Dict[str, Any] = {
    "obs":      None,
    "steps":    [],      # list of step dicts
    "done":     False,
    "total_n":  0.0,
    "ep_seed":  42,
}


def _reset_episode(seed: int = 42):
    env.seed = seed
    obs = env.reset()
    _episode["obs"]     = obs
    _episode["steps"]   = []
    _episode["done"]    = False
    _episode["total_n"] = 0.0
    _episode["ep_seed"] = seed
    return obs


def _obs_payload(obs: dict) -> dict:
    """Package obs dict + flat array for the frontend."""
    flat = flatten_obs(obs).tolist()
    return {"obs": obs, "obs_flat": flat}


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    seed: int = 42

class StepRequest(BaseModel):
    action_index: Optional[int] = None  # None → DQN greedy; 0-4 → manual

class ChatRequest(BaseModel):
    message: str

class ConfigRequest(BaseModel):
    provider: str
    model: Optional[str]  = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None   # for Ollama


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Frontend not found. Place index.html in static/</h1>", 404)


# ── Episode endpoints ─────────────────────────────────────────────────────────

@app.post("/api/episode/start")
async def start_episode(req: StartRequest):
    """Reset the environment and return the initial observation."""
    obs = _reset_episode(req.seed)
    obs_flat = flatten_obs(obs).tolist()
    q_vals   = dqn.q_values(np.array(obs_flat)).tolist()
    explanation = await llm.explain_state(obs, day=0)
    llm.clear_history()

    return {
        **_obs_payload(obs),
        "step":          0,
        "episode_done":  False,
        "q_values":      q_vals,
        "action_labels": [f"{a} kg/ha" for a in ACTIONS],
        "llm_explanation": explanation,
        "dqn_status":    dqn.status(),
    }


@app.post("/api/episode/step")
async def step_episode(req: StepRequest):
    """Advance the simulation by one day."""
    if _episode["done"]:
        raise HTTPException(status_code=400, detail="Episode finished. Call /api/episode/start to restart.")
    if _episode["obs"] is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /api/episode/start first.")

    obs_flat   = flatten_obs(_episode["obs"]).tolist()
    q_values   = dqn.q_values(np.array(obs_flat)).tolist()

    # Choose action
    if req.action_index is not None:
        if req.action_index not in range(len(ACTIONS)):
            raise HTTPException(400, f"action_index must be 0-{len(ACTIONS)-1}")
        action_idx = req.action_index
    else:
        action_idx = int(np.argmax(q_values))

    action_kg = float(ACTIONS[action_idx])

    # Step environment
    prev_obs = _episode["obs"]
    next_obs, raw_reward, done, _ = env.step({"anfer": action_kg})

    _episode["total_n"] += action_kg
    _episode["done"]     = done

    # Record step
    step_record = {
        "step":       len(_episode["steps"]) + 1,
        "obs":        prev_obs,
        "action_idx": action_idx,
        "action_kg":  action_kg,
        "q_values":   q_values,
        "raw_reward": raw_reward if raw_reward is not None else 0.0,
        "done":       done,
        "total_n":    _episode["total_n"],
    }
    _episode["steps"].append(step_record)
    _episode["obs"] = next_obs

    # LLM (fire concurrently for speed)
    state_task    = asyncio.create_task(llm.explain_state(next_obs, day=step_record["step"]))
    decision_task = asyncio.create_task(
        llm.explain_decision(prev_obs, q_values, action_idx, action_kg)
    )
    state_explanation, decision_explanation = await asyncio.gather(state_task, decision_task)

    # Summary for season chart
    season_summary = [
        {"step": s["step"], "action_kg": s["action_kg"], "total_n": s["total_n"],
         "yield": s["obs"].get("topwt", 0.0), "nstres": s["obs"].get("nstres", 1.0)}
        for s in _episode["steps"]
    ]

    return {
        **_obs_payload(next_obs),
        "step":                step_record["step"],
        "episode_done":        done,
        "action_idx":          action_idx,
        "action_kg":           action_kg,
        "q_values":            q_values,
        "action_labels":       [f"{a} kg/ha" for a in ACTIONS],
        "raw_reward":          step_record["raw_reward"],
        "total_n":             _episode["total_n"],
        "state_explanation":   state_explanation,
        "decision_explanation": decision_explanation,
        "season_summary":      season_summary,
        "final_yield":         next_obs.get("topwt", 0.0) if done else None,
    }


@app.post("/api/episode/auto")
async def auto_run(steps: int = 10):
    """Run N steps automatically with the DQN greedy policy. Returns final state."""
    if _episode["done"]:
        raise HTTPException(400, "Episode finished.")
    if _episode["obs"] is None:
        raise HTTPException(400, "No active episode.")

    steps = min(steps, 158)
    results = []
    for _ in range(steps):
        if _episode["done"]:
            break
        obs_flat  = flatten_obs(_episode["obs"]).tolist()
        q_vals    = dqn.q_values(np.array(obs_flat)).tolist()
        action_idx = int(np.argmax(q_vals))
        action_kg  = float(ACTIONS[action_idx])
        prev_obs   = _episode["obs"]
        next_obs, raw_reward, done, _ = env.step({"anfer": action_kg})
        _episode["total_n"] += action_kg
        _episode["done"]     = done
        step_record = {
            "step":       len(_episode["steps"]) + 1,
            "obs":        prev_obs,
            "action_idx": action_idx,
            "action_kg":  action_kg,
            "q_values":   q_vals,
            "raw_reward": raw_reward if raw_reward is not None else 0.0,
            "done":       done,
            "total_n":    _episode["total_n"],
        }
        _episode["steps"].append(step_record)
        _episode["obs"] = next_obs
        results.append({
            "step": step_record["step"],
            "action_kg": action_kg,
            "total_n": _episode["total_n"],
            "yield": next_obs.get("topwt", 0.0),
            "nstres": next_obs.get("nstres", 1.0),
            "done": done,
        })

    obs_flat  = flatten_obs(_episode["obs"]).tolist()
    q_vals    = dqn.q_values(np.array(obs_flat)).tolist()
    explanation = await llm.explain_state(_episode["obs"], day=len(_episode["steps"]))

    season_summary = [
        {"step": s["step"], "action_kg": s["action_kg"], "total_n": s["total_n"],
         "yield": s["obs"].get("topwt", 0.0), "nstres": s["obs"].get("nstres", 1.0)}
        for s in _episode["steps"]
    ]

    return {
        **_obs_payload(_episode["obs"]),
        "step":              len(_episode["steps"]),
        "episode_done":      _episode["done"],
        "q_values":          q_vals,
        "action_labels":     [f"{a} kg/ha" for a in ACTIONS],
        "total_n":           _episode["total_n"],
        "state_explanation": explanation,
        "season_summary":    season_summary,
        "steps_run":         results,
        "final_yield":       _episode["obs"].get("topwt", 0.0) if _episode["done"] else None,
    }


# ── LLM chat ─────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: ChatRequest):
    context = {
        "step":         len(_episode["steps"]),
        "current_obs":  _episode["obs"],
        "episode_done": _episode["done"],
        "total_n":      _episode["total_n"],
        "actions_taken": [s["action_kg"] for s in _episode["steps"][-5:]],
    }
    response = await llm.chat(req.message, context)
    return {"response": response}


# ── Training log ──────────────────────────────────────────────────────────────

@app.get("/api/training-log")
async def training_log():
    log_path = BASE_DIR / "dqn_training_log.json"
    if log_path.exists():
        try:
            with open(log_path, "r") as f:
                data = json.load(f)
            # Downsample for network efficiency (every 5th ep)
            def ds(lst): return lst[::5]
            return {
                "episode_rewards":        ds(data.get("episode_rewards", [])),
                "episode_shaped_rewards": ds(data.get("episode_shaped_rewards", [])),
                "episode_total_n":        ds(data.get("episode_total_n", [])),
                "episode_yield":          ds(data.get("episode_yield", [])),
                "epsilon":                ds(data.get("epsilon", [])),
                "total_episodes":         len(data.get("episode_rewards", [])),
                "downsampled": True,
            }
        except Exception as e:
            return JSONResponse({"error": str(e)}, 500)
    return JSONResponse({"error": "dqn_training_log.json not found in app directory."}, 404)


# ── Config ────────────────────────────────────────────────────────────────────

@app.get("/api/config")
async def get_config():
    return llm.get_config()


@app.post("/api/config")
async def set_config(req: ConfigRequest):
    llm.update_config(req.dict(exclude_none=True))
    return {"status": "ok", "config": llm.get_config()}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status":    "ok",
        "dqn":       dqn.status(),
        "llm":       llm.get_config(),
        "env":       "MockCropEnv (DSSAT-compatible)",
    }
