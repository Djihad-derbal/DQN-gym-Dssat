"""
llm_service.py — Configurable multi-provider LLM service.

Supported providers:
  - "openai"    — OpenAI (gpt-4o-mini by default, any model string works)
  - "anthropic" — Anthropic Claude (claude-haiku-3-5 by default)
  - "ollama"    — Local Ollama (llama3.2 by default, http://localhost:11434)
  - "mock"      — No API key needed; returns rule-based canned responses

The LLM plays THREE roles in the system:
  1. State Explainer   — translates crop obs dict → plain-English farm status
  2. Decision Narrator — explains WHY the DQN chose a specific fertilizer dose
  3. Strategy Advisor  — free-form chat about nitrogen management
"""

import os
import json
import asyncio
from typing import Optional, Dict, Any, List

# ─── Lazy imports so the app starts even if SDKs are absent ─────────────────

def _try_import(module: str):
    try:
        import importlib
        return importlib.import_module(module)
    except ImportError:
        return None


# ── Context prompts ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an agronomy assistant embedded in a precision-agriculture
decision-support tool. Your ONLY job is to explain nitrogen management for corn,
the DSSAT crop simulation, and the DQN agent's decisions.

GROUND TRUTH — established facts you may rely on:
- Simulation: DSSAT DSCSM047 crop model, Iowa USA corn, ~158-day season.
- Optimal seasonal N for Iowa corn: 150-250 kg/ha (university extension consensus).
- DQN action set: {0, 40, 80, 120, 160} kg N/ha per day.
- Reward function (Wu et al., CVPRW 2022):
    r = 0.1·Yield - 0.1·N_applied - 0.1·Leaching - 1.0·OverFertPenalty
- Variables in the obs dict (use ONLY these names and the values provided):
    dap = days after planting
    istage = 1 emergence, 2 vegetative, 3 tasseling, 4 grain fill, 6 maturity
    vstage = vegetative leaf stage (V-stage)
    xlai = leaf area index (m^2/m^2)
    nstres = nitrogen stress; 1.0 = no stress, 0.0 = max stress
    cumsumfert = cumulative N applied (kg/ha)
    tleachd = cumulative N leached (kg/ha)
    topwt = aboveground biomass / yield proxy (kg/ha)
    rain = today's rainfall (mm)
    pcngrn = grain N content (%)

STRICT RULES:
1. Use ONLY the numbers from the obs dict the user provides. Never invent values.
2. Never fabricate citations, study results, field trial data, or named researchers
   beyond Wu et al. (2022) above.
3. If the question is OFF-TOPIC (politics, general chat, other crops, other countries'
   regulations, etc.), reply exactly: "That's outside the scope of this tool — I can
   only help with corn nitrogen management and the DQN simulation."
4. If you don't know, say "I don't have reliable data on that." Do NOT guess.
5. Do not recommend specific commercial fertilizer products or brands.
6. Use established agronomic principles only. No speculation about novel techniques.
7. Answer in 2-4 sentences. Plain language. No emoji. No bullet lists unless asked.
"""

VARIABLE_GLOSSARY = {
    "cumsumfert": "total nitrogen applied so far (kg/ha)",
    "dap":        "days after planting",
    "dtt":        "daily thermal time (°C·days, drives growth rate)",
    "istage":     "growth stage (1=emergence, 2=vegetative, 3=tasseling, 4=grain fill, 6=maturity)",
    "nstres":     "nitrogen stress factor (1.0 = no stress, 0.0 = maximum stress)",
    "pcngrn":     "grain nitrogen content (%)",
    "rain":       "rainfall today (mm)",
    "tleachd":    "cumulative nitrogen leached from soil (kg/ha)",
    "topwt":      "aboveground dry biomass / yield (kg/ha)",
    "vstage":     "vegetative leaf stage (V1, V2, … V18)",
    "xlai":       "leaf area index (m²/m²)",
}

ACTION_LABELS = {0: "Apply 0 kg/ha (skip)", 40: "Apply 40 kg/ha (light)",
                 80: "Apply 80 kg/ha (moderate)", 120: "Apply 120 kg/ha (heavy)",
                 160: "Apply 160 kg/ha (maximum)"}


def _obs_to_text(obs: dict) -> str:
    """Format obs dict as readable text."""
    lines = []
    for k in sorted(obs.keys()):
        gloss = VARIABLE_GLOSSARY.get(k, k)
        lines.append(f"  {gloss}: {obs[k]:.2f}")
    return "\n".join(lines)


def _mock_explain_state(obs: dict) -> str:
    dap   = int(obs.get("dap", 0))
    nstres = obs.get("nstres", 1.0)
    vstage = obs.get("vstage", 0.0)
    cumN  = obs.get("cumsumfert", 0.0)
    lai   = obs.get("xlai", 0.0)
    rain  = obs.get("rain", 0.0)

    stress = "no nitrogen stress" if nstres > 0.8 else (
             "mild nitrogen stress" if nstres > 0.5 else "significant nitrogen stress")
    rain_note = f" It rained {rain:.1f} mm today." if rain > 2 else " No rain today."

    return (f"Day {dap} of the growing season. The crop is at vegetative stage "
            f"V{vstage:.1f} with a leaf area index of {lai:.2f}. "
            f"Nitrogen status: {stress} (score {nstres:.2f}). "
            f"Total N applied so far: {cumN:.0f} kg/ha.{rain_note}")


def _mock_explain_decision(obs: dict, q_values: List[float],
                            action_idx: int, action_kg: float) -> str:
    dap    = int(obs.get("dap", 0))
    nstres = obs.get("nstres", 1.0)
    cumN   = obs.get("cumsumfert", 0.0)
    best_q = max(q_values)
    margin = best_q - sorted(q_values)[-2] if len(q_values) > 1 else 0.0

    if action_kg == 0:
        reason = (f"The DQN skips fertilization today (day {dap}). "
                  f"With {cumN:.0f} kg/ha already applied and a stress score of "
                  f"{nstres:.2f}, adding more nitrogen would incur an "
                  f"over-fertilization penalty without meaningful yield gain.")
    else:
        reason = (f"The DQN applies {action_kg:.0f} kg/ha today (day {dap}). "
                  f"The nitrogen stress factor is {nstres:.2f}, indicating the crop "
                  f"needs more nitrogen. The Q-value margin over the next-best action "
                  f"is {margin:.2f}, showing moderate confidence in this decision.")
    return reason


def _mock_chat(message: str, context: dict) -> str:
    msg = message.lower()
    if "leach" in msg:
        return ("Nitrate leaching happens when rain washes nitrogen below the root zone. "
                "Splitting fertilizer into smaller doses timed around crop demand "
                "can significantly reduce losses to groundwater.")
    if "yield" in msg or "harvest" in msg:
        return ("Corn yield in this model is driven by nitrogen availability, "
                "radiation interception (leaf area), and thermal time accumulation. "
                "The DQN tries to maximise final topwt while minimising N overuse.")
    if "stress" in msg or "nstres" in msg:
        return ("The nstres score ranges from 0 (max stress) to 1 (no stress). "
                "Values below 0.6 typically reduce photosynthesis and grain filling. "
                "The DQN usually applies N when it sees nstres dropping below ~0.7.")
    if "action" in msg or "dose" in msg or "how much" in msg:
        return ("The DQN picks from five doses: 0, 40, 80, 120, or 160 kg N/ha per day. "
                "It balances immediate yield benefit against the leaching penalty (w3) "
                "and the over-fertilization penalty (w4) in its reward function.")
    if "dqn" in msg or "reinforcement" in msg or "ai" in msg:
        return ("The DQN was trained for 1200 episodes on the DSSAT DSCSM047 crop model "
                "using the reward from Wu et al. (CVPRW 2022): r = 0.1·Yield − 0.1·N_applied "
                "− 0.1·Leaching − 1.0·Penalty. It outperforms naive fixed-application baselines "
                "by learning season-aware timing.")
    dap = context.get("step", 0)
    return (f"Great question! Currently on day {dap} of the growing season. "
            "I'm running in offline mode — connect an LLM API key in Settings "
            "for richer agronomic explanations.")


# ── LLM Service class ────────────────────────────────────────────────────────

class LLMService:

    def __init__(self):
        self.provider  = os.getenv("LLM_PROVIDER", "mock")
        self.model     = os.getenv("LLM_MODEL", "")
        self.api_key   = os.getenv("LLM_API_KEY", "")
        self.base_url  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._history: List[dict] = []    # advisor chat history

    # ── Config ──────────────────────────────────────────────────────────────

    def get_config(self) -> dict:
        return {
            "provider":  self.provider,
            "model":     self.model or self._default_model(),
            "has_key":   bool(self.api_key),
            "base_url":  self.base_url,
            "providers": ["openai", "anthropic", "gemini", "groq", "ollama", "mock"],
        }

    def update_config(self, cfg: dict):
        new_provider = cfg.get("provider", self.provider)
        if new_provider != self.provider and "model" not in cfg:
            self.model = ""
        self.provider = new_provider
        if "model" in cfg:
            self.model = cfg["model"] or ""
        if cfg.get("api_key"):
            self.api_key = cfg["api_key"]
        if cfg.get("base_url"):
            self.base_url = cfg["base_url"]
        self._history.clear()
        print(f"[LLM] Provider set to: {self.provider} / {self.model or self._default_model()}")

    def _default_model(self) -> str:
        return {
            "openai":    "gpt-4o-mini",
            "anthropic": "claude-haiku-4-5",
            "gemini":    "gemini-2.0-flash",
            "groq":      "llama-3.3-70b-versatile",
            "ollama":    "llama3.2",
            "mock":      "mock",
        }.get(self.provider, "mock")

    def _model(self) -> str:
        return self.model or self._default_model()

    def _needs_key(self) -> bool:
        return self.provider in ("openai", "anthropic", "gemini", "groq")

    def _should_mock(self) -> bool:
        return self.provider == "mock" or (self._needs_key() and not self.api_key)

    # ── Role 1: State Explainer ──────────────────────────────────────────────

    async def explain_state(self, obs: dict, day: int) -> str:
        if self._should_mock():
            return _mock_explain_state(obs)

        prompt = (f"Day {day} crop state:\n{_obs_to_text(obs)}\n\n"
                  "In 2-3 sentences, explain the current crop status to a farmer.")
        return await self._call(prompt)

    # ── Role 2: Decision Narrator ────────────────────────────────────────────

    async def explain_decision(self, obs: dict, q_values: List[float],
                                action_idx: int, action_kg: float) -> str:
        if self._should_mock():
            return _mock_explain_decision(obs, q_values, action_idx, action_kg)

        q_str = ", ".join(
            f"{a}kg/ha → Q={q:.2f}"
            for a, q in zip([0, 40, 80, 120, 160], q_values)
        )
        prompt = (
            f"Current crop state (day {int(obs.get('dap',0))}):\n{_obs_to_text(obs)}\n\n"
            f"DQN Q-values: {q_str}\n"
            f"Chosen action: {action_kg:.0f} kg/ha (index {action_idx}).\n\n"
            "In 2-3 sentences, explain to a farmer WHY the AI chose this fertilization dose."
        )
        return await self._call(prompt)

    # ── Role 3: Strategy Advisor ─────────────────────────────────────────────

    async def chat(self, message: str, context: dict) -> str:
        if self._should_mock():
            return _mock_chat(message, context)

        # Build context summary
        ctx_lines = []
        if context.get("step"):
            ctx_lines.append(f"Current day: {context['step']}")
        if context.get("current_obs"):
            ctx_lines.append("Current crop state:\n" + _obs_to_text(context["current_obs"]))
        if context.get("actions_taken"):
            acts = context["actions_taken"]
            ctx_lines.append(f"Last {len(acts)} actions (kg/ha): {acts}")

        ctx_block = "\n".join(ctx_lines) if ctx_lines else "No active episode."
        user_msg  = f"[Context]\n{ctx_block}\n\n[Farmer question]\n{message}"

        self._history.append({"role": "user", "content": user_msg})
        response = await self._call_chat()
        self._history.append({"role": "assistant", "content": response})

        # Keep history bounded
        if len(self._history) > 20:
            self._history = self._history[-20:]

        return response

    def clear_history(self):
        self._history.clear()

    # ── Internal HTTP call ───────────────────────────────────────────────────

    async def _call(self, user_prompt: str) -> str:
        """Single-turn call (for explainer / narrator)."""
        messages = [{"role": "user", "content": user_prompt}]
        try:
            if self.provider == "openai":
                return await self._openai(messages)
            elif self.provider == "anthropic":
                return await self._anthropic(messages)
            elif self.provider == "gemini":
                return await self._gemini(messages)
            elif self.provider == "groq":
                return await self._groq(messages)
            elif self.provider == "ollama":
                return await self._ollama(messages)
        except Exception as e:
            return f"[LLM error: {e}] " + _mock_explain_state({})
        return _mock_explain_state({})

    async def _call_chat(self) -> str:
        """Multi-turn call using self._history."""
        try:
            if self.provider == "openai":
                return await self._openai(self._history)
            elif self.provider == "anthropic":
                return await self._anthropic(self._history)
            elif self.provider == "gemini":
                return await self._gemini(self._history)
            elif self.provider == "groq":
                return await self._groq(self._history)
            elif self.provider == "ollama":
                return await self._ollama(self._history)
        except Exception as e:
            return f"[LLM error: {e}]"
        return "Provider not configured."

    # ── Provider implementations ─────────────────────────────────────────────

    async def _openai(self, messages: list) -> str:
        import httpx
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        body = {
            "model":    self._model(),
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            "max_tokens": 256,
            "temperature": 0.2,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post("https://api.openai.com/v1/chat/completions",
                                  json=body, headers=headers)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()

    async def _anthropic(self, messages: list) -> str:
        import httpx
        headers = {
            "x-api-key":         self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type":      "application/json",
        }
        body = {
            "model":      self._model(),
            "system":     SYSTEM_PROMPT,
            "messages":   messages,
            "max_tokens": 256,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post("https://api.anthropic.com/v1/messages",
                                  json=body, headers=headers)
            r.raise_for_status()
            return r.json()["content"][0]["text"].strip()

    async def _groq(self, messages: list) -> str:
        import httpx
        # Groq uses the OpenAI-compatible Chat Completions schema.
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        body = {
            "model":       self._model(),
            "messages":    [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            "max_tokens":  256,
            "temperature": 0.2,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post("https://api.groq.com/openai/v1/chat/completions",
                                  json=body, headers=headers)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()

    async def _gemini(self, messages: list) -> str:
        import httpx
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user",
             "parts": [{"text": m["content"]}]}
            for m in messages
        ]
        body = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.9,
                "maxOutputTokens": 256,
            },
        }
        url = (f"https://generativelanguage.googleapis.com/v1beta/"
               f"models/{self._model()}:generateContent?key={self.api_key}")
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, json=body)
            r.raise_for_status()
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    async def _ollama(self, messages: list) -> str:
        import httpx
        body = {
            "model":    self._model(),
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            "stream":   False,
            "options":  {"temperature": 0.2, "top_p": 0.9, "num_predict": 256},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{self.base_url}/api/chat",
                                  json=body)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
