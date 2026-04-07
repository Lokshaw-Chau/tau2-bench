#!/usr/bin/env python3
"""Edit the config block below, then run `uv run python scripts/run_text_eval.py`."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from render_tau2_summary import render_summary_report

from tau2 import TextRunConfig
from tau2.runner import run_domain

# ============================= Editable Config ==============================
AGENT_MODEL = "openai/glm-5"
MODEL_LABEL = "glm-5-zai-airline-temp0_user-gpt52-low"
API_BASE_URL = None
API_KEY = None
USER_SIMULATOR_MODEL = "openai/gpt-5.2"

DOMAINS = ["airline"]
NUM_TRIALS = 4
NUM_TASKS = None
TASK_IDS = None
MAX_CONCURRENCY = 48
MAX_STEPS = 200
SEED = 300
AUTO_RESUME = False
BATCH_NAME = None
VERBOSE_LOGS = True

AGENT_LLM_ARGS = {
    "temperature": 0,
    "extra_body": {
        "enable_thinking": True,
    },
}

USER_LLM_ARGS = {
    "reasoning_effort": "low",
    "temperature": 1.0,
    "top_p": 0.95,
}
# ===========================================================================


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
SIMULATIONS_ROOT = ROOT / "data" / "simulations"
SUMMARY_PATH = ROOT / "data" / "benchmark_summary.md"
MANIFEST_NAME = "run_manifest.json"


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "run"


def _resolve_secret(explicit: str | None, env_name: str) -> str | None:
    if explicit:
        return explicit
    env_value = os.getenv(env_name)
    if env_value:
        return env_value
    env_file_values = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}
    value = env_file_values.get(env_name)
    return str(value) if value else None


def _merge_llm_args(base_args: dict, api_base: str, api_key: str) -> dict:
    merged = dict(base_args)
    merged["api_base"] = api_base
    merged["api_key"] = api_key
    return merged


def _default_model_label() -> str:
    if MODEL_LABEL:
        return MODEL_LABEL
    return AGENT_MODEL.split("/")[-1]


def _build_batch_name(model_label: str) -> str:
    if BATCH_NAME:
        return BATCH_NAME
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    user_name = USER_SIMULATOR_MODEL.split("/")[-1]
    return "_".join(
        [
            timestamp,
            _sanitize_name(model_label),
            _sanitize_name(user_name),
        ]
    )


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _make_manifest(batch_name: str, model_label: str) -> dict[str, Any]:
    created_at = datetime.now(UTC).isoformat()
    return {
        "batch_name": batch_name,
        "model_label": model_label,
        "agent_model": AGENT_MODEL,
        "user_simulator_model": USER_SIMULATOR_MODEL,
        "domains": list(DOMAINS),
        "num_trials": NUM_TRIALS,
        "num_tasks": NUM_TASKS,
        "task_ids": TASK_IDS,
        "seed": SEED,
        "created_at": created_at,
        "completed": False,
        "completed_at": None,
    }


def main() -> None:
    api_base = _resolve_secret(API_BASE_URL, "OPENAI_BASE_URL")
    api_key = _resolve_secret(API_KEY, "OPENAI_API_KEY")

    if not api_base:
        raise RuntimeError(
            f"Missing API base URL. Set API_BASE_URL at the top of this script or provide OPENAI_BASE_URL in {ENV_PATH}."
        )
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set API_KEY at the top of this script or provide OPENAI_API_KEY in {ENV_PATH}."
        )

    model_label = _default_model_label()
    batch_name = _build_batch_name(model_label)
    batch_dir = SIMULATIONS_ROOT / batch_name
    manifest_path = batch_dir / MANIFEST_NAME
    manifest = _make_manifest(batch_name, model_label)
    _write_manifest(manifest_path, manifest)

    agent_llm_args = _merge_llm_args(AGENT_LLM_ARGS, api_base, api_key)
    user_llm_args = _merge_llm_args(USER_LLM_ARGS, api_base, api_key)

    print("Batch directory:", batch_dir)
    print("Model label:", model_label)
    print("Domains:", DOMAINS)

    for domain in DOMAINS:
        save_to = f"{batch_name}/{domain}"
        config = TextRunConfig(
            domain=domain,
            llm_agent=AGENT_MODEL,
            llm_args_agent=agent_llm_args,
            llm_user=USER_SIMULATOR_MODEL,
            llm_args_user=user_llm_args,
            num_trials=NUM_TRIALS,
            num_tasks=NUM_TASKS,
            task_ids=TASK_IDS,
            max_concurrency=MAX_CONCURRENCY,
            max_steps=MAX_STEPS,
            seed=SEED,
            save_to=save_to,
            auto_resume=AUTO_RESUME,
            verbose_logs=VERBOSE_LOGS,
        )
        run_domain(config)

    manifest["completed"] = True
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    _write_manifest(manifest_path, manifest)

    summary_path = render_summary_report(
        simulations_root=SIMULATIONS_ROOT,
        output_path=SUMMARY_PATH,
    )

    print("Summary file:", summary_path)


if __name__ == "__main__":
    main()
