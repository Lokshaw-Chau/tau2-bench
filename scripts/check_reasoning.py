#!/usr/bin/env python3
"""Minimal probe for checking whether an OpenAI-compatible endpoint returns reasoning."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI

# ============================= Editable Config ==============================
MODEL = "glm-5"
PROMPT = "Write a one-sentence bedtime story about a unicorn."
TEMPERATURE = 0
THINKING = {"type": "enabled"}
# ===========================================================================


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"


def main() -> None:
    env = dotenv_values(ENV_PATH)
    api_base = str(env.get("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "")
    api_key = str(env.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or "")
    api_base = api_base.rstrip("/")

    if not api_base:
        raise RuntimeError(f"Missing OPENAI_BASE_URL in {ENV_PATH}")
    if not api_key:
        raise RuntimeError(f"Missing OPENAI_API_KEY in {ENV_PATH}")

    client = OpenAI(
        base_url=api_base,
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": PROMPT,
            }
        ],
        # temperature=TEMPERATURE,
        # thinking={"type": "enabled"},
        # stream=True,
        extra_body={
            "enable_thinking": True,
            # "thinking": {
            #     "type": "enabled",
            # },
            "temperature": TEMPERATURE,
        },
    )
    # for chunk in response:
    #     if chunk.choices[0].delta.reasoning_content:
    #         print(chunk.choices[0].delta.reasoning_content, end='')
    #     if chunk.choices[0].delta.content:
    #         print(chunk.choices[0].delta.content, end='')
    response_data = response.model_dump()
    message = response_data["choices"][0]["message"]
    usage = response_data.get("usage", {})
    completion_details = usage.get("completion_tokens_details", {})
    reasoning_content = message.get("reasoning_content")
    reasoning_tokens = completion_details.get("reasoning_tokens")

    print(f"api_base: {api_base}")
    print(f"model: {response_data.get('model')}")
    print(f"reasoning_content_present: {bool(reasoning_content)}")
    print(f"reasoning_tokens: {reasoning_tokens}")
    print()
    print("assistant_content:")
    print(message.get("content", ""))
    print()
    print("reasoning_content:")
    print(reasoning_content or "<empty>")
    print()
    print("usage:")
    print(json.dumps(usage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
