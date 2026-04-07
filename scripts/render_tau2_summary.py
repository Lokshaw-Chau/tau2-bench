#!/usr/bin/env python3
"""Render a Markdown leaderboard-style summary for completed tau2 text batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tau2.data_model.simulation import Results
from tau2.metrics.agent_metrics import compute_metrics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIMULATIONS_ROOT = ROOT / "data" / "simulations"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "benchmark_summary.md"
MANIFEST_NAME = "run_manifest.json"
PASS_KS = (1, 2, 3, 4)
PREFERRED_DOMAIN_ORDER = [
    "airline",
    "retail",
    "telecom",
    "banking_knowledge",
    "mock",
]


def _fmt_pass(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _iter_completed_manifests(simulations_root: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    if not simulations_root.exists():
        return manifests

    for manifest_path in sorted(simulations_root.glob(f"*/{MANIFEST_NAME}")):
        manifest = _load_json(manifest_path)
        if not manifest.get("completed"):
            continue
        manifest["_manifest_path"] = str(manifest_path)
        manifests.append(manifest)
    return manifests


def _pick_latest_by_model(manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_by_model: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        model_label = manifest["model_label"]
        sort_key = manifest.get("completed_at") or manifest.get("created_at") or ""
        current = latest_by_model.get(model_label)
        if current is None:
            latest_by_model[model_label] = manifest
            continue
        current_key = current.get("completed_at") or current.get("created_at") or ""
        if sort_key >= current_key:
            latest_by_model[model_label] = manifest
    return sorted(latest_by_model.values(), key=lambda item: item["model_label"])


def _ordered_domains(manifests: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []

    for domain in PREFERRED_DOMAIN_ORDER:
        for manifest in manifests:
            if domain in manifest.get("domains", []) and domain not in seen:
                ordered.append(domain)
                seen.add(domain)
                break

    for manifest in manifests:
        for domain in manifest.get("domains", []):
            if domain not in seen:
                ordered.append(domain)
                seen.add(domain)

    return ordered


def _load_domain_metrics(
    batch_dir: Path, domains: list[str]
) -> dict[str, dict[int, float | None]]:
    domain_metrics: dict[str, dict[int, float | None]] = {}
    for domain in domains:
        results_path = batch_dir / domain / "results.json"
        if not results_path.exists():
            domain_metrics[domain] = {k: None for k in PASS_KS}
            continue
        metrics = compute_metrics(Results.load(results_path))
        domain_metrics[domain] = {k: metrics.pass_hat_ks.get(k) for k in PASS_KS}
    return domain_metrics


def build_summary_markdown(simulations_root: Path) -> str:
    manifests = _pick_latest_by_model(_iter_completed_manifests(simulations_root))
    domains = _ordered_domains(manifests)

    headers = ["Model"]
    for domain in domains:
        for k in PASS_KS:
            headers.append(f"{domain} pass^{k}")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for manifest in manifests:
        batch_dir = simulations_root / manifest["batch_name"]
        row = [manifest["model_label"]]
        metrics_by_domain = _load_domain_metrics(batch_dir, domains)
        for domain in domains:
            for k in PASS_KS:
                row.append(_fmt_pass(metrics_by_domain[domain][k]))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def render_summary_report(
    simulations_root: Path = DEFAULT_SIMULATIONS_ROOT,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_summary_markdown(simulations_root),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a Markdown summary for completed tau2 model batches."
    )
    parser.add_argument(
        "--simulations-root",
        default=str(DEFAULT_SIMULATIONS_ROOT),
        help="Directory containing batch folders under data/simulations.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output Markdown path.",
    )
    args = parser.parse_args()

    output_path = render_summary_report(
        Path(args.simulations_root),
        Path(args.output),
    )
    print(output_path)


if __name__ == "__main__":
    main()
