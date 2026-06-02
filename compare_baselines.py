#!/usr/bin/env python3
"""
Compare baseline results across nodes.
Point it at two result JSONs and it produces a side-by-side quality comparison.

Usage:
    python3 compare_baselines.py results/baseline_3080ti_*.json results/baseline_t1000_*.json
"""

import json
import sys
import glob
from pathlib import Path


def load_result(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compare(files: list[str]):
    results = [load_result(f) for f in files]

    # Collect all experiment labels across all nodes
    all_exps = []
    for r in results:
        for exp_id, exp in r["experiments"].items():
            all_exps.append((r["node_id"], exp_id, exp))

    print(f"\n{'='*72}")
    print(f"  CROSS-NODE BASELINE COMPARISON")
    print(f"{'='*72}")

    # Header
    print(f"\n  {'Node':<28} {'Exp':<4} {'Label':<25} {'t/s':>6} {'KW%':>6} {'Lat':>7}")
    print(f"  {'-'*68}")

    for node_id, exp_id, exp in all_exps:
        s = exp.get("summary", {})
        tps     = s.get("avg_tokens_per_sec")
        kw      = s.get("avg_keyword_score")
        latency = s.get("avg_elapsed_sec")
        label   = exp["label"]

        tps_str = f"{tps:.1f}" if tps else "—"
        kw_str  = f"{kw*100:.0f}%" if kw is not None else "—"
        lat_str = f"{latency:.1f}s" if latency else "—"

        print(f"  {node_id:<28} {exp_id:<4} {label:<25} {tps_str:>6} {kw_str:>6} {lat_str:>7}")

    # Per-prompt breakdown
    print(f"\n\n  PER-PROMPT QUALITY BREAKDOWN")
    print(f"  {'='*68}")

    prompt_ids = [p["id"] for p in [
        {"id": "factual_simple"},
        {"id": "factual_technical"},
        {"id": "reasoning_simple"},
        {"id": "reasoning_multi_step"},
        {"id": "code_generation"},
        {"id": "long_context"},
    ]]

    for pid in prompt_ids:
        print(f"\n  [{pid}]")
        for node_id, exp_id, exp in all_exps:
            for p in exp["prompts"]:
                if p["prompt_id"] == pid:
                    q = p.get("quality")
                    inf = p["inference"]
                    if inf["success"] and q:
                        kw   = f"{q['keyword_hits']}/{q['keyword_total']}"
                        coh  = "OK" if q["coherent"] else "!INCOHERENT"
                        lat  = inf.get("elapsed_sec", "?")
                        out  = inf.get("output", "")[:80].replace("\n", " ").strip()
                        print(f"    {node_id}/{exp_id}: {kw} keywords | {lat}s | {coh}")
                        print(f"      \"{out}...\"")
                    else:
                        err = inf.get("error", "unknown")[:50]
                        print(f"    {node_id}/{exp_id}: FAILED — {err}")

    # Attestation summary
    print(f"\n\n  ATTESTATION HASHES")
    print(f"  {'='*68}")
    for r in results:
        a = r.get("attestation", {})
        print(f"  {r['node_id']}:")
        print(f"    Binary: {a.get('binary_hash','?')[:32]}...")
        print(f"    Model:  {a.get('model_hash','?')[:32]}...")

    hashes = [r.get("attestation", {}).get("model_hash") for r in results]
    if len(set(h for h in hashes if h)) == 1:
        print(f"\n  ✓ Model hashes match across all nodes — same weights confirmed")
    else:
        print(f"\n  ✗ Model hash mismatch — nodes may be running different weights!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Try to find results automatically
        result_files = sorted(glob.glob("results/baseline_*.json"))
        if not result_files:
            result_files = sorted(glob.glob(
                str(Path.home() / "distributed-inference/results/baseline_*.json")
            ))
        if not result_files:
            print("Usage: python3 compare_baselines.py <result1.json> <result2.json> ...")
            sys.exit(1)
        print(f"Found {len(result_files)} result file(s): {[Path(f).name for f in result_files]}")
    else:
        result_files = sys.argv[1:]

    compare(result_files)
