#!/usr/bin/env python3
"""
Baseline runner for distributed inference experiment.
Tests multiple GPU layer configurations and logs quality/performance metrics.

Run on each machine to establish baselines before building the coordinator.
"""

import subprocess
import json
import time
import os
import sys
import hashlib
import platform
import datetime
from pathlib import Path

# -------------------------------------------------------------
# Config
# -------------------------------------------------------------

INSTALL_DIR = Path.home() / "distributed-inference"
CONFIG_FILE = INSTALL_DIR / "node_config.json"
RESULTS_DIR = INSTALL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Prompts designed to test different failure modes:
# - factual recall (hallucination shows up here first under quantization pressure)
# - reasoning (multi-step logic degrades with fewer layers in VRAM)
# - long context (quality drops as context fills KV cache)
TEST_PROMPTS = [
    {
        "id": "factual_simple",
        "category": "factual",
        "prompt": "[INST] What is the capital of France, and what is the approximate population of that city? [/INST]",
        "expected_keywords": ["paris", "million"],
    },
    {
        "id": "factual_technical",
        "category": "factual",
        "prompt": "[INST] Explain what a transformer neural network is in 3 sentences. [/INST]",
        "expected_keywords": ["attention", "layer", "token"],
    },
    {
        "id": "reasoning_simple",
        "category": "reasoning",
        "prompt": "[INST] If I have 3 boxes with 4 apples each, and I give away 5 apples total, how many apples do I have left? Show your work. [/INST]",
        "expected_keywords": ["12", "7"],
    },
    {
        "id": "reasoning_multi_step",
        "category": "reasoning",
        "prompt": "[INST] A train leaves city A at 9am going 60mph. Another leaves city B at 10am going 80mph toward city A. The cities are 280 miles apart. At what time do the trains meet? [/INST]",
        "expected_keywords": ["11", "12"],
    },
    {
        "id": "code_generation",
        "category": "code",
        "prompt": "[INST] Write a Python function that takes a list of integers and returns only the prime numbers. [/INST]",
        "expected_keywords": ["def", "return", "for"],
    },
    {
        "id": "long_context",
        "category": "long_context",
        "prompt": "[INST] " + ("The following is background context: " + "The quick brown fox jumps over the lazy dog. " * 50) + "\n\nBased on the above text, what animal jumps over what other animal? [/INST]",
        "expected_keywords": ["fox", "dog"],
    },
]

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def load_config() -> dict:
    if not CONFIG_FILE.exists():
        print(f"[ERROR] Config not found at {CONFIG_FILE}")
        print("        Run setup_3080ti.sh first.")
        sys.exit(1)
    with open(CONFIG_FILE) as f:
        return json.load(f)


def hash_file(path: str) -> str:
    """SHA256 of binary — used for software attestation later."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_gpu_stats() -> dict:
    """Grab current GPU state before each run."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.free,temperature.gpu,clocks.current.graphics",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        return {
            "name":        parts[0],
            "vram_total":  int(parts[1]),
            "vram_free":   int(parts[2]),
            "temperature": int(parts[3]),
            "clock_mhz":   int(parts[4]),
        }
    except Exception as e:
        return {"error": str(e)}


def score_output(output: str, expected_keywords: list) -> dict:
    """
    Simple quality scoring:
    - keyword_hits:  how many expected keywords appear in output
    - length:        output token count proxy
    - coherent:      heuristic — does it look like a real sentence
    """
    output_lower = output.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in output_lower)
    words = output.split()

    # Incoherence signals: very short, repetitive, or garbled
    unique_ratio = len(set(words)) / max(len(words), 1)
    coherent = len(words) > 5 and unique_ratio > 0.3

    return {
        "keyword_hits":    hits,
        "keyword_total":   len(expected_keywords),
        "keyword_score":   round(hits / max(len(expected_keywords), 1), 2),
        "output_words":    len(words),
        "unique_ratio":    round(unique_ratio, 3),
        "coherent":        coherent,
    }


def run_inference(llama_bin: str, model_path: str, prompt: str,
                  gpu_layers: int, n_predict: int = 256) -> dict:
    """
    Run a single inference call and return timing + output.
    """
    cmd = [
        llama_bin,
        "-m",        model_path,
        "-p",        prompt,
        "--n-predict", str(n_predict),
        "--gpu-layers", str(gpu_layers),
        "--ctx-size", "2048",
        "--temp",    "0.1",       # low temp for reproducibility
        "--repeat-penalty", "1.1",
        "--log-disable",          # suppress llama.cpp internal logs
        "-no-cnv",                # disable conversation mode
    ]

    gpu_before = get_gpu_stats()
    t_start = time.perf_counter()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        t_end = time.perf_counter()
        elapsed = round(t_end - t_start, 3)

        if result.returncode != 0:
            return {
                "success": False,
                "error":   result.stderr[-500:],
                "elapsed": elapsed,
            }

        # llama.cpp prints the prompt then the completion
        # strip the prompt from the output
        full_output = result.stdout.strip()
        # Try to isolate just the generated text after [/INST]
        if "[/INST]" in full_output:
            generated = full_output.split("[/INST]")[-1].strip()
        else:
            generated = full_output

        # Parse timing stats from llama.cpp stderr
        tokens_per_sec = None
        for line in result.stderr.split("\n"):
            if "eval time" in line and "tokens per second" in line:
                try:
                    tokens_per_sec = float(line.split("(")[-1].split(" t/s")[0].strip())
                except Exception:
                    pass

        return {
            "success":        True,
            "output":         generated,
            "elapsed_sec":    elapsed,
            "tokens_per_sec": tokens_per_sec,
            "gpu_before":     gpu_before,
            "gpu_after":      get_gpu_stats(),
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout", "elapsed": 300}
    except Exception as e:
        return {"success": False, "error": str(e), "elapsed": 0}


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def run_baselines():
    config = load_config()

    llama_bin   = config["llama_cpp_bin"]
    model_path  = config["model_path"]
    node_id     = config["node_id"]
    experiments = config["experiments"]

    # Software attestation hash — store this, it's the seed for later verification
    print(f"\n[ATTEST] Hashing inference binary...")
    bin_hash   = hash_file(llama_bin)
    model_hash = hash_file(model_path)   # this will take ~10s for a 4GB file
    print(f"         Binary: {bin_hash[:16]}...")
    print(f"         Model:  {model_hash[:16]}...")

    all_results = {
        "node_id":     node_id,
        "timestamp":   datetime.datetime.utcnow().isoformat(),
        "platform":    platform.platform(),
        "attestation": {
            "binary_hash": bin_hash,
            "model_hash":  model_hash,
            "binary_path": llama_bin,
            "model_path":  model_path,
        },
        "experiments": {}
    }

    for exp_id, exp_cfg in experiments.items():
        label      = exp_cfg["label"]
        gpu_layers = exp_cfg["gpu_layers"]

        print(f"\n{'='*60}")
        print(f"  Experiment {exp_id}: {label}")
        print(f"  GPU layers: {gpu_layers}")
        print(f"{'='*60}")

        exp_results = {
            "label":      label,
            "gpu_layers": gpu_layers,
            "prompts":    []
        }

        for prompt_cfg in TEST_PROMPTS:
            pid = prompt_cfg["id"]
            print(f"\n  → [{pid}] ", end="", flush=True)

            result = run_inference(
                llama_bin, model_path,
                prompt_cfg["prompt"],
                gpu_layers
            )

            if result["success"]:
                quality = score_output(result["output"], prompt_cfg["expected_keywords"])
                print(f"{result['elapsed_sec']}s | "
                      f"keywords {quality['keyword_hits']}/{quality['keyword_total']} | "
                      f"{'OK' if quality['coherent'] else 'INCOHERENT'}")
                print(f"     Output: {result['output'][:120].strip()}...")
            else:
                quality = None
                print(f"FAILED — {result.get('error','unknown')[:60]}")

            exp_results["prompts"].append({
                "prompt_id":  pid,
                "category":   prompt_cfg["category"],
                "inference":  result,
                "quality":    quality,
            })

        # Summary for this experiment
        if exp_results["prompts"]:
            successes = [p for p in exp_results["prompts"] if p["inference"]["success"]]
            if successes:
                avg_elapsed = round(
                    sum(p["inference"]["elapsed_sec"] for p in successes) / len(successes), 2
                )
                avg_keyword = round(
                    sum(p["quality"]["keyword_score"] for p in successes if p["quality"]) /
                    max(len(successes), 1), 2
                )
                avg_tps = None
                tps_vals = [p["inference"].get("tokens_per_sec") for p in successes
                            if p["inference"].get("tokens_per_sec")]
                if tps_vals:
                    avg_tps = round(sum(tps_vals) / len(tps_vals), 1)

                exp_results["summary"] = {
                    "success_rate":     round(len(successes) / len(exp_results["prompts"]), 2),
                    "avg_elapsed_sec":  avg_elapsed,
                    "avg_keyword_score": avg_keyword,
                    "avg_tokens_per_sec": avg_tps,
                }

                print(f"\n  Summary: {avg_elapsed}s avg | "
                      f"keyword score {avg_keyword} | "
                      f"{avg_tps or '?'} t/s")

        all_results["experiments"][exp_id] = exp_results

    # Save results
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"baseline_{node_id}_{ts}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to: {out_file}")
    print_summary(all_results)


def print_summary(results: dict):
    print(f"\n{'='*60}")
    print(f"  BASELINE SUMMARY — {results['node_id']}")
    print(f"{'='*60}")
    print(f"  {'Experiment':<25} {'Avg t/s':>8} {'Keyword':>8} {'Latency':>8}")
    print(f"  {'-'*55}")

    for exp_id, exp in results["experiments"].items():
        s = exp.get("summary", {})
        tps     = s.get("avg_tokens_per_sec", "—")
        kw      = s.get("avg_keyword_score", "—")
        latency = s.get("avg_elapsed_sec", "—")
        label   = exp["label"]
        print(f"  {label:<25} {str(tps):>8} {str(kw):>8} {str(latency):>8}s")

    print(f"\n  Attestation hashes stored in results JSON.")
    print(f"  Send results JSON to coordinator to register this node.")


if __name__ == "__main__":
    run_baselines()
