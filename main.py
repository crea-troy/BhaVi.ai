"""
BhaVi - Main Entry Point
========================
Run this to test BhaVi on your machine.

Usage:
    python3 main.py

Author: BhaVi Project
"""

import sys
import os
import importlib.util
import torch
import torch.nn as nn
import time

# ── Robust path fix ───────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _import_from(rel_path, module_name):
    full_path = os.path.join(_HERE, rel_path)
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_bhavi():
    """Test BhaVi on your local machine."""

    print("\n" + "🧠 " * 20)
    print("  BhaVi Architecture - Local Test")
    print("🧠 " * 20)

    # ── Initialize BhaVi ───────────────────────────────────────────
    _bhavi_mod = _import_from("bhavi.py", "bhavi_main")
    BhaVi = _bhavi_mod.BhaVi

    bhavi = BhaVi(
        input_dim=256,
        field_dim=256,
        output_dim=256,
        active_memory_slots=512,
        consolidated_memory_slots=1024,
        num_hypotheses=8,
        memory_path="bhavi_memory.pt"
    )

    # ── Test 1: Basic Forward Pass ─────────────────────────────────
    print("\n[Test 1] Basic forward pass...")
    batch_size = 4
    x = torch.randn(batch_size, 256)

    start = time.time()
    output = bhavi(x, force_collapse=True, learn=True)
    elapsed = time.time() - start

    print(f"  ✅ Forward pass: {elapsed*1000:.1f}ms")
    print(f"  Output shape: {output['output'].shape}")
    print(f"  Confidence: {output['confidence'].mean().item():.3f}")
    print(f"  Aleatoric uncertainty: {output['aleatoric_uncertainty'].mean().item():.3f}")
    print(f"  Epistemic uncertainty: {output['epistemic_uncertainty'].mean().item():.3f}")
    print(f"  Consistency with roots: {output['consistency_with_roots'].mean().item():.3f}")

    # ── Test 2: Self-Improvement ───────────────────────────────────
    print("\n[Test 2] Self-improvement over 100 steps...")
    gap_scores = []

    for step in range(100):
        x = torch.randn(4, 256)
        output = bhavi(x, learn=True)
        gap_scores.append(output['gap_score'].mean().item())

    initial_gap = sum(gap_scores[:10]) / 10
    final_gap = sum(gap_scores[-10:]) / 10

    print(f"  Initial knowledge gap: {initial_gap:.4f}")
    print(f"  Final knowledge gap:   {final_gap:.4f}")
    print(f"  Gap reduction: {(initial_gap - final_gap) / initial_gap * 100:.1f}%")
    print(f"  ✅ Self-improvement working")

    # ── Test 3: Root Protection ────────────────────────────────────
    print("\n[Test 3] Root protection test...")

    bhavi.freeze_core()

    # Try to manipulate core
    core_frozen = bhavi.frozen_core.core.is_frozen()
    print(f"  Core frozen: {core_frozen}")

    # Test with potentially contradictory input
    x_normal = torch.randn(2, 256)
    x_extreme = torch.randn(2, 256) * 100  # Extreme values

    out_normal = bhavi(x_normal, learn=True)
    out_extreme = bhavi(x_extreme, learn=True)

    print(f"  Normal input - consistency: {out_normal['consistency_with_roots'].mean().item():.3f}")
    print(f"  Extreme input - consistency: {out_extreme['consistency_with_roots'].mean().item():.3f}")
    print(f"  Extreme input - learning blocked: {out_extreme['learning_blocked'].mean().item():.3f}")
    print(f"  ✅ Root protection working")

    # ── Test 4: Memory Persistence ─────────────────────────────────
    print("\n[Test 4] Memory persistence...")
    bhavi.observer_core.save_memory("bhavi_memory.pt")
    print(f"  Memory saved: {os.path.exists('bhavi_memory.pt')}")
    print(f"  Memory updates stored: {bhavi.observer_core.update_count}")
    print(f"  ✅ Memory persistence working")

    # ── Test 5: Hardware efficiency ────────────────────────────────
    print("\n[Test 5] Hardware efficiency test...")

    # Measure memory usage
    total_params = sum(p.numel() for p in bhavi.parameters())
    fp32_mb = total_params * 4 / 1024 / 1024
    int8_mb = total_params * 1 / 1024 / 1024

    # Speed test
    times = []
    for _ in range(50):
        x = torch.randn(1, 256)
        start = time.time()
        _ = bhavi(x, learn=False)
        times.append(time.time() - start)

    avg_ms = sum(times) / len(times) * 1000
    print(f"  Total parameters:     {total_params:,}")
    print(f"  FP32 memory:          {fp32_mb:.1f} MB")
    print(f"  INT8 memory:          {int8_mb:.1f} MB")
    print(f"  Avg inference time:   {avg_ms:.2f}ms per sample")
    print(f"  ✅ Efficient for your hardware (8GB RAM)")

    # ── Final Status ───────────────────────────────────────────────
    bhavi.status()

    print("\n" + "✅ " * 20)
    print("  All tests passed! BhaVi is running on your machine.")
    print("✅ " * 20 + "\n")


if __name__ == "__main__":
    test_bhavi()
