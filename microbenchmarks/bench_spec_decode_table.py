#!/usr/bin/env python3
"""
Speculative Decoding Latency Table
====================================

Attention block latency vs T for speculative decoding.
Llama-3-8B, BS=8, Q=8, N=4096, acceptance=8 -> 512 speculative steps.

T = number of reallocations over 512 steps.
  T=1   : upfront (capacity = ctx+N, zero realloc)
  T=16  : BMC sqrt(512) ~ 22 (grow every ~32 steps)
  T=512 : iterative (realloc every step)

Output: single-row table of normalized latency.

Usage:
  python bench_spec_decode_table.py
"""

import math
import statistics
import sys

import torch
import torch.nn.functional as F

# ─── Sanity Checks ───────────────────────────────────────────────────────────

def _sanity_check():
    if not torch.cuda.is_available():
        sys.exit("ERROR: No GPU detected. Run inside a ROCm/CUDA-enabled environment.")
    try:
        q = torch.randn(1, 1, 1, 128, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 1, 4, 128, dtype=torch.bfloat16, device="cuda")
        F.scaled_dot_product_attention(q, k, k, scale=1.0, is_causal=False, enable_gqa=True)
        del q, k
    except Exception as e:
        sys.exit(f"ERROR: SDPA with enable_gqa failed: {e}")

# ─── Constants ───────────────────────────────────────────────────────────────

torch.manual_seed(42)

H = 32
H_KV = 8
D = 128
LAYERS = 32
DTYPE = torch.bfloat16
WARMUP = 5
NUM_RUNS = 3

CONTEXT = 256
N = 4096
ACCEPT = 8
Q_TOKENS = 8
BATCH = 8
STEPS = N // ACCEPT  # 512


class CudaTimer:
    def __init__(self):
        self.s = torch.cuda.Event(enable_timing=True)
        self.e = torch.cuda.Event(enable_timing=True)
    def start(self): self.s.record()
    def stop(self):
        self.e.record(); torch.cuda.synchronize()
        return self.s.elapsed_time(self.e)


def run_T(T, device):
    """Run 512 speculative steps with T total reallocations."""
    scale = D ** -0.5

    if T <= 1:
        grow_chunk = 0
        capacity = CONTEXT + N
    else:
        grow_chunk = max(1, STEPS // T)
        capacity = CONTEXT + grow_chunk * ACCEPT

    seq_len = CONTEXT
    kv = [(torch.randn(BATCH, H_KV, capacity, D, dtype=DTYPE, device=device),
           torch.randn(BATCH, H_KV, capacity, D, dtype=DTYPE, device=device))
          for _ in range(LAYERS)]
    q = torch.randn(BATCH, H, Q_TOKENS, D, dtype=DTYPE, device=device)
    timer = CudaTimer()

    for _ in range(WARMUP):
        for k, v in kv:
            F.scaled_dot_product_attention(q, k, v, scale=scale,
                                           is_causal=False, enable_gqa=True)
    torch.cuda.synchronize()

    seq_len = CONTEXT
    cap = capacity
    timer.start()
    for step in range(STEPS):
        seq_len += ACCEPT
        if grow_chunk > 0 and seq_len > cap:
            new_cap = cap + grow_chunk * ACCEPT
            new_kv = []
            for k, v in kv:
                nk = torch.empty(BATCH, H_KV, new_cap, D, dtype=DTYPE, device=device)
                nv = torch.empty(BATCH, H_KV, new_cap, D, dtype=DTYPE, device=device)
                nk[:, :, :seq_len - ACCEPT, :] = k[:, :, :seq_len - ACCEPT, :]
                nv[:, :, :seq_len - ACCEPT, :] = v[:, :, :seq_len - ACCEPT, :]
                new_kv.append((nk, nv))
            del kv
            kv = new_kv
            cap = new_cap

        for k, v in kv:
            F.scaled_dot_product_attention(q, k, v, scale=scale,
                                           is_causal=False, enable_gqa=True)
    total_ms = timer.stop()

    del kv; torch.cuda.empty_cache()
    return total_ms


def main():
    _sanity_check()

    device = "cuda"

    props = torch.cuda.get_device_properties(0)
    print("=" * 70)
    print(f"  Spec. Decoding Latency vs T (Llama-3-8B)")
    print("=" * 70)
    print(f"  GPU:    {props.name}")
    print(f"  BS={BATCH}, ctx={CONTEXT}, N={N}, Q={Q_TOKENS}, "
          f"accept={ACCEPT}, steps={STEPS}")
    print("=" * 70)

    T_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    print(f"\n  {'T':>6} {'ms':>10} {'Normalized':>12} {'Note':>8}")
    print(f"  {'─' * 40}")

    results = {}
    baseline = None
    for T in T_values:
        times = []
        for _ in range(NUM_RUNS):
            torch.cuda.empty_cache()
            times.append(run_T(T, device))
        avg = statistics.mean(times)
        results[T] = avg
        if baseline is None:
            baseline = avg
        norm = avg / baseline
        note = ""
        if T == 1: note = "Upfront"
        elif T == int(math.sqrt(STEPS)): note = "sqrt(S) *"
        elif T == STEPS: note = "Iter"
        print(f"  {T:>6} {avg:>10.1f} {norm:>12.3f} {note:>8}", flush=True)

    min_T = min(results, key=results.get)
    sqrt_steps = int(math.sqrt(STEPS))
    print(f"\n  Optimal T={min_T} ({results[min_T]/baseline:.3f}x), sqrt({STEPS})={sqrt_steps}")

    header = " & ".join([str(T) for T in T_values])
    vals = " & ".join([
        f"\\textbf{{{results[T]/baseline:.3f}}}" if T == min_T
        else f"{results[T]/baseline:.2f}"
        for T in T_values
    ])
    print(f"\n  LaTeX:")
    print(f"  $T$ & {header} \\\\")
    print(f"  Latency & {vals} \\\\")

    print("\nDone.")


if __name__ == "__main__":
    main()
