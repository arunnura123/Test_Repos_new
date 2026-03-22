#!/usr/bin/env python3
"""
BMC DMA Transfer Benchmark
===========================

Measures GPU<->CPU KV cache transfer efficiency across three strategies:

  1. PAGED      — N/B separate DMA transfers (vLLM default, block_size=16)
  2. LMCACHE    — Chunked transfers in 2MB blocks (LMCache v0.12 default)
  3. CONTIGUOUS — 1 single DMA transfer per sequence (BMC)

Simulates the agentic workload scenario:
  1. Agent runs inference (KV cache on GPU)
  2. Agent calls a tool (KV cache offloaded to CPU during tool wait)
  3. Tool returns (KV cache restored to GPU)
  4. Agent continues inference

Metrics:
  - Offload latency (GPU->CPU)
  - Restore latency (CPU->GPU)
  - Throughput (GB/s)
  - DMA setup overhead at scale (1, 10, 50, 100 concurrent agents)

Usage:
  python bench_dma_transfer.py
  python bench_dma_transfer.py --model-configs llama-3-8b --seq-lens 1024 4096
  python bench_dma_transfer.py --num-agents 1 10 50 100
"""

import argparse
import math
import sys

import torch

# ─── Sanity Checks ───────────────────────────────────────────────────────────

def _sanity_check():
    if not torch.cuda.is_available():
        sys.exit("ERROR: No GPU detected. Run inside a ROCm/CUDA-enabled environment.")
    if torch.cuda.device_count() < 1:
        sys.exit("ERROR: torch.cuda.device_count() == 0.")
    try:
        t = torch.randn(1, device="cuda")
        t_cpu = t.cpu()
        del t, t_cpu
    except RuntimeError as e:
        sys.exit(f"ERROR: GPU allocation or transfer failed: {e}")

# ─── Constants ───────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "llama-3-8b":  {"num_kv_heads": 8,  "head_dim": 128, "num_layers": 32},
    "qwen2-7b":    {"num_kv_heads": 4,  "head_dim": 128, "num_layers": 28},
    "llama-3-70b": {"num_kv_heads": 8,  "head_dim": 128, "num_layers": 80},
}


class CudaTimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_event.record()

    def stop(self):
        self.end_event.record()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)


def kv_size_bytes(seq_len, num_kv_heads, head_dim, num_layers, dtype):
    elem = torch.tensor([], dtype=dtype).element_size()
    return 2 * num_layers * seq_len * num_kv_heads * head_dim * elem


def bench_paged_transfer(seq_len, block_size, num_kv_heads, head_dim,
                          num_layers, dtype, device, num_agents, num_warmup=3,
                          num_runs=10):
    """Simulate paged KV offload: N/B separate cudaMemcpyAsync calls."""
    num_blocks = math.ceil(seq_len / block_size)
    block_shape = (num_kv_heads, head_dim, block_size)

    gpu_blocks = [torch.randn(block_shape, dtype=dtype, device=device)
                  for _ in range(num_blocks * num_agents * num_layers * 2)]
    cpu_blocks = [torch.empty_like(b, device="cpu").pin_memory()
                  for b in gpu_blocks]

    timer = CudaTimer()
    total_blocks = num_blocks * num_agents * num_layers * 2

    for _ in range(num_warmup):
        for i in range(total_blocks):
            cpu_blocks[i].copy_(gpu_blocks[i], non_blocking=True)
        torch.cuda.synchronize()

    offload_times = []
    for _ in range(num_runs):
        timer.start()
        for i in range(total_blocks):
            cpu_blocks[i].copy_(gpu_blocks[i], non_blocking=True)
        ms = timer.stop()
        offload_times.append(ms)

    restore_times = []
    for _ in range(num_runs):
        timer.start()
        for i in range(total_blocks):
            gpu_blocks[i].copy_(cpu_blocks[i], non_blocking=True)
        ms = timer.stop()
        restore_times.append(ms)

    del gpu_blocks, cpu_blocks
    torch.cuda.empty_cache()

    return {
        "offload_ms": sum(offload_times) / len(offload_times),
        "restore_ms": sum(restore_times) / len(restore_times),
        "num_dma_calls": total_blocks,
    }


def bench_lmcache_transfer(seq_len, num_kv_heads, head_dim, num_layers,
                            dtype, device, num_agents, chunk_bytes=2*1024*1024,
                            num_warmup=3, num_runs=10):
    """Simulate LMCache-style chunked offload: fixed 2MB chunk DMA transfers."""
    elem = torch.tensor([], dtype=dtype).element_size()
    total_bytes = 2 * num_layers * seq_len * num_kv_heads * head_dim * elem
    total_per_agent = total_bytes
    num_chunks = math.ceil(total_per_agent / chunk_bytes)

    chunk_elems = chunk_bytes // elem
    gpu_chunks = [torch.randn(chunk_elems, dtype=dtype, device=device)
                  for _ in range(num_chunks * num_agents)]
    cpu_chunks = [torch.empty_like(c, device="cpu").pin_memory()
                  for c in gpu_chunks]

    timer = CudaTimer()
    total_chunks = num_chunks * num_agents

    for _ in range(num_warmup):
        for i in range(total_chunks):
            cpu_chunks[i].copy_(gpu_chunks[i], non_blocking=True)
        torch.cuda.synchronize()

    offload_times = []
    for _ in range(num_runs):
        timer.start()
        for i in range(total_chunks):
            cpu_chunks[i].copy_(gpu_chunks[i], non_blocking=True)
        ms = timer.stop()
        offload_times.append(ms)

    restore_times = []
    for _ in range(num_runs):
        timer.start()
        for i in range(total_chunks):
            gpu_chunks[i].copy_(cpu_chunks[i], non_blocking=True)
        ms = timer.stop()
        restore_times.append(ms)

    del gpu_chunks, cpu_chunks
    torch.cuda.empty_cache()

    return {
        "offload_ms": sum(offload_times) / len(offload_times),
        "restore_ms": sum(restore_times) / len(restore_times),
        "num_dma_calls": total_chunks,
    }


def bench_contiguous_transfer(seq_len, num_kv_heads, head_dim, num_layers,
                               dtype, device, num_agents, num_warmup=3,
                               num_runs=10):
    """Simulate contiguous KV offload: 1 cudaMemcpyAsync per sequence."""
    shape_per_agent = (num_layers * 2, seq_len, num_kv_heads, head_dim)
    gpu_bufs = [torch.randn(shape_per_agent, dtype=dtype, device=device)
                for _ in range(num_agents)]
    cpu_bufs = [torch.empty_like(b, device="cpu").pin_memory()
                for b in gpu_bufs]

    timer = CudaTimer()

    for _ in range(num_warmup):
        for i in range(num_agents):
            cpu_bufs[i].copy_(gpu_bufs[i], non_blocking=True)
        torch.cuda.synchronize()

    offload_times = []
    for _ in range(num_runs):
        timer.start()
        for i in range(num_agents):
            cpu_bufs[i].copy_(gpu_bufs[i], non_blocking=True)
        ms = timer.stop()
        offload_times.append(ms)

    restore_times = []
    for _ in range(num_runs):
        timer.start()
        for i in range(num_agents):
            gpu_bufs[i].copy_(cpu_bufs[i], non_blocking=True)
        ms = timer.stop()
        restore_times.append(ms)

    del gpu_bufs, cpu_bufs
    torch.cuda.empty_cache()

    return {
        "offload_ms": sum(offload_times) / len(offload_times),
        "restore_ms": sum(restore_times) / len(restore_times),
        "num_dma_calls": num_agents,
    }


def main():
    p = argparse.ArgumentParser(
        description="BMC DMA Transfer Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--model-configs", nargs="+",
                   choices=list(MODEL_CONFIGS.keys()),
                   default=["llama-3-8b", "qwen2-7b"])
    p.add_argument("--seq-lens", type=int, nargs="+",
                   default=[512, 1024, 2048, 4096, 8192])
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-agents", type=int, nargs="+",
                   default=[1, 10, 50, 100])
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--num-runs", type=int, default=10)
    args = p.parse_args()

    _sanity_check()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    device = "cuda"

    props = torch.cuda.get_device_properties(0)
    print("=" * 110)
    print("  BMC DMA Transfer Benchmark — Paged vs LMCache vs Contiguous KV Offload")
    print("=" * 110)
    print(f"  GPU:        {props.name}")
    print(f"  dtype:      {args.dtype}")
    print(f"  block_size: {args.block_size} (paged)")
    print(f"  runs:       {args.num_runs}")
    print("=" * 110)

    for model_name in args.model_configs:
        cfg = MODEL_CONFIGS[model_name]
        print(f"\n  Model: {model_name} "
              f"({cfg['num_layers']}L, {cfg['num_kv_heads']}KV, D={cfg['head_dim']})")
        print(f"  {'─' * 130}")
        print(f"  {'SeqLen':>7} {'Agents':>7} │"
              f" {'Paged(ms)':>10} {'LMCache(ms)':>12} {'BMC(ms)':>9} │"
              f" {'P/BMC':>6} {'LM/BMC':>7} │"
              f" {'P_calls':>8} {'LM_calls':>9} {'B_calls':>8} │"
              f" {'KV(MB)':>8} {'BW_BMC':>10}")
        print(f"  {'─' * 130}")

        for seq_len in args.seq_lens:
            for num_agents in args.num_agents:
                kv_bytes = kv_size_bytes(seq_len, cfg["num_kv_heads"],
                                         cfg["head_dim"], cfg["num_layers"],
                                         dtype) * num_agents
                kv_mb = kv_bytes / 1e6

                free_gb = torch.cuda.mem_get_info()[0] / 1e9
                if kv_mb * 4 / 1000 > free_gb * 0.8:
                    print(f"  {seq_len:>7} {num_agents:>7} │ SKIP (need {kv_mb*4/1000:.1f}GB, {free_gb:.1f}GB free)")
                    continue

                try:
                    paged = bench_paged_transfer(
                        seq_len, args.block_size, cfg["num_kv_heads"],
                        cfg["head_dim"], cfg["num_layers"], dtype, device,
                        num_agents, num_runs=args.num_runs)

                    lmcache = bench_lmcache_transfer(
                        seq_len, cfg["num_kv_heads"], cfg["head_dim"],
                        cfg["num_layers"], dtype, device, num_agents,
                        chunk_bytes=2*1024*1024, num_runs=args.num_runs)

                    contig = bench_contiguous_transfer(
                        seq_len, cfg["num_kv_heads"], cfg["head_dim"],
                        cfg["num_layers"], dtype, device, num_agents,
                        num_runs=args.num_runs)

                    p_ms = (paged["offload_ms"] + paged["restore_ms"]) / 2
                    l_ms = (lmcache["offload_ms"] + lmcache["restore_ms"]) / 2
                    c_ms = (contig["offload_ms"] + contig["restore_ms"]) / 2
                    sp_p = p_ms / c_ms if c_ms > 0 else 0
                    sp_l = l_ms / c_ms if c_ms > 0 else 0
                    bw_c = (kv_bytes / 1e9) / (c_ms / 1000) if c_ms > 0 else 0

                    print(f"  {seq_len:>7} {num_agents:>7} │"
                          f" {p_ms:>10.2f} {l_ms:>12.2f} {c_ms:>9.2f} │"
                          f" {sp_p:>5.1f}× {sp_l:>6.1f}× │"
                          f" {paged['num_dma_calls']:>8}"
                          f" {lmcache['num_dma_calls']:>9}"
                          f" {contig['num_dma_calls']:>8} │"
                          f" {kv_mb:>7.1f} {bw_c:>9.1f}")

                except Exception as e:
                    print(f"  {seq_len:>7} {num_agents:>7} │ ERROR: {str(e)[:60]}")

                torch.cuda.empty_cache()

        print(f"  {'─' * 130}")

    print("\nDone.")


if __name__ == "__main__":
    main()
