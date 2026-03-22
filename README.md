# BMC Benchmarks

Benchmarking suite for **Block-level Memory-Contiguous (BMC) KV cache management** — comparing BMC's contiguous memory layout against vLLM's paged attention and LMCache's chunked offloading for LLM inference on AMD MI300X GPUs.

## Directory Structure

```
bmc_benchmarks/
├── README.md
├── requirements.txt
├── microbenchmarks/                    # Kernel-level and memory-level benchmarks
│   ├── bench_attention_kernel.py       # Paged vs BMC attention decode kernel
│   ├── bench_dma_transfer.py           # GPU↔CPU DMA transfer: paged vs LMCache vs contiguous
│   ├── bench_alloc_sweep.py            # Latency vs reallocation count T
│   └── bench_spec_decode_table.py      # Speculative decoding latency table
├── e2e_simulations/                    # Full transformer end-to-end benchmarks
│   ├── bench_e2e_transformer.py        # Full decode loop: Paged vs BMC with real layers
│   └── bench_multiround_agentic.py     # Multi-round agentic workload simulation
└── workloads/                          # Workload definitions for multi-round benchmark
    ├── W1_multi_turn_chat.json         # 10-round conversational chat
    ├── W2_long_document_qa.json        # 5-round QA on 8K document
    └── W3_agentic_tool_calling.json    # 8 sequential tool calls with variable output
```

## Scripts Overview

### Microbenchmarks

| Script | What it measures | Key metrics |
|--------|-----------------|-------------|
| `bench_attention_kernel.py` | Decode attention kernel latency (paged vs BMC upfront/sqrt/iterative) | Decode tok/s, speedup, realloc overhead, L2 hit rate (via rocprof) |
| `bench_dma_transfer.py` | GPU↔CPU KV cache transfer latency at scale | Offload/restore ms, DMA call count, bandwidth (GB/s) |
| `bench_alloc_sweep.py` | How latency varies with reallocation count T | Normalized latency vs T for different sequence lengths |
| `bench_spec_decode_table.py` | Speculative decoding attention latency vs T | Normalized latency across T=1..512 |

### End-to-End Simulations

| Script | What it measures | Key metrics |
|--------|-----------------|-------------|
| `bench_e2e_transformer.py` | Full transformer decode (RMSNorm+QKV+Attn+FFN) | Total decode ms, tok/s, speedup, peak memory |
| `bench_multiround_agentic.py` | Multi-round inference with CPU offload/restore cycles | Per-round decode/offload/restore time, DMA count, total latency |

### Workloads

| File | Scenario | Context | Rounds |
|------|----------|---------|--------|
| `W1_multi_turn_chat.json` | Conversational assistant with offload between turns | 2048 | 10 rounds × ~300 tokens |
| `W2_long_document_qa.json` | RAG pipeline with large document context | 8192 | 5 rounds × ~200 tokens |
| `W3_agentic_tool_calling.json` | AI agent making sequential tool calls | 512 | 8 calls × 50–1000 tokens |

## Prerequisites

- **Hardware**: AMD MI300X GPU (192 GB HBM3) — works on other CUDA/ROCm GPUs with reduced accuracy
- **Software**: Python 3.8+, PyTorch 2.1+ with CUDA/ROCm support
- **Optional**: vLLM (for native `paged_attention_v1` kernel; falls back to gather+SDPA without it)
- **Optional**: rocprof (for L2 cache hit/miss profiling on AMD GPUs)

```bash
pip install -r requirements.txt
```

## Quick Start

### Run all microbenchmarks

```bash
# Attention kernel: paged vs all BMC variants
python microbenchmarks/bench_attention_kernel.py --mode all --model-config llama-3-8b

# DMA transfer comparison
python microbenchmarks/bench_dma_transfer.py

# Allocation sweep
python microbenchmarks/bench_alloc_sweep.py

# Speculative decoding latency table
python microbenchmarks/bench_spec_decode_table.py
```

### Run end-to-end simulations

```bash
# Full transformer decode: paged vs BMC
python e2e_simulations/bench_e2e_transformer.py --model-config llama-3-8b --mode both

# Multi-round agentic workloads (all 3 workloads, all 4 strategies)
python e2e_simulations/bench_multiround_agentic.py

# Quick mode (4 layers, 3 rounds per workload)
python e2e_simulations/bench_multiround_agentic.py --quick
```

## Sanity Checks

Every script includes startup sanity checks that verify:

1. **GPU availability** — exits with a clear error if no CUDA/ROCm GPU is detected
2. **GPU allocation** — tests that tensors can be allocated on the GPU
3. **Kernel availability** — probes whether vLLM's `paged_attention_v1` is functional (falls back gracefully to gather+SDPA)
4. **GQA support** — probes whether `enable_gqa=True` is supported in the installed PyTorch version
5. **Memory checks** — warns if free GPU memory is critically low before large allocations
6. **OOM handling** — catches `OutOfMemoryError` per-config and skips instead of crashing

## Strategies Compared

| Strategy | Attention Kernel | KV Layout | DMA Pattern | Reallocation |
|----------|-----------------|-----------|-------------|-------------|
| **Paged (vLLM v0.11)** | `paged_attention_v1` | Scattered blocks `[N,H,D//x,B,x]` | N/B separate DMAs | None |
| **Paged-Packed (v0.12+)** | `paged_attention_v1` | Packed physical blocks (~2MB) | 1 DMA per block | None |
| **LMCache** | Gather + SDPA | Contiguous GPU, chunked CPU | Fixed 2MB chunks | None |
| **BMC √N** | SDPA (Flash Attention) | Contiguous `[B,H,S,D]` | 1 DMA per sequence | √N reallocations |
| **BMC Upfront** | SDPA (Flash Attention) | Contiguous (pre-allocated) | 1 DMA per sequence | None |
| **BMC Iterative** | SDPA (Flash Attention) | Contiguous (grow every step) | 1 DMA per sequence | N reallocations |

## Common Options

Most scripts accept these arguments:

| Flag | Description | Default |
|------|-------------|---------|
| `--model-config` | Model architecture (llama-3-8b, qwen2-7b, etc.) | llama-3-8b |
| `--dtype` | KV cache precision (`bfloat16` or `float16`) | bfloat16 |
| `--batch-size` | Batch size | 8 |
| `--block-size` | Paged attention block size | 16 |
| `--num-layers` | Override layer count (0 = model default) | 0 |
| `--num-runs` | Repetitions for statistical significance | 3 |
| `--csv` | Export results to CSV (attention kernel only) | None |
| `--quick` | Reduced config for fast iteration | Off |

## File Naming Convention

| Prefix | Meaning |
|--------|---------|
| `bench_` | Benchmark script (runnable) |
| `W{N}_` | Workload definition file |