# Running the live-stream editor on an RTX 4090: approaches compared

**Problem.** The streaming server's smallest published profile (RTX 5090, 32 GB,
`JOYOMNI_LOW_VRAM=1`) measured ~21.5 GiB resident and ~28 GiB peak at 840x480.
An RTX 4090 has 24 GB (~23.5 GiB usable, less with a desktop on it) and is an
Ada `sm_89` part. The kernel stack already covers `sm_89`: the cutlass FP8 GEMM
in `joyomni_ops` has an `sm89` path, and SageAttention 2 ships `sm89` kernels
that the bundled CUDA-graph patch already touches. So the blocker is **memory**,
with throughput second.

**Status:** none of this has run on a real 4090 yet. Numbers marked *computed*
come from the code. Numbers marked *estimated* use the cost/budget models in
`experiments/`, calibrated on the one published 5090 data point.
`deploy/tools/probe_rtx4090.py` is the first thing to run on the card.

## Where the memory goes (*computed*)

| item | size |
|---|---|
| DiT FP8 Linears, 40 blocks x (img + txt) | 15.00 GiB (0.375 GiB per block) |
| other DiT params (biases, modulation, embedders, FP8 scales) | 0.30 GiB |
| KV pool + CUDA-graph staging, 840x480 (700 tokens / latent frame, 640 KiB per token) | ~3.0 GiB |
| same at 672x384 (448 tokens) | ~1.9 GiB |
| RV2V reference image KV (32x32 latent) | +1.25 GiB |
| VAE + CUDA context + activations + workspaces | ~3.2 GiB (*estimated*: remainder of the 5090 number) |

The text stream cannot be dropped from VRAM. Text tokens attend jointly with
image tokens in every block, so the txt stream is recomputed at every denoise
step (`dit.py`, `skip_text_stream` is only used by KV-store / ref-prefill
forwards).

## Angles considered

| # | approach | VRAM saved | cost | risk | status |
|---|---|---|---|---|---|
| A | **Stream N FP8 DiT blocks from pinned host RAM** (`deploy/xvideo/blockswap.py`) | 0.375 GiB/block minus 0.75 GiB of slots | ~0.94 GiB PCIe per block per chunk; hidden behind compute up to ~6-8 blocks at 480p (*estimated*) | low: exact (same kernels, same weights); graph-capturable by construction, checked by the probe | **implemented, default in `run_server_4090.sh`** |
| B | **W4A16 INT4 text stream** (tinygemm) | ~3.5 GiB (txt stream 7.5 → ~4.0 GiB incl. group scales) | no PCIe; tinygemm at M≈400 may be slower than the FP8 GEMM; RTN int4 quality loss on text conditioning unknown | medium-high: changes numerics of every text token | prototype in `prototypes/int4_txt_stream/`, not wired in |
| C | **Lower resolution presets** (672x384) | ~1.1 GiB of KV/graph + smaller activations | lower output resolution | none | in presets `fast` / `safe` |
| D | **Meta-device DiT build + mmap adopt** | 0 VRAM, ~55 GB → page cache of *host* RAM at load | none | low; falls back to the old path on any error | implemented (`JOYOMNI_DIT_META_INIT=1`) |
| E | FP8 KV pool / graph buffers | ~1.5 GiB at 480p | attention would need FP8 K/V (Sage takes INT8 Q/K but bf16 V input), quantization error accumulates through the KV window | high | idea only |
| F | *(unconventional)* Freeze the txt stream after denoise step 1 and reuse its per-layer K/V for later steps, then stream the whole txt half | up to ~7 GiB | changes the model's math; the distilled 2-step sampler may not tolerate it | very high | idea only: needs a quality A/B on LongV2VBench before any code |
| G | Run the 7B text encoder on CPU | 0 resident VRAM (already offloaded) | multi-second prompt changes | n/a | already covered by `JOYOMNI_TE_OFFLOAD` |

Finding D surfaced while reading the loader. `MMDoubleStreamBlock` creates its
`FeedForward`s without `dtype`/`device`, so in the existing low-VRAM path the
MLPs (about 10.7B params) are first allocated in **fp32 on the CPU**. That is
~43 GB plus ~11 GB of bf16 for the rest, before the cast. Many 4090 desktops
have 32-64 GB of RAM, so this mattered as much as VRAM.

## Estimated budgets (`experiments/c_vram_budget/estimate.py`)

| config | est. resident | headroom on 23.5 GiB |
|---|---|---|
| 5090 reference 840x480, nothing streamed | 21.50 | +2.00 (not enough: 5090 peak was ~6.5 above resident) |
| 4090 balanced 840x480, 8 streamed | 19.25 | +4.25 |
| 4090 balanced + RV2V | 20.50 | +3.00 |
| 4090 fast 672x384, 4 streamed | 19.10 | +4.40 |
| 4090 safe 672x384, 10 streamed | 16.85 | +6.65 |

![VRAM budget](../visuals/rtx4090-vram-budget.svg)

The 5090's ~28 GiB "peak" is a caching-allocator high-water mark taken under a
30 GiB cap. How much of the ~6.5 GiB above resident is real transient demand
(graph-capture warm run, VAE max-autotune, prompt encoding) and how much is lazily
held cache is unknown. That is the biggest uncertainty here, and why the
presets keep ≥4 GiB free.

## Throughput (`experiments/a_block_streaming/cost_model.py`, *estimated*)

Assuming 180 sustained FP8 TFLOPS, 22 GB/s pinned H2D, 2 steps, ~400 text tokens
and 8 source frames per latent chunk. The 8 frames/chunk is inferred: it is the
only value consistent with the published 5090 24 FPS.

| config | DiT-bound FPS |
|---|---|
| 840x480, 0 streamed | ~20.8 |
| 840x480, 8 streamed | ~18.1 |
| 672x384, 4 streamed | ~29 |
| 672x384, 10 streamed | ~15 |

So a 4090 should land around **18-20 FPS at 840x480**, with 24 FPS realistic at
672x384. That follows from the 4090's lower FP8 throughput (spec-sheet FP8 dense with fp16 accumulate:
~330 TFLOPS vs ~419 on the 5090), not from streaming.

## Decision

Ship **A + C + D** as the 4090 profile (`deploy/run_server_4090.sh`, presets
`balanced` / `fast` / `safe`). A and D are exact: they move bytes, never change
math. C is a user-visible choice. Keep **B** as the escape hatch if a card
cannot hold even the `safe` preset (for example a 4090 that also drives two
4K monitors), and re-evaluate **F** only with a quality benchmark.

## Next experiments (on the 4090)

1. Run `deploy/tools/probe_rtx4090.py`. Record cutlass vs fast-accum TFLOPS,
   Sage vs SDPA, H2D GB/s. Feed the measured TFLOPS/GB/s into `cost_model.py`.
2. Launch `balanced` with `--profile-timings` (`bash deploy/run_server_4090.sh --profile-timings`).
   Log `runtime loaded; device memory used` and `torch.cuda.max_memory_allocated`
   after a 2-minute session with and without an RV2V reference.
3. Sweep `JOYOMNI_DIT_OFFLOAD_BLOCKS` in {4, 6, 8, 10, 12} at 840x480. Plot
   per-chunk `dit=` time vs offload count to find the knee and re-fit the cost model.
4. Try `JOYOMNI_DIT_OFFLOAD_SLOTS=3`: one more 0.375 GiB slot gives every copy a
   longer hiding window. It should flatten the knee if the copies are latency-bound
   rather than bandwidth-bound.
5. Prototype B on the GPU. On CPU with random Gaussian weights (K=N=1024),
   RTN int4 g128 already gives **~9.8% relative error vs bf16**, against ~3.7%
   for the FP8 path (`joyomni_ops/README.md`). That is ~2.6x the noise on every
   text token, a real quality risk to A/B before anything else.
   Run `python prototypes/int4_txt_stream/int4_linear.py` on the GPU:
   check `rel_err_tinygemm_vs_dequant` ≈ 0 (packing convention), then time
   M=400 int4 vs FP8, then a visual A/B on the README showcase prompts.
6. Stream blocks in priority order rather than evenly spaced. A non-uniform
   schedule (keep more blocks resident near the start of the network, where
   there is less compute ahead to hide behind) may hide more copies.
