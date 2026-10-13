#!/usr/bin/env python
"""How many DiT blocks can be streamed from host RAM before the copies stop hiding
behind compute?  First-order model, no GPU needed:

    python experiments/a_block_streaming/cost_model.py
    python experiments/a_block_streaming/cost_model.py --h2d-gbps 24 --tflops 200

Per chunk the server runs `steps` denoise forwards (img + txt streams, tokens =
current latent frame + source/ref latent frame + text) and one KV-store forward
(img stream only, current frame only). deploy/xvideo/blockswap.py copies the
img half of an offloaded block for every forward and the txt half only for the
denoise forwards.

Assumptions (all overridable): effective FP8 GEMM throughput `--tflops` (the
4090 peak is 330 dense with fp16 accumulate / 165 with fp32 accumulate --
realistic sustained is lower; measure it with deploy/tools/probe_rtx4090.py),
GEMMs dominate the DiT time, copies overlap perfectly with compute until the
copy engine is saturated. Attention, VAE and the rest of the pipeline run in
parallel worker threads and are ignored here.
"""
from __future__ import annotations

import argparse

HIDDEN, MLP, DEPTH = 4096, 4, 40
PARAMS_PER_STREAM = HIDDEN * 3 * HIDDEN + HIDDEN * HIDDEN + 2 * HIDDEN * MLP * HIDDEN  # 201M
STREAM_BYTES = PARAMS_PER_STREAM  # FP8


def latent_tokens(w, h):
    return (h * 2 // 3 // 16) * (w * 2 // 3 // 16)


def chunk_cost(w, h, txt_tokens, steps, tflops, h2d_gbps, offload):
    ct = latent_tokens(w, h)
    img_tok_denoise = 2 * ct           # current + source frame (ref_video_latent)
    flops_block_denoise = 2 * PARAMS_PER_STREAM * (img_tok_denoise + txt_tokens)
    flops_block_store = 2 * PARAMS_PER_STREAM * ct
    per_forward_ms = [flops_block_denoise * DEPTH / (tflops * 1e12) * 1e3] * steps + \
                     [flops_block_store * DEPTH / (tflops * 1e12) * 1e3]
    compute_ms = sum(per_forward_ms)
    bytes_per_block = steps * 2 * STREAM_BYTES + STREAM_BYTES
    copy_ms = offload * bytes_per_block / (h2d_gbps * 1e9) * 1e3
    # copies can only hide behind the resident blocks' compute
    hide_ms = compute_ms * (DEPTH - offload) / DEPTH
    exposed_ms = max(0.0, copy_ms - hide_ms)
    return ct, compute_ms, copy_ms, exposed_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflops", type=float, default=180.0, help="sustained FP8 GEMM TFLOPS")
    ap.add_argument("--h2d-gbps", type=float, default=22.0, help="pinned H2D GB/s (PCIe 4.0 x16 ~22-25)")
    ap.add_argument("--txt-tokens", type=int, default=400)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--frames-per-chunk", type=int, default=8,
                    help="source frames per latent chunk (VAE temporal factor; check the server log)")
    args = ap.parse_args()

    print(f"assumed {args.tflops:.0f} TFLOPS sustained, {args.h2d_gbps:.0f} GB/s H2D, "
          f"{args.steps} steps, {args.txt_tokens} text tokens, {args.frames_per_chunk} frames/chunk\n")
    print("| resolution | offloaded blocks | DiT compute ms/chunk | H2D ms/chunk | exposed ms | DiT-bound FPS |")
    print("|---|---|---|---|---|---|")
    for (w, h) in [(840, 480), (672, 384)]:
        for off in (0, 4, 6, 8, 12, 16, 20):
            ct, comp, copy, exp = chunk_cost(w, h, args.txt_tokens, args.steps, args.tflops, args.h2d_gbps, off)
            fps = args.frames_per_chunk / ((comp + exp) / 1e3)
            print(f"| {w}x{h} ({ct} tok) | {off} | {comp:.0f} | {copy:.0f} | {exp:.0f} | {fps:.1f} |")


if __name__ == "__main__":
    main()
