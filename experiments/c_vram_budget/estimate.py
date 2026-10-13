#!/usr/bin/env python
"""VRAM budget estimator for the streaming server on 24 GB cards (RTX 4090).

Pure arithmetic, no GPU or weights needed:

    python experiments/c_vram_budget/estimate.py            # markdown table
    python experiments/c_vram_budget/estimate.py --svg visuals/rtx4090-vram-budget.svg

What is computed vs. assumed
  * DiT weight bytes: exact, from the architecture in deploy/xvideo/config.py +
    deploy/xvideo/models/dit/dit.py (40 double-stream blocks, hidden 4096, MLP x4).
  * KV / CUDA-graph buffers: exact formulas from deploy/xvideo/serving/graph_runner.py
    (pool = 2 chunks + ref, stage = 1 chunk, pub = 2 chunks) plus the two eager
    python-cache chunks (sink + previous) and the ref-image KV when RV2V is on.
  * "Everything else" (VAE weights + compiled-VAE workspaces, CUDA context,
    activation / graph pool, cuBLAS workspaces, attention scratch) is NOT derivable
    from the repo (the VAE config lives on the Hub). It is calibrated from the one
    published data point -- RTX 5090, 480p24, low-VRAM: ~21.5 GiB resident
    (DEPLOYMENT.md) -- and scaled with token count. Treat it as +/- 1.5 GiB.
  * The published ~28 GiB *peak* (under a 30 GiB allocator cap) is a caching-
    allocator high-water mark; how much of it is real demand vs. lazily-held cache
    is unknown, which is exactly why the 4090 presets leave >=4 GiB of headroom.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

GIB = 2 ** 30
HIDDEN, HEADS, DEPTH, MLP = 4096, 32, 40, 4
LAT_C = 64
STEM_GROUP, STEM_STRIDE, VAE_SPATIAL = 2, 3, 16   # latent_h = H * 2 / 3 / 16


def dit_fp8_block_bytes() -> int:
    per_stream = HIDDEN * 3 * HIDDEN + HIDDEN * HIDDEN + 2 * HIDDEN * MLP * HIDDEN
    return 2 * per_stream  # img + txt, 1 byte/param


def dit_other_bytes() -> int:
    """bf16 params that stay resident (biases, modulation tables, norms, embedders)
    + fp32 per-channel FP8 scales."""
    per_block = 2 * (
        6 * HIDDEN                                   # ModulateWan table
        + 3 * HIDDEN + HIDDEN + MLP * HIDDEN + HIDDEN  # Linear biases
        + 2 * (HIDDEN // HEADS)                      # q/k RMSNorm weights
    )
    scales = 2 * (3 * HIDDEN + HIDDEN + MLP * HIDDEN + HIDDEN) * 4  # fp32 bytes
    glob = (
        LAT_C * HIDDEN + HIDDEN                      # img_in (patch 1x1x1)
        + 256 * HIDDEN + HIDDEN + HIDDEN * HIDDEN + HIDDEN   # time_embedder
        + HIDDEN * 6 * HIDDEN + 6 * HIDDEN           # time_proj
        + 2 * (HIDDEN * HIDDEN + HIDDEN)             # text_embedder (PixArt proj)
        + HIDDEN * LAT_C + LAT_C                     # proj_out
    )
    return (DEPTH * per_block + glob) * 2 + DEPTH * scales


def latent_tokens(w: int, h: int) -> int:
    lh = h * STEM_GROUP // STEM_STRIDE // VAE_SPATIAL
    lw = w * STEM_GROUP // STEM_STRIDE // VAE_SPATIAL
    return lh * lw


def kv_bytes_per_token() -> int:
    return DEPTH * HIDDEN * 2 * 2  # layers x (heads*head_dim) x bf16 x (K and V)


def kv_and_graph_bytes(ct: int, ref_tokens: int) -> int:
    graph = (2 * ct + ref_tokens) + ct + 2 * ct      # pool + stage + pub (double-buffered)
    eager = 2 * ct + ref_tokens                      # sink + prev chunk, ref-image KV in python cache
    return (graph + eager) * kv_bytes_per_token()


REF_TOKENS_768 = (768 * STEM_GROUP // STEM_STRIDE // VAE_SPATIAL) ** 2  # square 768 ref image -> 32x32

# calibration: RTX 5090 480p24 low-VRAM, no offload, no ref image
_CAL_RESIDENT = 21.5 * GIB
_CAL_CT = latent_tokens(840, 480)


def _other_calibrated(ct: int) -> float:
    base = _CAL_RESIDENT - DEPTH * dit_fp8_block_bytes() - dit_other_bytes() - kv_and_graph_bytes(_CAL_CT, 0)
    # assume half of "other" scales with tokens (activations, VAE feature caches),
    # half is fixed (VAE weights, CUDA context, workspaces)
    return base * (0.5 + 0.5 * ct / _CAL_CT)


@dataclass
class Config:
    name: str
    w: int
    h: int
    offload: int
    slots: int = 2
    ref: bool = False

    def rows(self) -> dict[str, float]:
        ct = latent_tokens(self.w, self.h)
        blk = dit_fp8_block_bytes()
        resident_blocks = DEPTH - self.offload + (min(self.slots, self.offload) if self.offload else 0)
        return {
            "DiT weights (GPU)": (resident_blocks * blk + dit_other_bytes()) / GIB,
            "KV + graph buffers": kv_and_graph_bytes(ct, REF_TOKENS_768 if self.ref else 0) / GIB,
            "VAE, context, activations (calibrated)": _other_calibrated(ct) / GIB,
        }

    def host_offload_gib(self) -> float:
        return self.offload * dit_fp8_block_bytes() / GIB


CONFIGS = [
    Config("RTX 5090 reference 840x480, no offload", 840, 480, 0),
    Config("4090 balanced 840x480, 8 blocks streamed", 840, 480, 8),
    Config("4090 balanced + RV2V reference image", 840, 480, 8, ref=True),
    Config("4090 fast 672x384, 4 blocks streamed", 672, 384, 4),
    Config("4090 safe 672x384, 10 blocks streamed", 672, 384, 10),
]

CARD_GIB = 23.5  # 24 GB card as seen by mem_get_info, before any desktop usage


def table() -> str:
    lines = [
        "| config | tokens/latent frame | DiT on GPU | KV + graph | other (cal.) | **est. resident** | headroom on 23.5 GiB | pinned host for DiT |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in CONFIGS:
        r = c.rows()
        tot = sum(r.values())
        lines.append(
            f"| {c.name} | {latent_tokens(c.w, c.h)} | {r['DiT weights (GPU)']:.2f} | "
            f"{r['KV + graph buffers']:.2f} | {r['VAE, context, activations (calibrated)']:.2f} | "
            f"**{tot:.2f}** | {CARD_GIB - tot:+.2f} | {c.host_offload_gib():.2f} |"
        )
    return "\n".join(lines)


def svg() -> str:
    """Horizontal stacked bars, one per config, with the 24 GB card line."""
    rows = [(c.name, c.rows()) for c in CONFIGS]
    keys = list(rows[0][1].keys())
    W, left, right, top, bar_h, gap = 880, 300, 40, 92, 30, 18
    H = top + len(rows) * (bar_h + gap) + 70
    xmax = 26.0
    sx = (W - left - right) / xmax

    def x(v):
        return left + v * sx

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
        f'aria-labelledby="t d" font-family="system-ui, -apple-system, Segoe UI, sans-serif">',
        '<title id="t">Estimated resident VRAM per configuration</title>',
        '<desc id="d">Stacked bars of DiT weights, KV and graph buffers, and calibrated other '
        'memory for each configuration, against the 23.5 GiB usable on a 24 GB RTX 4090.</desc>',
        "<style>",
        ".viz-root{--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--grid:#e4e3df;"
        "--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--limit:#0b0b0b}",
        "@media (prefers-color-scheme: dark){.viz-root{--surface-1:#1a1a19;--text-primary:#f0efec;"
        "--text-secondary:#c3c2b7;--grid:#3a3936;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--limit:#f0efec}}",
        "text{fill:var(--text-primary);font-size:12px}.muted{fill:var(--text-secondary)}",
        "</style>",
        f'<g class="viz-root"><rect width="{W}" height="{H}" fill="var(--surface-1)"/>',
        '<text x="20" y="26" font-size="15" font-weight="600">Estimated resident VRAM (GiB)</text>',
        '<text x="20" y="46" class="muted">DiT weights and KV/graph buffers are computed from the code; '
        '"VAE, context, activations" is calibrated</text>',
        '<text x="20" y="62" class="muted">from the published RTX 5090 resident number (+/- 1.5 GiB). '
        'Transient peaks come on top.</text>',
    ]
    for g in range(0, 27, 4):
        out.append(f'<line x1="{x(g):.1f}" x2="{x(g):.1f}" y1="{top - 6}" y2="{H - 50}" stroke="var(--grid)" stroke-width="1"/>')
        out.append(f'<text x="{x(g):.1f}" y="{H - 34}" text-anchor="middle" class="muted">{g}</text>')
    for i, (name, r) in enumerate(rows):
        y = top + i * (bar_h + gap)
        out.append(f'<text x="{left - 10}" y="{y + bar_h / 2 + 4:.1f}" text-anchor="end">{name}</text>')
        acc = 0.0
        for k_i, k in enumerate(keys):
            v = r[k]
            x0 = x(acc) + (2 if k_i else 0)
            wseg = max(0.0, v * sx - (2 if k_i else 0))
            rx = 4 if k_i == len(keys) - 1 else 0
            out.append(
                f'<rect x="{x0:.1f}" y="{y}" width="{wseg:.1f}" height="{bar_h}" rx="{rx}" '
                f'fill="var(--series-{k_i + 1})"><title>{name}\n{k}: {v:.2f} GiB</title></rect>'
            )
            acc += v
        out.append(f'<text x="{x(acc) + 6:.1f}" y="{y + bar_h / 2 + 4:.1f}">{acc:.2f}</text>')
    out.append(f'<line x1="{x(CARD_GIB):.1f}" x2="{x(CARD_GIB):.1f}" y1="{top - 10}" y2="{H - 50}" '
               f'stroke="var(--limit)" stroke-width="2" stroke-dasharray="5 4"/>')
    out.append(f'<text x="{x(CARD_GIB) - 4:.1f}" y="{top - 14}" text-anchor="middle" class="muted">4090 usable ~23.5</text>')
    lx = left
    for k_i, k in enumerate(keys):
        out.append(f'<rect x="{lx}" y="{H - 20}" width="12" height="12" rx="2" fill="var(--series-{k_i + 1})"/>')
        out.append(f'<text x="{lx + 18}" y="{H - 10}">{k}</text>')
        lx += 18 + 7.2 * len(k) + 24
    out.append("</g></svg>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--svg", default=None, help="write the chart to this path")
    args = ap.parse_args()
    print(f"FP8 DiT block: {dit_fp8_block_bytes() / GIB:.3f} GiB, all 40 blocks: "
          f"{DEPTH * dit_fp8_block_bytes() / GIB:.2f} GiB, other resident DiT params: "
          f"{dit_other_bytes() / GIB:.2f} GiB, KV per token: {kv_bytes_per_token() / 2**10:.0f} KiB\n")
    print(table())
    if args.svg:
        with open(args.svg, "w") as f:
            f.write(svg())
        print(f"\nwrote {args.svg}")


if __name__ == "__main__":
    main()
