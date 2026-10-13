# Experiments: fitting the streaming editor on a 24 GB RTX 4090

Each subdirectory is one line of attack. The write-up comparing them is
[`notes/comparison-rtx4090.md`](../notes/comparison-rtx4090.md).

| dir | what | run |
|---|---|---|
| `a_block_streaming/` | cost model for streaming FP8 DiT blocks over PCIe (the implementation is `deploy/xvideo/blockswap.py`) | `python experiments/a_block_streaming/cost_model.py --tflops <probe> --h2d-gbps <probe>` |
| `c_vram_budget/` | VRAM budget per resolution / offload count, renders `visuals/rtx4090-vram-budget.svg` | `python experiments/c_vram_budget/estimate.py --svg visuals/rtx4090-vram-budget.svg` |
| `../prototypes/int4_txt_stream/` | (approach B) W4A16 INT4 text stream with torch's tinygemm, not wired into the server | `python prototypes/int4_txt_stream/int4_linear.py` |

All of these are models or prototypes. Ground truth comes from
`deploy/tools/probe_rtx4090.py` and the server's own log on the card.
