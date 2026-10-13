"""PROTOTYPE (not wired into the server): W4A16 weight-only INT4 for the DiT text stream.

Idea: the text stream runs on only ~100-450 tokens per forward but owns half of
the DiT weights (7.5 GiB in FP8). Small-M GEMMs are weight-bandwidth bound, so
4-bit weights (group-wise, bf16 activations) halve that half: ~3.75 GiB saved,
with no PCIe traffic at all -- an alternative to block streaming.

GPU path: torch's built-in tinygemm kernel (`torch._weight_int4pack_mm`, sm_80+),
packed with `torch._convert_weight_to_int4pack`. CPU / reference path: explicit
dequantize + matmul, used for accuracy checks and to validate the GPU packing
convention at runtime (see `selftest_cuda`).

Open questions (why this is a prototype): image quality impact of RTN int4 on
the text stream is unmeasured; tinygemm speed at M~400 vs the FP8 cutlass GEMM
on sm_89 is unmeasured; packing API details changed across torch releases (2.9
is assumed). Run `python prototypes/int4_txt_stream/int4_linear.py` on the GPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def quantize_int4_groupwise(w: torch.Tensor, group_size: int = 128):
    """Asymmetric RTN int4 per (output row, K-group).

    w: (N, K) float. Returns q (N, K) uint8 in [0, 15], scale (N, K/g), zero (N, K/g)
    such that  w ~= (q - 8) * scale + zero   (tinygemm's min-based convention).
    """
    N, K = w.shape
    assert K % group_size == 0, (K, group_size)
    wg = w.float().reshape(N, K // group_size, group_size)
    wmin = wg.amin(-1)
    wmax = wg.amax(-1)
    scale = ((wmax - wmin) / 15.0).clamp_min(1e-8)
    q = torch.round((wg - wmin.unsqueeze(-1)) / scale.unsqueeze(-1)).clamp(0, 15)
    zero = wmin + 8.0 * scale  # so that (q - 8) * scale + zero == q * scale + wmin
    return q.reshape(N, K).to(torch.uint8), scale, zero


def dequantize_int4_groupwise(q, scale, zero, group_size: int = 128) -> torch.Tensor:
    N, K = q.shape
    qg = q.float().reshape(N, K // group_size, group_size)
    return ((qg - 8.0) * scale.unsqueeze(-1) + zero.unsqueeze(-1)).reshape(N, K)


class Int4WeightOnlyLinear(nn.Module):
    def __init__(self, lin: nn.Linear, group_size: int = 128, inner_k_tiles: int = 8):
        super().__init__()
        w = lin.weight.data
        self.in_features, self.out_features = w.shape[1], w.shape[0]
        self.group_size = group_size
        q, scale, zero = quantize_int4_groupwise(w, group_size)
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.data.to(torch.bfloat16), requires_grad=False)
        self.use_tinygemm = w.is_cuda
        if self.use_tinygemm:
            packed_u8 = (q[:, ::2] << 4 | q[:, 1::2]).contiguous()  # (N, K/2), high nibble first
            self.register_buffer("w_packed", torch._convert_weight_to_int4pack(packed_u8, inner_k_tiles))
            sz = torch.stack([scale, zero], dim=-1).transpose(0, 1).contiguous()  # (K/g, N, 2)
            self.register_buffer("scales_and_zeros", sz.to(torch.bfloat16))
        else:
            self.register_buffer("q", q)
            self.register_buffer("scale", scale)
            self.register_buffer("zero", zero)

    def weight_bytes(self) -> int:
        if self.use_tinygemm:
            return self.w_packed.numel() * self.w_packed.element_size() + \
                self.scales_and_zeros.numel() * self.scales_and_zeros.element_size()
        return self.q.numel() // 2 + (self.scale.numel() + self.zero.numel()) * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x2 = x.reshape(-1, shp[-1]).to(torch.bfloat16)
        if self.use_tinygemm:
            y = torch._weight_int4pack_mm(x2, self.w_packed, self.group_size, self.scales_and_zeros)
        else:
            w = dequantize_int4_groupwise(self.q, self.scale, self.zero, self.group_size)
            y = (x2.float() @ w.t()).to(torch.bfloat16)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shp[:-1], self.out_features)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def selftest(device: str = "cpu", K: int = 4096, N: int = 4096, M: int = 400) -> dict:
    torch.manual_seed(0)
    lin = nn.Linear(K, N, bias=True, device=device, dtype=torch.bfloat16)
    nn.init.normal_(lin.weight, std=0.02)
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    ref = lin(x)
    q4 = Int4WeightOnlyLinear(lin)
    out = q4(x)
    res = {"device": device, "rel_err_vs_bf16": _rel(out, ref),
           "weight_bytes_int4": q4.weight_bytes(), "weight_bytes_fp8": K * N}
    if device != "cpu":
        # validate the packing convention: tinygemm must equal explicit dequant
        q, s, z = quantize_int4_groupwise(lin.weight.data)
        w_dq = dequantize_int4_groupwise(q, s, z).to(torch.bfloat16)
        res["rel_err_tinygemm_vs_dequant"] = _rel(out, x @ w_dq.t() + lin.bias)
    return res


if __name__ == "__main__":
    print(selftest("cpu", K=1024, N=1024, M=64))
    if torch.cuda.is_available():
        print(selftest("cuda"))
