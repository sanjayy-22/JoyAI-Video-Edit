"""CPU parity tests for xvideo.blockswap (no GPU / joyomni_ops needed).

Run from deploy/:  PYTHONPATH=. python -m pytest tests/test_blockswap.py -q
           or:     PYTHONPATH=. python tests/test_blockswap.py

Uses stand-in blocks whose FP8 Linears mirror Fp8Linear's storage contract
(weight_fp8 is (K, N) column-major float8_e4m3fn). On CPU the streamer copies
synchronously, which exercises the packing / slot-view / prefix-copy logic; the
CUDA stream + event schedule is exercised by tools/probe_rtx4090.py on a GPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from xvideo.blockswap import DiTBlockStreamer, select_offload_indices

FP8 = torch.float8_e4m3fn


class FakeFp8Linear(nn.Module):
    def __init__(self, k: int, n: int, gen: torch.Generator):
        super().__init__()
        w = (torch.randn(n, k, generator=gen) * 0.5).to(FP8)  # (N, K) row-major
        self.register_buffer("weight_fp8", w.t(), persistent=False)  # (K, N) col-major

    def forward(self, x):
        return x @ self.weight_fp8.float()


class FakeBlock(nn.Module):
    def __init__(self, d: int, gen: torch.Generator):
        super().__init__()
        for stream in ("img", "txt"):
            setattr(self, f"_fp8_{stream}_attn_qkv", FakeFp8Linear(d, 3 * d, gen))
            setattr(self, f"_fp8_{stream}_attn_proj", FakeFp8Linear(d, d, gen))
            setattr(self, f"_fp8_{stream}_mlp_up", FakeFp8Linear(d, 4 * d, gen))
            setattr(self, f"_fp8_{stream}_mlp_down", FakeFp8Linear(4 * d, d, gen))

    def _stream(self, s, x):
        q = getattr(self, f"_fp8_{s}_attn_qkv")(x)[..., : x.shape[-1]]
        x = x + 0.01 * getattr(self, f"_fp8_{s}_attn_proj")(torch.tanh(q))
        h = getattr(self, f"_fp8_{s}_mlp_up")(x)
        return x + 0.01 * getattr(self, f"_fp8_{s}_mlp_down")(torch.tanh(h))

    def forward(self, img, txt, skip_text_stream=False):
        img = self._stream("img", img)
        if not skip_text_stream:
            txt = self._stream("txt", txt)
        return img, txt


class FakeTransformer(nn.Module):
    def __init__(self, depth: int = 10, d: int = 32, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.double_blocks = nn.ModuleList(FakeBlock(d, gen) for _ in range(depth))

    def forward(self, hidden_states, txt, skip_text_stream=False):
        img = hidden_states
        for b in self.double_blocks:
            img, txt = b(img, txt, skip_text_stream=skip_text_stream)
        return img, txt


def _inputs(d=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, 5, d, generator=g), torch.randn(2, 3, d, generator=g)


def _run(tf, img, txt, skip):
    with torch.no_grad():
        return tf(hidden_states=img, txt=txt, skip_text_stream=skip)


def test_select_offload_indices():
    for n in (1, 4, 8, 12, 20, 39, 60):
        idx = select_offload_indices(40, n)
        assert len(idx) == min(n, 39)
        assert len(set(idx)) == len(idx)
        assert all(1 <= i < 40 for i in idx)
        assert idx == sorted(idx)
    assert select_offload_indices(40, 0) == []
    # evenly spread: 8 of 40 -> gaps of ~5
    gaps = [b - a for a, b in zip(select_offload_indices(40, 8), select_offload_indices(40, 8)[1:])]
    assert max(gaps) - min(gaps) <= 1, gaps


def _check(num_slots, indices, depth=10):
    tf = FakeTransformer(depth=depth)
    img, txt = _inputs()
    ref_full = _run(tf, img, txt, False)
    ref_skip = _run(tf, img, txt, True)

    streamer = DiTBlockStreamer(tf, indices, num_slots=num_slots, pin_memory=False)
    # weights now live in shared slots with the original column-major layout
    for idx in indices:
        w = tf.double_blocks[idx]._fp8_img_mlp_up.weight_fp8
        assert w.stride() == (1, w.shape[0])
        assert any(w.untyped_storage().data_ptr() == s.untyped_storage().data_ptr() for s in streamer.slots)

    # interleave img-only and full forwards: stale txt halves in a slot must never leak
    for skip in (True, False, True, True, False, False):
        out = _run(tf, img, txt, skip)
        ref = ref_skip if skip else ref_full
        torch.testing.assert_close(out[0], ref[0], rtol=0, atol=0)
        torch.testing.assert_close(out[1], ref[1], rtol=0, atol=0)

    expect_saved = streamer.host_bytes - streamer.slot_bytes
    assert expect_saved == (len(indices) - streamer.num_slots) * streamer.slots[0].numel()
    streamer.remove()


def test_parity_two_slots():
    _check(2, select_offload_indices(10, 4))


def test_parity_one_slot():
    _check(1, select_offload_indices(10, 3))


def test_parity_slots_ge_blocks():
    _check(4, [2, 7])


def test_parity_adjacent_blocks():
    _check(2, [1, 2, 3, 4, 5, 6, 7, 8, 9])


if __name__ == "__main__":
    test_select_offload_indices()
    test_parity_two_slots()
    test_parity_one_slot()
    test_parity_slots_ge_blocks()
    test_parity_adjacent_blocks()
    print("blockswap CPU parity: OK")
