"""DiT block streaming: keep some FP8 DiT blocks in pinned host RAM (≤24GB cards).

The 16B DiT is ~15 GiB in FP8. On a 24GB card (RTX 4090) that plus the VAE,
KV pools, CUDA-graph buffers and activations does not leave enough headroom,
so this module parks the FP8 weights of N evenly spaced double-stream blocks in
pinned host memory and streams them back into a small ring of GPU "slots" right
before each of those blocks runs.

How it works
  * Each offloaded block's FP8 Linears (img/txt attn qkv+proj, mlp up+down) are
    packed into one flat pinned host buffer, img stream first, so a forward with
    ``skip_text_stream=True`` (KV store / ref-image prefill) copies only the img
    prefix.
  * ``JOYOMNI_DIT_OFFLOAD_SLOTS`` GPU slots (default 2) hold the weights of the
    offloaded blocks in flight. Offloaded block j always uses slot j % S, and its
    Fp8Linear ``weight_fp8`` buffers are permanent views into that slot, so the
    kernels see the same pointers every forward (CUDA-graph friendly).
  * Copies run on a dedicated copy stream: the first S copies start when the
    transformer forward starts; copy j (j >= S) starts as soon as block j-S
    (the previous user of the slot) has finished. Offloaded block j waits on its
    copy's event before running. All cross-stream ordering is event-based and
    every copy is joined back into the compute stream within the same forward,
    so the whole schedule is captured correctly by ``torch.cuda.graph``.

Cost per chunk (2 denoise steps + 1 img-only KV-store forward) is ~0.94 GiB of
H2D traffic per offloaded block. It overlaps with the compute of the resident
blocks between offloaded ones; PCIe 4.0 x16 moves ~20-25 GB/s pinned.

Env knobs:
  JOYOMNI_DIT_OFFLOAD_BLOCKS=<int>  number of DiT blocks to stream (default 0 = off)
  JOYOMNI_DIT_OFFLOAD_SLOTS=<int>   GPU slot ring size (default 2, min 1)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn

_STREAMS = ("img", "txt")
_LINEARS = ("attn_qkv", "attn_proj", "mlp_up", "mlp_down")


def offload_block_count() -> int:
    raw = os.environ.get("JOYOMNI_DIT_OFFLOAD_BLOCKS", "").strip()
    if not raw:
        return 0
    return max(0, int(raw))


def offload_slot_count() -> int:
    raw = os.environ.get("JOYOMNI_DIT_OFFLOAD_SLOTS", "").strip()
    return max(1, int(raw)) if raw else 2


def select_offload_indices(num_blocks: int, count: int) -> list[int]:
    """Spread `count` offloaded blocks evenly over [1, num_blocks) so every
    prefetch has resident blocks in front of it to hide behind. Block 0 is kept
    resident (its copy could only overlap with the input embedding)."""
    count = min(int(count), max(0, num_blocks - 1))
    if count <= 0:
        return []
    picked: list[int] = []
    for i in range(count):
        idx = 1 + int(round((i + 0.5) * (num_blocks - 1) / count - 0.5))
        idx = min(max(idx, 1), num_blocks - 1)
        while idx in picked and idx < num_blocks - 1:
            idx += 1
        if idx in picked:  # fall back to the first free index
            idx = next(j for j in range(1, num_blocks) if j not in picked)
        picked.append(idx)
    return sorted(picked)


@dataclass
class _Entry:
    module: nn.Module
    numel: int
    offset: int
    shape: tuple[int, int]


@dataclass
class _OffloadedBlock:
    index: int
    slot: int
    host: torch.Tensor
    img_numel: int
    total_numel: int
    entries: list[_Entry] = field(default_factory=list)


def _fp8_linears(block: nn.Module) -> list[tuple[str, nn.Module]]:
    found = []
    for stream in _STREAMS:
        for name in _LINEARS:
            mod = getattr(block, f"_fp8_{stream}_{name}", None)
            if mod is not None and isinstance(getattr(mod, "weight_fp8", None), torch.Tensor):
                found.append((stream, mod))
    return found


def _pinned_empty(numel: int, dtype: torch.dtype, pin: bool) -> torch.Tensor:
    """Page-locked host buffer of exactly `numel` elements.

    torch's pinned caching allocator rounds every request up to a power of two
    (a 384 MiB block would take 512 MiB), so allocate pageable memory and
    register it with the driver instead; fall back to pin_memory=True."""
    if not pin:
        return torch.empty(numel, dtype=dtype, device="cpu")
    buf = torch.empty(numel, dtype=dtype, device="cpu")
    try:
        err = torch.cuda.cudart().cudaHostRegister(buf.data_ptr(), buf.numel() * buf.element_size(), 0)
        if int(err) == 0 and buf.is_pinned():
            return buf
    except Exception:  # noqa: BLE001
        pass
    return torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)


def _column_major_flat(w: torch.Tensor) -> torch.Tensor:
    """Fp8Linear stores weight (K, N) column-major (stride (1, K)); its transpose
    (N, K) is contiguous. Return that contiguous storage flattened."""
    K, N = w.shape
    if w.stride() != (1, K):
        raise ValueError(f"expected column-major (K,N) weight with stride (1,{K}), got {w.stride()}")
    return w.t().contiguous().view(-1)


def _view_as_column_major(flat: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    K, N = shape
    return flat.view(N, K).t()


class DiTBlockStreamer:
    """Streams the FP8 weights of selected DiT blocks from pinned host RAM."""

    def __init__(self, transformer: nn.Module, indices: list[int], *,
                 num_slots: int = 2, pin_memory: bool = True):
        self.transformer = transformer
        blocks = transformer.double_blocks
        self.indices = sorted(int(i) for i in indices)
        self.num_slots = max(1, min(int(num_slots), len(self.indices) or 1))
        self.offloaded: list[_OffloadedBlock] = []
        self._order: dict[int, int] = {}  # block index -> position j in offloaded list
        self._skip_txt = False
        self._last_stream = None
        self._handles = []

        if not self.indices:
            raise ValueError("DiTBlockStreamer needs at least one block index")

        first = blocks[self.indices[0]]
        self.device = next(iter(m.weight_fp8 for _, m in _fp8_linears(first))).device
        self.dtype = next(iter(m.weight_fp8 for _, m in _fp8_linears(first))).dtype
        self.is_cuda = self.device.type == "cuda"

        slot_numel = 0
        packed: list[tuple[int, list[tuple[str, nn.Module]]]] = []
        for idx in self.indices:
            linears = _fp8_linears(blocks[idx])
            if not linears:
                raise RuntimeError(
                    f"block {idx} has no installed FP8 Linears; block streaming needs "
                    "JOYOMNI_FP8_IMG=1 JOYOMNI_FP8_TXT=1 (and JOYOMNI_LOW_VRAM=1)"
                )
            # img stream first so img-only forwards copy a prefix
            linears.sort(key=lambda sm: _STREAMS.index(sm[0]))
            packed.append((idx, linears))
            slot_numel = max(slot_numel, sum(m.weight_fp8.numel() for _, m in linears))

        self.slots = [
            torch.empty(slot_numel, dtype=self.dtype, device=self.device)
            for _ in range(self.num_slots)
        ]

        for j, (idx, linears) in enumerate(packed):
            slot = j % self.num_slots
            total = sum(m.weight_fp8.numel() for _, m in linears)
            host = _pinned_empty(total, self.dtype, pin_memory and self.is_cuda)
            entries: list[_Entry] = []
            offset = 0
            img_numel = 0
            for stream, mod in linears:
                w = mod.weight_fp8
                shape = (int(w.shape[0]), int(w.shape[1]))
                n = w.numel()
                host[offset:offset + n].copy_(_column_major_flat(w))
                entries.append(_Entry(module=mod, numel=n, offset=offset, shape=shape))
                if stream == "img":
                    img_numel = offset + n
                offset += n
            ob = _OffloadedBlock(index=idx, slot=slot, host=host, img_numel=img_numel,
                                 total_numel=total, entries=entries)
            # Re-point the Fp8Linears at their slot view; this drops the last
            # reference to the resident GPU weight.
            for e in entries:
                e.module.weight_fp8 = _view_as_column_major(
                    self.slots[slot][e.offset:e.offset + e.numel], e.shape
                )
            self.offloaded.append(ob)
            self._order[idx] = j

        if self.is_cuda:
            self.copy_stream = torch.cuda.Stream(device=self.device)
            n = len(self.offloaded)
            self._start_evt = torch.cuda.Event()
            self._copy_done = [torch.cuda.Event() for _ in range(n)]
            self._compute_done = [torch.cuda.Event() for _ in range(n)]
        else:
            self.copy_stream = None

        self._install_hooks()
        if self.is_cuda:
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ stats
    @property
    def host_bytes(self) -> int:
        return sum(ob.host.numel() * ob.host.element_size() for ob in self.offloaded)

    @property
    def slot_bytes(self) -> int:
        return sum(s.numel() * s.element_size() for s in self.slots)

    def describe(self) -> str:
        gib = 2 ** 30
        return (
            f"DiT block streaming: {len(self.offloaded)} blocks {self.indices} in pinned host RAM "
            f"({self.host_bytes / gib:.2f} GiB), {self.num_slots} GPU slots "
            f"({self.slot_bytes / gib:.2f} GiB) -> net VRAM saved "
            f"{(self.host_bytes - self.slot_bytes) / gib:.2f} GiB"
        )

    # ------------------------------------------------------------------ hooks
    def _install_hooks(self) -> None:
        tf = self.transformer
        self._handles.append(tf.register_forward_pre_hook(self._on_forward_start, with_kwargs=True))
        for idx in self.indices:
            block = tf.double_blocks[idx]
            self._handles.append(block.register_forward_pre_hook(self._make_pre_hook(idx)))
            self._handles.append(block.register_forward_hook(self._make_post_hook(idx)))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _copy_len(self, ob: _OffloadedBlock) -> int:
        return ob.img_numel if self._skip_txt else ob.total_numel

    def _issue_copy(self, j: int, wait_evt) -> None:
        ob = self.offloaded[j]
        n = self._copy_len(ob)
        dst = self.slots[ob.slot][:n]
        src = ob.host[:n]
        if not self.is_cuda:
            dst.copy_(src)
            return
        cs = self.copy_stream
        cs.wait_event(wait_evt)
        with torch.cuda.stream(cs):
            dst.copy_(src, non_blocking=True)
            self._copy_done[j].record(cs)

    def _on_forward_start(self, _module, args, kwargs):
        self._skip_txt = bool(kwargs.get("skip_text_stream", False))
        if self.is_cuda:
            # Forwards come from different threads/streams (session init prefill
            # on the caller's stream, chunks on the session's DiT stream). The
            # slots are shared, so a forward on a new stream must first wait for
            # everything queued on the previous one. Not during capture: the
            # graph runner synchronizes the device right before capturing.
            cur = torch.cuda.current_stream(self.device)
            if not torch.cuda.is_current_stream_capturing():
                last = self._last_stream
                if last is not None and last != cur:
                    cur.wait_stream(last)
                self._last_stream = cur
            # Fork the copy stream off the compute stream: orders the first S
            # copies after everything already queued (incl. the previous
            # forward's use of the slots) and pulls the copy stream into an
            # active graph capture.
            self._start_evt.record(torch.cuda.current_stream(self.device))
        for j in range(min(self.num_slots, len(self.offloaded))):
            self._issue_copy(j, self._start_evt if self.is_cuda else None)
        return None

    def _make_pre_hook(self, idx: int):
        j = self._order[idx]

        def hook(_module, _args):
            if self.is_cuda:
                torch.cuda.current_stream(self.device).wait_event(self._copy_done[j])
            return None

        return hook

    def _make_post_hook(self, idx: int):
        j = self._order[idx]
        nxt = j + self.num_slots

        def hook(_module, _args, _output):
            if nxt >= len(self.offloaded):
                return None
            if self.is_cuda:
                evt = self._compute_done[j]
                evt.record(torch.cuda.current_stream(self.device))
                self._issue_copy(nxt, evt)
            else:
                self._issue_copy(nxt, None)
            return None

        return hook


def maybe_enable_block_streaming(transformer: nn.Module) -> DiTBlockStreamer | None:
    """Enable streaming per JOYOMNI_DIT_OFFLOAD_BLOCKS. Call after the FP8
    Linears are installed (low-VRAM load does this block by block)."""
    count = offload_block_count()
    if count <= 0:
        return None
    num_blocks = len(transformer.double_blocks)
    indices = select_offload_indices(num_blocks, count)
    streamer = DiTBlockStreamer(transformer, indices, num_slots=offload_slot_count())
    transformer._joyomni_block_streamer = streamer
    print(f"#####[BLOCK-STREAM] {streamer.describe()}", flush=True)
    return streamer
