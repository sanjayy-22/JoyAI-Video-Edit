#!/usr/bin/env python
"""Pre-flight probe for running the streaming server on an RTX 4090 (sm_89, 24 GB).

Run from the repo root inside the deploy env (no model weights needed):

    PYTHONPATH=deploy python deploy/tools/probe_rtx4090.py            # all checks
    PYTHONPATH=deploy python deploy/tools/probe_rtx4090.py --quick    # skip timings

Each check prints PASS / WARN / FAIL / SKIP. It verifies the pieces the 4090
profile (deploy/run_server_4090.sh) depends on:
  1. GPU / driver / free VRAM / host RAM
  2. joyomni_ops built for sm_89 incl. the cutlass FP8 GEMM (accuracy + speed)
  3. JOYOMNI_FP8_FAST_ACCUM Triton GEMM compiles on sm_89 (accuracy + speed vs cutlass)
  4. SageAttention sm_89 kernels (accuracy + speed vs SDPA)
  5. pinned H2D bandwidth (drives the DiT block-streaming cost)
  6. DiT block streaming (xvideo/blockswap.py) is exact in eager mode AND inside
     a captured CUDA graph, and how much its copies cost
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # deploy/

import torch  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []
GIB = 2 ** 30


def report(name: str, status: str, detail: str = "") -> None:
    RESULTS.append((name, status, detail))
    print(f"[{status:4s}] {name}: {detail}", flush=True)


def section(fn):
    def wrapped(*a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as exc:  # noqa: BLE001
            report(fn.__name__, "FAIL", f"{exc!r}")
            traceback.print_exc()
            return None
    return wrapped


def _time_cuda(fn, iters=20, warmup=3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def _host_ram_gib() -> tuple[float, float]:
    total = avail = 0.0
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            if k == "MemTotal":
                total = int(v.split()[0]) / 2 ** 20
            elif k == "MemAvailable":
                avail = int(v.split()[0]) / 2 ** 20
    except OSError:
        pass
    return total, avail


@section
def check_gpu(dev: torch.device) -> None:
    if not torch.cuda.is_available():
        report("gpu", "FAIL", "torch.cuda.is_available() is False")
        sys.exit(1)
    p = torch.cuda.get_device_properties(dev)
    cc = (p.major, p.minor)
    free_b, total_b = torch.cuda.mem_get_info(dev)
    report("gpu", "PASS" if cc == (8, 9) else "WARN",
           f"{p.name} sm_{cc[0]}{cc[1]} {p.multi_processor_count} SMs, "
           f"torch {torch.__version__} (CUDA {torch.version.cuda})"
           + ("" if cc == (8, 9) else " -- this probe targets sm_89 (RTX 4090)"))
    used_by_others = (total_b - free_b) / GIB
    status = "PASS" if used_by_others < 1.0 else "WARN"
    report("vram", status,
           f"{total_b / GIB:.2f} GiB total, {free_b / GIB:.2f} GiB free before this probe "
           f"({used_by_others:.2f} GiB used by context/other processes"
           + ("; a desktop on this GPU eats into the 24 GB budget" if status == "WARN" else "") + ")")
    total, avail = _host_ram_gib()
    status = "PASS" if total >= 60 else ("WARN" if total >= 46 else "FAIL")
    report("host_ram", status,
           f"{total:.0f} GiB total, {avail:.0f} GiB available. Need roughly: text encoder 16 "
           f"(pinned when JOYOMNI_TE_PIN=1) + streamed DiT blocks 0.375/block + page cache "
           f"for the 32 GB checkpoint during load. 64 GB recommended; with 32-48 GB set JOYOMNI_TE_PIN=0.")


@section
def check_joyomni_ops(dev: torch.device, quick: bool) -> None:
    import joyomni_ops as jo
    if not jo.has_fp8():
        report("joyomni_ops_fp8", "FAIL", "built without FP8 GEMM (JOYOMNI_OPS_NO_FP8). "
               "Rebuild with cutlass, or run with JOYOMNI_FP8_IMG=0 JOYOMNI_FP8_TXT=0 (bf16 DiT will NOT fit 24 GB)")
        return
    torch.manual_seed(0)
    fp8 = torch.float8_e4m3fn
    worst = 0.0
    for (M, K, N) in [(1400, 4096, 12288), (1400, 16384, 4096), (400, 4096, 16384)]:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        xq = torch.empty(M, K, device=dev, dtype=fp8)
        xs = torch.empty(M, 1, device=dev, dtype=torch.float32)
        jo.sgl_per_token_quant_fp8(x, xq, xs)
        w_kn = w.t().contiguous()
        ws = w_kn.abs().amax(0).float().clamp_min(1e-8) / 448.0
        wq = (w_kn.float() / ws).clamp(-448, 448).to(fp8).t().contiguous().t()
        y = jo.fp8_scaled_mm(xq, wq, xs, ws, torch.bfloat16, None)
        ref = x.float() @ w.float().t()
        rel = ((y.float() - ref).norm() / ref.norm()).item()
        worst = max(worst, rel)
        if not quick and (M, K, N) == (1400, 4096, 12288):
            ms = _time_cuda(lambda: jo.fp8_scaled_mm(xq, wq, xs, ws, torch.bfloat16, None))
            report("cutlass_fp8_gemm_speed", "PASS",
                   f"M{M} K{K} N{N}: {ms:.3f} ms = {2 * M * K * N / ms / 1e9:.0f} TFLOPS")
    report("joyomni_ops_fp8", "PASS" if worst < 0.06 else "FAIL",
           f"sm_89 cutlass fp8_scaled_mm rel err {worst:.3%} (FP8 quantization alone is ~3-4%)")
    q = torch.randn(1, 64, 4, 128, device=dev, dtype=torch.bfloat16)
    out = jo.rmsnorm(q.reshape(-1, 128), torch.ones(128, device=dev, dtype=torch.bfloat16), 1e-6)
    report("joyomni_ops_fused", "PASS" if torch.isfinite(out).all() else "FAIL", "rmsnorm kernel runs on sm_89")


@section
def check_fast_accum(dev: torch.device, quick: bool) -> None:
    os.environ["JOYOMNI_FP8_FAST_ACCUM"] = "1"
    import xvideo.models.dit.fp8_linear as fl
    fl = importlib.reload(fl)
    if not fl._FAST_ACCUM:
        report("fast_accum", "FAIL", "module did not pick up JOYOMNI_FP8_FAST_ACCUM=1")
        return
    torch.manual_seed(0)
    worst = 0.0
    for (M, K, N) in [(1400, 4096, 12288), (1400, 16384, 4096), (400, 4096, 16384), (700, 4096, 4096)]:
        lin = torch.nn.Linear(K, N, bias=True, device=dev, dtype=torch.bfloat16)
        torch.nn.init.normal_(lin.weight, std=0.02)
        f = fl.Fp8Linear.from_linear(lin)
        x = torch.nn.functional.gelu(torch.randn(M, K, device=dev, dtype=torch.bfloat16))  # all-positive-ish
        y = f(x)
        ref = lin(x).float()
        rel = ((y.float() - ref).norm() / ref.norm()).item()
        if not torch.isfinite(y).all():
            rel = float("inf")
        worst = max(worst, rel)
        if not quick and (M, K, N) == (1400, 4096, 12288):
            ms = _time_cuda(lambda: f(x))
            report("fast_accum_speed", "PASS",
                   f"quant+GEMM M{M} K{K} N{N}: {ms:.3f} ms = {2 * M * K * N / ms / 1e9:.0f} TFLOPS "
                   "(compare with cutlass_fp8_gemm_speed; keep JOYOMNI_FP8_FAST_ACCUM=1 only if faster)")
    report("fast_accum", "PASS" if worst < 0.08 else "FAIL",
           f"Triton fp16-accumulate FP8 GEMM compiles on sm_89, rel err {worst:.3%}")
    os.environ.pop("JOYOMNI_FP8_FAST_ACCUM", None)
    importlib.reload(fl)


@section
def check_sage(dev: torch.device, quick: bool) -> None:
    try:
        from sageattention import sageattn
    except Exception as exc:  # noqa: BLE001
        report("sageattention", "WARN", f"not importable ({exc!r}); run with JOYOMNI_SAGE_ATTN=0 (SDPA/flash, slower)")
        return
    torch.manual_seed(0)
    # NHD layout as the DiT uses it: current 700 + ref 700 + txt ~400 queries, ~3000 keys
    q = torch.randn(1, 1800, 32, 128, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, 3200, 32, 128, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, 3200, 32, 128, device=dev, dtype=torch.bfloat16)
    out = sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
    report("sageattention", "PASS" if cos > 0.99 else "FAIL", f"sm_89 kernels run, cosine vs SDPA {cos:.5f}")
    # the server captures attention in a CUDA graph; the bundled patch is what makes that work
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        gout = sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
    g.replay()
    torch.cuda.synchronize()
    same = torch.allclose(gout.float(), out.float(), atol=2e-2, rtol=0)
    report("sageattention_graph", "PASS" if same else "FAIL",
           "graph replay matches eager" if same else
           "graph replay differs -> SageAttention was built WITHOUT sageattention-cudagraph-stream.patch")
    if not quick:
        ms_s = _time_cuda(lambda: sageattn(q, k, v, tensor_layout="NHD", is_causal=False))
        ms_d = _time_cuda(lambda: torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)))
        report("attention_speed", "PASS", f"sage {ms_s:.3f} ms vs SDPA {ms_d:.3f} ms per call")


@section
def check_h2d(dev: torch.device, quick: bool) -> float:
    from xvideo.blockswap import _pinned_empty
    n = 384 * 2 ** 20  # one DiT block (img+txt FP8 weights) = 384 MiB
    host = _pinned_empty(n, torch.uint8, True)
    host.fill_(1)
    dst = torch.empty(n, dtype=torch.uint8, device=dev)
    ms = _time_cuda(lambda: dst.copy_(host, non_blocking=True), iters=5 if quick else 15, warmup=2)
    gbps = n / ms / 1e6
    status = "PASS" if gbps >= 18 else "WARN"
    report("h2d_bandwidth", status,
           f"{gbps:.1f} GB/s pinned H2D ({ms:.1f} ms per 384 MiB DiT block; pinned={host.is_pinned()})"
           + ("" if status == "PASS" else " -- check the slot is PCIe 4.0 x16 (nvidia-smi -q | grep -A3 'Link')"))
    return gbps


@section
def check_block_streaming(dev: torch.device, quick: bool) -> None:
    import torch.nn as nn
    from xvideo.blockswap import DiTBlockStreamer, select_offload_indices
    sys.path.insert(0, str(HERE.parent / "tests"))
    from test_blockswap import FakeTransformer

    class GpuLinear(nn.Module):
        """Stand-in with Fp8Linear's weight contract, computing in bf16."""
        def __init__(self, src):
            super().__init__()
            self.register_buffer("weight_fp8", src.weight_fp8.to(dev), persistent=False)

        def forward(self, x):
            return x @ self.weight_fp8.to(torch.bfloat16)

    d, depth = (512, 12) if quick else (2048, 16)
    tf = FakeTransformer(depth=depth, d=d)
    for b in tf.double_blocks:
        for name, child in list(b.named_children()):
            setattr(b, name, GpuLinear(child))
    tf = tf.to(dev)
    gen = torch.Generator(device=dev).manual_seed(1)
    img = torch.randn(1, 1400, d, device=dev, dtype=torch.bfloat16, generator=gen)
    txt = torch.randn(1, 400, d, device=dev, dtype=torch.bfloat16, generator=gen)

    @torch.no_grad()
    def run(skip):
        return tf(hidden_states=img, txt=txt, skip_text_stream=skip)

    ref_full, ref_skip = run(False), run(True)
    t_res = _time_cuda(lambda: run(False), iters=5) if not quick else 0.0

    streamer = DiTBlockStreamer(tf, select_offload_indices(depth, depth // 4), num_slots=2)
    ok_eager = all(
        torch.equal(a, b) for a, b in zip(run(False), ref_full)
    ) and all(torch.equal(a, b) for a, b in zip(run(True), ref_skip))
    report("blockswap_eager", "PASS" if ok_eager else "FAIL",
           f"{streamer.describe()} -> outputs {'bit-identical' if ok_eager else 'DIFFER'}")

    # graph: img-only forward then full forward in one capture, like the server's chunk graph
    static = {}

    def body():
        static["skip"] = run(True)
        static["full"] = run(False)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        body()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    # scribble over the slots so a graph that did NOT capture the copies fails loudly
    for slot in streamer.slots:
        slot.zero_()
    g.replay()
    torch.cuda.synchronize()
    # allclose, not equal: cuBLAS may pick another algorithm under capture. A
    # missing/misordered copy reads zeroed or foreign weights and is far off.
    def close(xs, ys):
        return all(torch.allclose(a.float(), b.float(), rtol=2e-2, atol=2e-2) for a, b in zip(xs, ys))

    ok_graph = close(static["full"], ref_full) and close(static["skip"], ref_skip)
    report("blockswap_cuda_graph", "PASS" if ok_graph else "FAIL",
           "copies are captured and ordered inside torch.cuda.graph" if ok_graph else
           "graph replay output differs from resident reference")
    if not quick:
        t_str = _time_cuda(lambda: run(False), iters=5)
        t_graph = _time_cuda(g.replay, iters=5)
        report("blockswap_overhead", "PASS",
               f"toy model: resident {t_res:.1f} ms, streamed eager {t_str:.1f} ms, "
               f"streamed graph (skip+full) {t_graph:.1f} ms -- the toy has far less compute per "
               "block than the real 16B DiT, so real overlap is better than this")
    streamer.remove()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--quick", action="store_true", help="skip timings")
    args = ap.parse_args()
    dev = torch.device(args.device)
    if not torch.cuda.is_available():
        report("gpu", "FAIL", "torch.cuda.is_available() is False (no driver / CPU-only torch)")
        sys.exit(1)
    torch.cuda.set_device(dev)

    check_gpu(dev)
    check_joyomni_ops(dev, args.quick)
    check_fast_accum(dev, args.quick)
    check_sage(dev, args.quick)
    check_h2d(dev, args.quick)
    check_block_streaming(dev, args.quick)

    fails = [r for r in RESULTS if r[1] == "FAIL"]
    warns = [r for r in RESULTS if r[1] == "WARN"]
    print("\n==== summary ====")
    print(f"{len(RESULTS) - len(fails) - len(warns)} pass, {len(warns)} warn, {len(fails)} fail")
    for name, _, detail in fails:
        print(f"  FAIL {name}: {detail}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
