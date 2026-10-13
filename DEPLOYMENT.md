# Deployment Guide

This guide takes JoyAI-Video-Edit streaming v2v editing from a fresh clone to a
fully working server: environment, weights, configuration, and launch.

JoyAI-Video-Edit is a real-time, streaming **video-to-video editing** service. A
client streams source frames over a WebSocket; the server runs a streaming DiT +
xVAE pipeline (with face/person presence gating) and streams edited frames back.

- **Serving stack:** FastAPI + WebSocket, served by `uvicorn`.
- **Entry point:** [`xvideo/serving/serve_joyomni_streaming.py`](deploy/xvideo/serving/serve_joyomni_streaming.py)
- **Launcher:** [`run_server.sh`](deploy/run_server.sh) (binds `0.0.0.0:8080` by default).
- **Web UI:** [`static/index.html`](deploy/static/index.html), served at `GET /`.

---

## 1. Layout

```
deploy/
├── run_server.sh          # launcher (env-driven)
├── run_server_4090.sh     # RTX 4090 (24 GB) defaults on top of run_server.sh (§5)
├── tools/probe_rtx4090.py # pre-flight GPU probe for the 4090 profile (§5.3)
├── tests/                 # CPU unit tests (e.g. DiT block streaming parity)
├── requirements.txt       # pinned Python deps (SageAttention / flash-attn-4 / joyomni_ops built separately, §2)
├── sageattention-cudagraph-stream.patch  # stream fix for SageAttention (see §2)
├── joyomni_ops/           # in-tree CUDA op library (FP8 GEMM + fused kernels); pip install
├── xvideo/                # service code
│   ├── config.py          # runtime/model config defaults
│   ├── utils.py           # resize buckets, seeding helpers
│   ├── inductor_autotune_fix.py  # torch 2.9+ compile-cache fix
│   ├── lowvram.py         # low-VRAM mode switches (JOYOMNI_LOW_VRAM=1)
│   ├── blockswap.py       # stream FP8 DiT blocks from pinned host RAM (JOYOMNI_DIT_OFFLOAD_BLOCKS)
│   ├── models/            # dit/, vae/, pipeline, flow-match scheduler, loaders
│   └── serving/           # FastAPI app, streaming runtime, CUDA-graph runner, prompt-enhancement
├── static/index.html      # browser client
├── rv2v_reference/        # reference images for the UI
├── recordings/            # session recordings (created at runtime; git-ignored)
└── deps/                  # weights + compile cache — NOT in git
    ├── checkpoints/       # DiT / xVAE / MiMo-VL / onnx detectors (~51G)
    └── cache*/            # torchinductor / triton / nv_compute (one root per GPU model, §4)
```

> **`deploy/deps/` is git-ignored.** It must exist on disk for the server to
> start, but it is not tracked by this repo — you populate it in §3.

---

## 2. Prepare the environment

> All commands in this guide run from the **repo root** (the directory
> containing `deploy/`), unless a step explicitly `cd`s elsewhere.

SageAttention and `flash-attn-4` are **not** on PyPI and are not in
`requirements.txt`; build them separately after the base deps. The FP8 kernels
are provided by the in-tree `joyomni_ops` library, built in the next step.

```bash
conda create -n joyai-video-edit python=3.10 -y
conda activate joyai-video-edit

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r deploy/requirements.txt
```

Both CUDA builds below (SageAttention and joyomni_ops) need **`nvcc` ≥ 12.8 in
the env** — Blackwell (`sm_100`/`sm_120`) support landed in 12.8; an older nvcc
either refuses the arch or emits no valid kernel. Install it into the conda env
(self-contained) and verify:

```bash
conda install -c conda-forge cuda-nvcc=12.8 cuda-cudart-dev=12.8 \
                              libcublas-dev libcusparse-dev libcusolver-dev
nvcc --version | grep release                    # -> release 12.8
nvcc --list-gpu-code | grep -E "sm_100|sm_120"   # Blackwell target present
```

(`libcublas-dev` / `libcusparse-dev` / `libcusolver-dev` supply the
`cublas_v2.h` / `cusparse.h` / `cusolverDn.h` headers that PyTorch's CUDA
headers include during the build.)

Then install the attention and kernel dependencies:

- **SageAttention 2.2.0** (*RTX 5090 only: GeForce runs fp32-accum SDPA at half
  rate, so int8 sage wins there; RTX PRO 6000 is net faster on plain cuDNN at
  the serving resolutions, and B200 uses FA4*) — INT8 quantized attention, used for all
  DiT denoise attention when `JOYOMNI_SAGE_ATTN=1`. Build from source with the
  bundled CUDA-graph stream fix:

  ```bash
  # from the repo root
  git clone https://github.com/thu-ml/SageAttention.git deploy/tmp/SageAttention
  cd deploy/tmp/SageAttention
  git checkout d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
  git apply ../../sageattention-cudagraph-stream.patch
  export CUDA_HOME=$CONDA_PREFIX
  export TORCH_CUDA_ARCH_LIST=12.0
  EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 python setup.py install
  cd -
  ```

  The patch routes every kernel launch through `at::cuda::getCurrentCUDAStream()`
  instead of the default stream — without it, upstream SageAttention records
  empty CUDA graphs (kernels escape capture) and the server's graph path
  produces noise. `JOYOMNI_SAGE_ATTN` is the only switch: default `0` → SDPA
  (cuDNN). Launch with `JOYOMNI_SAGE_ATTN=1` on an RTX 5090 (see §4); leave it
  unset on RTX PRO 6000 / B200.
- **flash-attn-4** (`4.0.0b13`, *required on FA4 machines — B200; skip on
  RTX PRO 6000 / 5090, its JIT does not support sm_120*) — provides
  `flash_attn.cute`; kernels JIT at runtime, no build step. Deps must be
  pinned exactly:

  ```bash
  python -m pip install flash-attn-4==4.0.0b13 \
    nvidia-cutlass-dsl==4.5.1 quack-kernels==0.4.1 apache-tvm-ffi==0.1.12
  ```

  If it can't be imported or its kernel fails at runtime, the DiT
  automatically falls back to cuDNN.
- **joyomni_ops** — in-tree CUDA op library ([`deploy/joyomni_ops/`](deploy/joyomni_ops/))
  providing the FP8 GEMM + fused norm/rope kernels the DiT uses (extracted from
  sgl-kernel, Apache-2.0, no `sgl_kernel`/`sglang` runtime dependency).

  The FP8 GEMM is built with [cutlass](https://github.com/NVIDIA/cutlass)
  (nvcc ≥ 12.8 for Blackwell — installed above). Build against a pinned
  cutlass checkout:

  ```bash
  git clone https://github.com/NVIDIA/cutlass.git deploy/tmp/cutlass
  git -C deploy/tmp/cutlass checkout dcf215af
  # build only this machine's arch (the default is a 5-arch fat binary —
  # sm_80..120a — which multiplies compile time ~5x); auto-detected:
  export JOYOMNI_OPS_CUDA_ARCHS=$(python -c "import torch; cc = torch.cuda.get_device_capability(0); print(f'{cc[0]}{cc[1]}a' if cc[0] >= 9 else f'{cc[0]}{cc[1]}')")
  echo "building joyomni_ops for sm_$JOYOMNI_OPS_CUDA_ARCHS"
  JOYOMNI_OPS_CUTLASS_DIR=$(pwd)/deploy/tmp/cutlass \
    python -m pip install --no-build-isolation ./deploy/joyomni_ops
  ```

  (`--no-build-isolation` reuses the env's existing `setuptools`/`torch` instead
  of pip fetching them into an isolated build env — required behind a restricted
  index/mirror, and it ensures the extension builds against the installed torch.)

> If you can't provide CUDA ≥ 12.8 (or cutlass), build the light variant
> (`JOYOMNI_OPS_NO_FP8=1 python -m pip install --no-build-isolation ./deploy/joyomni_ops`) and disable
> **both** FP8 paths — `JOYOMNI_FP8_IMG=0 JOYOMNI_FP8_TXT=0` — so nothing calls the
> FP8 kernel; the DiT then runs those Linears in bf16. SageAttention is
> independent of this — if absent, attention uses the SDPA/cuDNN path.

Verify the key runtime imports:

```bash
python - <<'PY'
import torch, cv2, av, transformers, diffusers
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| avail", torch.cuda.is_available(), "| gpus", torch.cuda.device_count())
print("cv2", cv2.__version__, "| transformers", transformers.__version__)
try:
    import sageattention; print("sageattention: OK (used when JOYOMNI_SAGE_ATTN=1 - RTX 5090)")
except Exception as e:
    print("sageattention: absent -> SDPA/cuDNN (only needed when JOYOMNI_SAGE_ATTN=1)")
try:
    import flash_attn.cute; print("flash_attn.cute: OK (FA4 importable; kernels JIT at first use)")
except Exception as e:
    print("flash_attn: absent (optional) -> sage/SDPA path")
try:
    import joyomni_ops; print("joyomni_ops: OK | has_fp8 =", joyomni_ops.has_fp8())
except Exception as e:
    print("joyomni_ops: MISSING ->", e, "(build deploy/joyomni_ops; set JOYOMNI_FP8_IMG=0 to skip FP8)")
PY
```

---

## 3. Fetch the weights

All weights live under `deploy/deps/checkpoints/` (~51 GB total). Create it and
download each dependency.

```bash
mkdir -p deploy/deps/checkpoints
```

**3a. DiT + xVAE** — the released JoyAI-Video-Edit weight repo:

```bash
hf download jdopensource/JoyAI-Video-Edit \
  --repo-type model \
  --local-dir deploy/deps/checkpoints/JoyAI-Video-Edit \
  --include "dit/joyai_video_edit_dit_0811.pth" "vae/*"
```

> The repo also ships the older `dit/joyai_video_edit_dit_0804.pth` (~32.5 GB);
> `--include` skips it — the server uses **0811**.

This should produce:

```
deploy/deps/checkpoints/JoyAI-Video-Edit/dit/joyai_video_edit_dit_0811.pth
deploy/deps/checkpoints/JoyAI-Video-Edit/vae/config.json
deploy/deps/checkpoints/JoyAI-Video-Edit/vae/diffusion_pytorch_model.safetensors
```

**3b. Text/vision encoder** — MiMo-VL:

```bash
hf download XiaomiMiMo/MiMo-VL-7B-RL-2508 \
  --repo-type model \
  --local-dir deploy/deps/checkpoints/MiMo-VL-7B-RL-2508
```

**3c. ONNX detectors** (*optional*):

```bash
# YuNet face detector (OpenCV Zoo, git-LFS — use the media.githubusercontent URL)
curl -L -o deploy/deps/checkpoints/face_detection_yunet_2023mar.onnx \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

YOLOv8n must be exported at **`imgsz=320`** — the server loads it via `cv2.dnn` at
a fixed 320×320 (see `_person_present`), so a default (640) or dynamic export
throws a Reshape error at load, and third-party pre-exported `yolov8n.onnx` on the
Hub (typically 640/dynamic) will *not* load. Installing `ultralytics` drags in a
full stack (its own torch/CUDA wheels + non-headless `opencv-python`) that would
overwrite this project's pinned `torch` and `opencv-python-headless` — so export
in a **throwaway env**, never the deploy env:

```bash
conda create -n yolo-export python=3.10 -y
conda activate yolo-export

# CPU-only torch is enough for export and avoids pulling multi-GB CUDA wheels.
pip install --index-url https://pypi.org/simple/ \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  ultralytics onnx onnxslim

# ultralytics pulls non-headless opencv-python (needs libGL, absent on headless
# boxes -> "libGL.so.1: cannot open shared object file"). Swap to headless:
pip uninstall -y opencv-python
pip install --index-url https://pypi.org/simple/ opencv-python-headless

python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=320, opset=12)"

conda deactivate
mv yolov8n.onnx deploy/deps/checkpoints/    # move the export into place
conda env remove -n yolo-export -y          # optional: drop the throwaway env
```

| File | Purpose |
| --- | --- |
| `face_detection_yunet_2023mar.onnx` | YuNet face-presence gate |
| `yolov8n.onnx` | YOLOv8n person-presence gate |

> The detectors are **optional**: if a file is missing the server just disables
> that gate (edits run unconditionally). The DiT, VAE, and MiMo-VL weights are the
> only hard requirements.

Final tree:

```
deploy/deps/checkpoints/
├── JoyAI-Video-Edit/
│   ├── dit/joyai_video_edit_dit_0811.pth
│   └── vae/{config.json, diffusion_pytorch_model.safetensors}
├── MiMo-VL-7B-RL-2508/
├── face_detection_yunet_2023mar.onnx
└── yolov8n.onnx
```

---

## 4. Launch

Every setting is a plain environment variable with a working default. The
launcher activates conda itself (`JOYOMNI_CONDA_SH` / `JOYOMNI_CONDA_ENV`) and
runs every stage on one device; when several GPU models share a checkout, each
gets its own `JOYOMNI_CACHE_ROOT`. UI: `http://<server-ip>:8080/`.

**NVIDIA B200 — 720p @ 30 FPS:**

```bash
JOYOMNI_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh \
JOYOMNI_CONDA_ENV=joyai-video-edit \
JOYOMNI_CACHE_ROOT=$PWD/deploy/deps/cache_b200 \
JOYOMNI_WIDTH=1248 JOYOMNI_HEIGHT=720 JOYOMNI_FPS=30 \
bash deploy/run_server.sh
```

**RTX PRO 6000 — 480p @ 24 FPS:**

```bash
JOYOMNI_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh \
JOYOMNI_CONDA_ENV=joyai-video-edit \
JOYOMNI_CACHE_ROOT=$PWD/deploy/deps/cache_pro6000 \
bash deploy/run_server.sh
```

**RTX PRO 6000 — 720p @ 16 FPS:**

```bash
JOYOMNI_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh \
JOYOMNI_CONDA_ENV=joyai-video-edit \
JOYOMNI_CACHE_ROOT=$PWD/deploy/deps/cache_pro6000 \
JOYOMNI_WIDTH=1248 JOYOMNI_HEIGHT=720 JOYOMNI_FPS=16 \
bash deploy/run_server.sh
```

**RTX 4090 (24 GB):** use [`run_server_4090.sh`](deploy/run_server_4090.sh), see §5.

**RTX 5090 — 480p @ 24 FPS:**

```bash
JOYOMNI_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh \
JOYOMNI_CONDA_ENV=joyai-video-edit \
JOYOMNI_CACHE_ROOT=$PWD/deploy/deps/cache_rtx5090 \
JOYOMNI_SAGE_ATTN=1 JOYOMNI_FP8_FAST_ACCUM=1 JOYOMNI_LOW_VRAM=1 \
bash deploy/run_server.sh
```

Key variables:

| Variable | Meaning |
| --- | --- |
| `JOYOMNI_CONDA_SH` / `JOYOMNI_CONDA_ENV` | conda `profile.d/conda.sh` + env name/prefix; the launcher activates it itself. Omit both to use the caller's python. |
| `JOYOMNI_DEVICE` | CUDA device for all stages (default `cuda:0`). |
| `JOYOMNI_HOST` / `JOYOMNI_PORT` | bind address (default `0.0.0.0:8080`). |
| `JOYOMNI_WIDTH` / `JOYOMNI_HEIGHT` / `JOYOMNI_FPS` | Output resolution and frame rate (default `840` / `480` / `24` = 480p @ 24 FPS). Per-GPU commands above. |
| `JOYOMNI_FP8_IMG` / `JOYOMNI_FP8_TXT` | FP8 image / text paths via `joyomni_ops` (default `1` / `1`). Set both `0` to run bf16 (e.g. a `JOYOMNI_OPS_NO_FP8=1` build). |
| `JOYOMNI_CUDA_GRAPH` | capture the steady-state chunk loop into a CUDA graph (default `1`; the biggest single speedup). `0` runs eager. |
| `JOYOMNI_SAGE_ATTN` | SageAttention for all DiT attention (default `0` → SDPA/cuDNN; set `1` on RTX 5090). |
| `JOYOMNI_FP8_FAST_ACCUM` | FP8 GEMMs accumulate in fp16 via a Triton kernel (default `0`; set `1` on RTX 5090, where fp32-accumulate tensor MMAs run at half rate — they run at full rate on RTX PRO 6000 / B200, so leave it unset there). |
| `JOYOMNI_LOW_VRAM` | low-VRAM layout — CPU-staged FP8 DiT load + text-encoder CPU offload (default `0`; set `1` on ≤48 GB cards — the full layout needs ~46.5 GB steady at 720p16). 480p24 measured ~21.5 GiB resident / ~28 GiB peak under a 30 GiB allocator cap — fits 32 GB cards. |
| `JOYOMNI_CACHE_ROOT` | compile-cache root — torchinductor / triton / nv_compute caches live under it (default `deploy/deps/cache`). Give each GPU model its own root when several share a checkout (per-card commands above). |
| `JOYOMNI_CKPT_ROOT` | override the checkpoints dir (default `deploy/deps/checkpoints`). |
| `JOYOMNI_DIT_CKPT` / `JOYOMNI_VAE_CKPT` / `JOYOMNI_TEXT_ENCODER_CKPT` / `JOYOMNI_FACE_ONNX` / `JOYOMNI_PERSON_ONNX` | override individual weight paths (default: derived from `JOYOMNI_CKPT_ROOT`). |
| `JOYOMNI_RECORD_DIR` | recording output dir. |
| `PE_MODEL` / `OPENAI_BASE_URL` / `OPENAI_API_KEY` | Prompt-enhancement endpoint: OpenAI-compatible, or Anthropic-protocol when the base URL contains `/anthropic` (key sent as `Authorization: Bearer`). If unset, the server falls back to the raw user prompt. |
| `JOYOMNI_DIT_OFFLOAD_BLOCKS` | stream this many FP8 DiT blocks from pinned host RAM instead of keeping them in VRAM ([`xvideo/blockswap.py`](deploy/xvideo/blockswap.py); default `0`; needs `JOYOMNI_LOW_VRAM=1`). Each block frees 0.375 GiB of VRAM and costs ~0.94 GiB of PCIe traffic per chunk. See §5. |
| `JOYOMNI_DIT_OFFLOAD_SLOTS` | GPU slot ring for streamed blocks (default `2`; each slot is 0.375 GiB). |
| `JOYOMNI_DIT_META_INIT` | build the DiT on the meta device and adopt the mmap'd checkpoint in place (default `0`; low-VRAM mode only). Cuts the host-RAM peak during load from roughly 55 GB (the CPU build allocates the MLPs in fp32) to page cache. |

---

## 5. RTX 4090 (24 GB, Ada `sm_89`)

> **Status: not yet validated on real RTX 4090 hardware.** Everything below was
> derived from the code and the published RTX 5090 numbers, and is checked by a
> probe script you run first. Expect to tune the preset to your machine.

**Why the 5090 profile does not fit as-is.** The 16B DiT is 15.0 GiB in FP8. The
RTX 5090 low-VRAM profile measured ~21.5 GiB resident / ~28 GiB peak at 480p
(§4), and a 24 GB card exposes ~23.5 GiB (less if it also drives a desktop). The
4090 profile therefore streams a few DiT blocks from pinned host RAM, overlapped
with compute, and offers lower-resolution presets. Estimated budgets:
[`experiments/c_vram_budget/estimate.py`](experiments/c_vram_budget/estimate.py),
trade-offs: [`notes/comparison-rtx4090.md`](notes/comparison-rtx4090.md).

**Machine requirements**

| | |
| --- | --- |
| GPU | RTX 4090 24 GB in a PCIe 4.0 x16 slot. Ideally **not** driving a display (use an iGPU / second GPU): every GiB the desktop takes comes out of the model's budget. |
| Driver | R570 or newer (CUDA 12.8 runtime used by the `cu128` torch wheels). |
| Host RAM | **64 GB recommended.** Text encoder 16 GB (pinned with `JOYOMNI_TE_PIN=1`) + streamed DiT blocks (3 GB at the default preset) + page cache while the 32 GB checkpoint loads. With 48 GB, set `JOYOMNI_TE_PIN=0`. |
| Disk | ~60 GB free (51 GB weights + compile caches). An NVMe makes the first load much faster. |

**5.1 Environment.** Follow §2, with these per-arch changes:

```bash
conda create -n joyai-video-edit python=3.10 -y
conda activate joyai-video-edit
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r deploy/requirements.txt
conda install -c conda-forge cuda-nvcc=12.8 cuda-cudart-dev=12.8 \
                              libcublas-dev libcusparse-dev libcusolver-dev

# SageAttention 2.2.0: same commit + CUDA-graph patch, but build the sm_89 kernels
git clone https://github.com/thu-ml/SageAttention.git deploy/tmp/SageAttention
cd deploy/tmp/SageAttention
git checkout d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
git apply ../../sageattention-cudagraph-stream.patch
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST=8.9
EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=16 python setup.py install
cd -

# joyomni_ops for sm_89 (the cutlass FP8 GEMM has a dedicated sm_89 path)
git clone https://github.com/NVIDIA/cutlass.git deploy/tmp/cutlass
git -C deploy/tmp/cutlass checkout dcf215af
JOYOMNI_OPS_CUDA_ARCHS=89 JOYOMNI_OPS_CUTLASS_DIR=$(pwd)/deploy/tmp/cutlass \
  python -m pip install --no-build-isolation ./deploy/joyomni_ops
```

Skip **flash-attn-4** (Blackwell/Hopper only; the DiT falls back automatically).
Use `MAX_JOBS` that fits your RAM (each nvcc job can take 2-4 GB).

**5.2 Weights.** Exactly §3 (DiT 0811 + xVAE, MiMo-VL, optional detectors).

**5.3 Probe the GPU before the first launch** (no weights needed, ~1 min):

```bash
PYTHONPATH=deploy python deploy/tools/probe_rtx4090.py
```

It checks the sm_89 FP8 GEMM, whether the `JOYOMNI_FP8_FAST_ACCUM` Triton GEMM
compiles and is faster than cutlass on your card, the SageAttention sm_89
kernels incl. CUDA-graph replay (fails if the stream patch was not applied),
pinned H2D bandwidth, and that DiT block streaming is exact both eagerly and
inside a captured CUDA graph. Fix every `FAIL` before launching. If
`fast_accum_speed` is slower than `cutlass_fp8_gemm_speed`, launch with
`JOYOMNI_FP8_FAST_ACCUM=0`.

**5.4 Launch**

```bash
JOYOMNI_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh \
JOYOMNI_CONDA_ENV=joyai-video-edit \
bash deploy/run_server_4090.sh                       # balanced: 840x480 @ 18 FPS
# JOYOMNI_4090_PRESET=fast bash deploy/run_server_4090.sh   # 672x384 @ 24 FPS
# JOYOMNI_4090_PRESET=safe bash deploy/run_server_4090.sh   # 672x384 @ 15 FPS, most headroom
```

[`run_server_4090.sh`](deploy/run_server_4090.sh) only sets defaults
(`JOYOMNI_LOW_VRAM=1 JOYOMNI_SAGE_ATTN=1 JOYOMNI_FP8_FAST_ACCUM=1
JOYOMNI_DIT_META_INIT=1 JOYOMNI_DIT_OFFLOAD_BLOCKS=<preset>`, its own
`JOYOMNI_CACHE_ROOT=deploy/deps/cache_rtx4090`) and then runs `run_server.sh`;
any variable you export wins. The FPS values are starting estimates from
[`experiments/a_block_streaming/cost_model.py`](experiments/a_block_streaming/cost_model.py), not measurements.

The first start compiles the VAE (`max-autotune`) and captures the CUDA graph
for both orientations; that can take many minutes and is cached for later
starts. Watch the log for:

- `#####[BLOCK-STREAM] DiT block streaming: N blocks [...] -> net VRAM saved X GiB`
- `#####[STREAM] runtime loaded; device memory used X/Y GiB`: this is your real
  resident number. Keep ≥3 GiB free for prompt encoding and RV2V reference images.
- `#####[GRAPH] captured whole-chunk graph`: if you instead see `capture failed`,
  the session runs eager (slower). Usually that means you are out of memory, so
  raise `JOYOMNI_DIT_OFFLOAD_BLOCKS`.

Open `http://localhost:8080`. Browsers allow the webcam only on `https://` or
`localhost`, so for a remote 4090 box use an SSH tunnel
(`ssh -L 8080:localhost:8080 user@gpu-box`) and open `http://localhost:8080` locally.

**5.5 Tuning**

| Symptom | Change |
| --- | --- |
| `CUDA out of memory` at load / warmup / first session | `JOYOMNI_DIT_OFFLOAD_BLOCKS` +4, or `JOYOMNI_4090_PRESET=safe`; free the GPU from the desktop |
| OOM only with a reference image (RV2V) | RV2V adds ~1.25 GiB of KV at 480p: +4 offloaded blocks |
| Output stutters / frames dropped, VRAM has headroom | lower `JOYOMNI_DIT_OFFLOAD_BLOCKS` (each block is ~46 ms of PCIe per chunk) or lower `JOYOMNI_FPS` |
| `fast_accum_speed` slower than cutlass in the probe | `JOYOMNI_FP8_FAST_ACCUM=0` |
| Host RAM exhausted / swapping at load | `JOYOMNI_TE_PIN=0`; keep `JOYOMNI_DIT_META_INIT=1` |
| `meta-init DiT load failed ... falling back` in the log | still works, but needs ~55 GB RAM at load; please report the error |
