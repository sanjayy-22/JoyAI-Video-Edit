"""CPU test for the JOYOMNI_DIT_META_INIT loader path (no GPU / weights needed).

Run from deploy/:  PYTHONPATH=. python -m pytest tests/test_meta_init.py -q

Builds a tiny Transformer3DModel, saves it the way the release checkpoint is
stored (prefixed keys), reloads it through torch.load(mmap=True) +
_build_dit_meta_assign, and checks every tensor matches and stays file-backed
(no CPU materialization).
"""
from __future__ import annotations

import torch

from xvideo.config import ExpConfig
from xvideo.models import models as M
from xvideo.models.dit import Transformer3DModel

TINY = {
    "hidden_size": 64, "in_channels": 8, "heads_num": 2, "mm_double_blocks_depth": 2,
    "out_channels": 8, "patch_size": [1, 1, 1], "rope_dim_list": [8, 12, 12],
    "text_states_dim": 32, "theta": 256, "chunk_size": 1, "causal": True,
    "local_window_size": 3, "global_sink_chunk": True, "source_id_rope_dim": 32,
    "source_id_rope_theta": 256.0,
}


def test_meta_assign_matches_cpu_build(tmp_path):
    torch.manual_seed(0)
    ref = Transformer3DModel(dtype=torch.bfloat16, device="cpu", **TINY).to(torch.bfloat16)
    ckpt = tmp_path / "dit.pth"
    torch.save({"model": {"module." + k: v for k, v in ref.state_dict().items()}}, ckpt)

    cfg = ExpConfig()
    cfg.dit_arch_config = {"params": dict(TINY)}
    sd = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)["model"]
    sd = {k[len("module."):]: v for k, v in sd.items()}
    sd = M._inflate_img_in(sd, M._img_in_weight_shape(cfg))

    model = M._build_dit_meta_assign(cfg, sd, torch.bfloat16)

    ref_sd = ref.state_dict()
    got_sd = model.state_dict()
    assert ref_sd.keys() == got_sd.keys()
    for k in ref_sd:
        assert got_sd[k].dtype == torch.bfloat16, k
        assert torch.equal(got_sd[k], ref_sd[k]), k
    # adopted in place: parameters share storage with the mmap'd checkpoint tensors
    p = model.double_blocks[0].img_mlp.net[0].proj.weight
    src = sd["double_blocks.0.img_mlp.net.0.proj.weight"]
    assert p.untyped_storage().data_ptr() == src.untyped_storage().data_ptr()
    assert not any(t.is_meta for t in model.parameters())


def test_cpu_build_mlp_is_fp32_without_cast():
    """Documents why meta init matters: FeedForward ignores the dtype argument."""
    m = Transformer3DModel(dtype=torch.bfloat16, device="cpu", **TINY)
    assert m.double_blocks[0].img_mlp.net[0].proj.weight.dtype == torch.float32
    assert m.double_blocks[0].img_attn_qkv.weight.dtype == torch.bfloat16


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        test_meta_assign_matches_cpu_build(Path(d))
    test_cpu_build_mlp_is_fp32_without_cast()
    print("meta-init loader: OK")
