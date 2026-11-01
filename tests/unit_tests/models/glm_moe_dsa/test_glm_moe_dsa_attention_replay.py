# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Activation-checkpoint replay of the cuDNN sparse-attention output (GLM DSA, GPU only)."""

import copy

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common import cudnn_sparse_attention as shared_adapter
from nemo_automodel.components.models.deepseek_v3.rope_utils import freqs_cis_from_position_ids
from nemo_automodel.components.models.glm_moe_dsa.model import GlmMoeDsaForCausalLM
from nemo_automodel.components.models.glm_moe_dsa.optimized_kernels import (
    is_cudnn_dsa_available,
    prepare_cudnn_dsa_packed_metadata,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_cudnn_dsa_available() or not shared_adapter._HAS_FLASH_MLA,
    reason="requires CUDA with the cuDNN DSA and FlashMLA kernels",
)

TOKENS = 1024


def _model(reuse_attention_output: bool) -> GlmMoeDsaForCausalLM:
    config = GlmMoeDsaConfig(
        vocab_size=256,
        hidden_size=512,
        num_attention_heads=16,
        num_key_value_heads=16,
        num_hidden_layers=2,
        intermediate_size=512,
        moe_intermediate_size=256,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=False,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        attention_bias=False,
        q_lora_rank=256,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        mlp_layer_types=["dense", "dense"],
        indexer_types=["full", "shared"],
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    )
    backend = BackendConfig(
        linear="torch",
        attn="cudnn",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
        reuse_sparse_attention_output_on_recompute=reuse_attention_output,
    )
    torch.manual_seed(1234)
    model = GlmMoeDsaForCausalLM(config, backend=backend)
    model.initialize_weights(buffer_device=torch.device("cuda"), dtype=torch.bfloat16)
    return model.cuda().train()


def _inputs(model: GlmMoeDsaForCausalLM):
    device = torch.device("cuda")
    torch.manual_seed(7)
    x = torch.randn(TOKENS, model.config.hidden_size, device=device, dtype=torch.bfloat16)
    position_ids = torch.arange(TOKENS, device=device)
    freqs_cis = freqs_cis_from_position_ids(
        position_ids, model.model.freqs.to(device), qkv_format="thd", for_fused_rope=False, cp_size=1
    )
    cu_seqlens = torch.tensor([0, TOKENS], dtype=torch.int32, device=device)
    packed = prepare_cudnn_dsa_packed_metadata(cu_seqlens, TOKENS)
    kwargs = {
        "cu_seqlens": cu_seqlens,
        "qkv_format": "thd",
        "_cudnn_dsa_packed_metadata": packed,
        "_cudnn_dsa_topk_length": packed.causal_lengths.clamp_max(int(model.config.index_topk)).contiguous(),
        "_cudnn_dsa_all_rows_nonempty": packed.all_rows_nonempty,
        "_cudnn_dsa_valid_row_indices": packed.valid_row_indices,
    }
    return x, freqs_cis, kwargs


class _CountingForward:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.inner(*args, **kwargs)


def _run(block, x, freqs_cis, kwargs):
    x = x.clone().requires_grad_(True)
    out, _ = block(x, freqs_cis=freqs_cis, **kwargs)
    out.float().square().mean().backward()
    grads = {name: param.grad.clone() for name, param in block.named_parameters() if param.grad is not None}
    return out.detach(), x.grad.clone(), grads


def test_recompute_reuses_recorded_attention_output(monkeypatch):
    model = _model(reuse_attention_output=True)
    block = model.model.layers["0"]
    x, freqs_cis, kwargs = _inputs(model)
    counter = _CountingForward(shared_adapter._FLASH_MLA_SPARSE_FWD)
    monkeypatch.setattr(shared_adapter, "_FLASH_MLA_SPARSE_FWD", counter)

    reference = copy.deepcopy(block)
    ref_out, ref_x_grad, ref_grads = _run(reference, x, freqs_cis, kwargs)
    assert counter.calls == 1

    plain = checkpoint_wrapper(copy.deepcopy(block), checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    counter.calls = 0
    _run(plain, x, freqs_cis, kwargs)
    assert counter.calls == 2, "without replay the recompute runs the forward kernel again"

    replayed_block = copy.deepcopy(block)
    replayed = checkpoint_wrapper(
        replayed_block,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        context_fn=replayed_block.nemo_checkpoint_context_fn(None),
    )
    counter.calls = 0
    out, x_grad, grads = _run(replayed, x, freqs_cis, kwargs)
    assert counter.calls == 1, "the recompute must reuse the recorded output"
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(x_grad, ref_x_grad)
    assert grads.keys() == ref_grads.keys()
    for name, grad in grads.items():
        torch.testing.assert_close(grad, ref_grads[name], msg=lambda m, n=name: f"{n}: {m}")


def test_flag_controls_whether_the_attention_replay_is_bound():
    from nemo_automodel.components.models.glm_moe_dsa.layers import glm_dsa_attention_replay

    for flag in (False, True):
        block = _model(reuse_attention_output=flag).model.layers["1"]
        assert block.skip_topk
        forward_ctx, _ = block.nemo_checkpoint_context_fn(None)()
        with forward_ctx:
            assert (glm_dsa_attention_replay.current() is not None) is flag
        assert glm_dsa_attention_replay.current() is None
