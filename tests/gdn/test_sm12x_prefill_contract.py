"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import ast
from pathlib import Path

import pytest
import torch

from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.gdn_kernels.blackwell_sm12x.gdn_prefill import (
    _build_sm12x_gate_cumsum,
    _build_sm12x_gate_cumsum_device,
    _build_sm12x_chunk_metadata,
    _build_sm12x_chunk_metadata_device,
    _prepare_sm12x_gate_cumsum,
    _prepare_sm12x_chunk_metadata,
    _sm12x_upper_bound_chunks,
    _validate_sm12x_gdn_prefill_inputs,
    chunk_gated_delta_rule_sm12x,
)


SM12X_KERNEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "flashinfer"
    / "gdn_kernels"
    / "blackwell_sm12x"
)


def _make_inputs(
    *,
    total_tokens: int = 64,
    num_seqs: int = 1,
    num_q_heads: int = 4,
    num_k_heads: int = 4,
    num_v_heads: int = 8,
    head_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
):
    q = torch.empty(total_tokens, num_q_heads, head_size, dtype=dtype)
    k = torch.empty(total_tokens, num_k_heads, head_size, dtype=dtype)
    v = torch.empty(total_tokens, num_v_heads, head_size, dtype=dtype)
    num_o_heads = max(num_q_heads, num_v_heads)
    gate = torch.empty(total_tokens, num_o_heads, dtype=torch.float32)
    beta = torch.empty_like(gate)
    output = torch.empty(total_tokens, num_o_heads, head_size, dtype=dtype)
    cu_seqlens = torch.linspace(
        0,
        total_tokens,
        num_seqs + 1,
        dtype=torch.int32,
    )
    state = torch.empty(
        num_seqs,
        num_o_heads,
        head_size,
        head_size,
        dtype=torch.float32,
    )
    return q, k, v, gate, beta, output, cu_seqlens, state


def _head_mapping(
    out_head: int,
    num_q_heads: int,
    num_v_heads: int,
) -> tuple[int, int, int]:
    if num_q_heads >= num_v_heads:
        group_size = num_q_heads // num_v_heads
        kv_head = out_head // group_size
        return out_head, kv_head, kv_head

    group_size = num_v_heads // num_q_heads
    qk_head = out_head // group_size
    return qk_head, qk_head, out_head


def _reference_sm12x_gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_f32 = q.float()
    k_f32 = k.float()
    v_f32 = v.float()
    gate_f32 = gate.float()
    beta_f32 = beta.float()
    num_seqs = cu_seqlens.numel() - 1
    num_q_heads = q.size(1)
    num_v_heads = v.size(1)
    num_o_heads = max(num_q_heads, num_v_heads)
    head_size = q.size(2)

    output = torch.empty(
        q.size(0),
        num_o_heads,
        head_size,
        dtype=torch.float32,
        device=q.device,
    )
    final_state = torch.empty(
        num_seqs,
        num_o_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device=q.device,
    )

    for seq_id in range(num_seqs):
        start = int(cu_seqlens[seq_id].item())
        end = int(cu_seqlens[seq_id + 1].item())
        for out_head in range(num_o_heads):
            q_head, k_head, v_head = _head_mapping(
                out_head,
                num_q_heads,
                num_v_heads,
            )
            state = (
                initial_state[seq_id, out_head].float().clone()
                if initial_state is not None
                else torch.zeros(head_size, head_size, dtype=torch.float32)
            )
            for token in range(start, end):
                state = state * gate_f32[token, out_head]
                projected = state @ k_f32[token, k_head]
                delta = (v_f32[token, v_head] - projected) * beta_f32[
                    token,
                    out_head,
                ]
                state = state + delta[:, None] * k_f32[token, k_head][None, :]
                output[token, out_head] = state @ (q_f32[token, q_head] * scale)
            final_state[seq_id, out_head] = state

    return output, final_state


def test_sm12x_gdn_prefill_contract_accepts_qwen_gva_shape():
    q, k, v, gate, beta, output, cu_seqlens, state = _make_inputs()

    config = _validate_sm12x_gdn_prefill_inputs(
        q=q,
        k=k,
        v=v,
        gate=gate,
        beta=beta,
        output=output,
        cu_seqlens=cu_seqlens,
        initial_state=state,
        output_state=state,
        checkpoint_every_n_tokens=0,
        cu_checkpoints=None,
        output_checkpoints=None,
    )

    assert config.total_tokens == 64
    assert config.num_q_heads == 4
    assert config.num_k_heads == 4
    assert config.num_v_heads == 8
    assert config.num_o_heads == 8
    assert config.head_size == 128
    assert config.is_gva
    assert not config.is_gqa


def test_sm12x_gdn_prefill_contract_requires_head_size_128():
    q, k, v, gate, beta, output, cu_seqlens, state = _make_inputs(head_size=64)

    with pytest.raises(ValueError, match="head_size=128"):
        _validate_sm12x_gdn_prefill_inputs(
            q=q,
            k=k,
            v=v,
            gate=gate,
            beta=beta,
            output=output,
            cu_seqlens=cu_seqlens,
            initial_state=state,
            output_state=state,
            checkpoint_every_n_tokens=0,
            cu_checkpoints=None,
            output_checkpoints=None,
        )


def test_sm12x_gdn_prefill_contract_rejects_incompatible_head_grouping():
    q, k, v, gate, beta, output, cu_seqlens, state = _make_inputs(
        num_q_heads=4,
        num_k_heads=2,
        num_v_heads=8,
    )

    with pytest.raises(ValueError, match="GVA"):
        _validate_sm12x_gdn_prefill_inputs(
            q=q,
            k=k,
            v=v,
            gate=gate,
            beta=beta,
            output=output,
            cu_seqlens=cu_seqlens,
            initial_state=state,
            output_state=state,
            checkpoint_every_n_tokens=0,
            cu_checkpoints=None,
            output_checkpoints=None,
        )


def test_sm12x_gdn_prefill_contract_rejects_bad_state_shape():
    q, k, v, gate, beta, output, cu_seqlens, _ = _make_inputs()
    bad_state = torch.empty(1, 8, 128, 64, dtype=torch.float32)

    with pytest.raises(ValueError, match="initial_state shape mismatch"):
        _validate_sm12x_gdn_prefill_inputs(
            q=q,
            k=k,
            v=v,
            gate=gate,
            beta=beta,
            output=output,
            cu_seqlens=cu_seqlens,
            initial_state=bad_state,
            output_state=None,
            checkpoint_every_n_tokens=0,
            cu_checkpoints=None,
            output_checkpoints=None,
        )


def test_sm12x_gdn_prefill_package_does_not_import_tcgen05():
    for path in SM12X_KERNEL_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported_names = [node.module or ""]
                imported_names.extend(alias.name for alias in node.names)
            else:
                continue

            assert all("tcgen05" not in name for name in imported_names), path


def test_sm12x_chunk_metadata_matches_expected_chunk_indices():
    cu_seqlens = torch.tensor([0, 64, 65, 194], dtype=torch.int32)

    chunk_indices, chunk_offsets = _build_sm12x_chunk_metadata(cu_seqlens)

    torch.testing.assert_close(
        chunk_indices,
        torch.tensor(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [2, 1],
                [2, 2],
            ],
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(
        chunk_offsets,
        torch.tensor([0, 1, 2, 5], dtype=torch.int32),
    )


def test_sm12x_prepare_chunk_metadata_uses_cpu_reference_for_cpu_tensor():
    cu_seqlens = torch.tensor([0, 64, 65, 194], dtype=torch.int32)

    chunk_indices, chunk_offsets = _prepare_sm12x_chunk_metadata(
        cu_seqlens,
        total_tokens=194,
    )

    expected_indices, expected_offsets = _build_sm12x_chunk_metadata(cu_seqlens)
    torch.testing.assert_close(chunk_indices, expected_indices)
    torch.testing.assert_close(chunk_offsets, expected_offsets)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm12x_chunk_metadata_device_matches_cpu_reference():
    cu_seqlens_cpu = torch.tensor([0, 64, 65, 194], dtype=torch.int32)
    expected_indices, expected_offsets = _build_sm12x_chunk_metadata(cu_seqlens_cpu)

    chunk_indices, chunk_offsets = _build_sm12x_chunk_metadata_device(
        cu_seqlens_cpu.cuda(),
        total_tokens=194,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(chunk_indices.cpu(), expected_indices)
    torch.testing.assert_close(chunk_offsets.cpu(), expected_offsets)


def test_sm12x_upper_bound_chunks_handles_ragged_positive_lengths():
    assert _sm12x_upper_bound_chunks(num_seqs=3, total_tokens=194) == 5


def test_sm12x_chunk_metadata_rejects_non_monotonic_cu_seqlens():
    cu_seqlens = torch.tensor([0, 64, 63], dtype=torch.int32)

    with pytest.raises(ValueError, match="monotonic"):
        _build_sm12x_chunk_metadata(cu_seqlens)


def test_sm12x_gate_cumsum_matches_per_sequence_log2_prefix():
    gate = torch.tensor(
        [
            [0.50, 1.00],
            [0.25, 0.75],
            [0.90, 0.80],
            [0.20, 0.40],
        ],
        dtype=torch.float32,
    )
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)

    actual = _build_sm12x_gate_cumsum(gate, cu_seqlens)
    expected = torch.empty_like(gate)
    expected[:2] = torch.log2(gate[:2] + 1e-10).cumsum(dim=0)
    expected[2:] = torch.log2(gate[2:] + 1e-10).cumsum(dim=0)

    torch.testing.assert_close(actual, expected)


def test_sm12x_prepare_gate_cumsum_uses_cpu_reference_for_cpu_tensor():
    gate = torch.full((4, 2), 0.5, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 1, 4], dtype=torch.int32)

    actual = _prepare_sm12x_gate_cumsum(gate, cu_seqlens)
    expected = _build_sm12x_gate_cumsum(gate, cu_seqlens)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm12x_gate_cumsum_device_matches_cpu_reference():
    gate_cpu = torch.tensor(
        [
            [0.50, 1.00, 0.70],
            [0.25, 0.75, 0.60],
            [0.90, 0.80, 0.40],
            [0.20, 0.40, 0.30],
            [0.95, 0.85, 0.75],
        ],
        dtype=torch.float32,
    )
    cu_seqlens_cpu = torch.tensor([0, 2, 5], dtype=torch.int32)
    expected = _build_sm12x_gate_cumsum(gate_cpu, cu_seqlens_cpu)

    actual = _build_sm12x_gate_cumsum_device(
        gate_cpu.cuda(),
        cu_seqlens_cpu.cuda(),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)


def test_sm12x_gdn_prefill_contract_rejects_total_token_mismatch():
    q, k, v, gate, beta, output, _, state = _make_inputs(num_seqs=2)
    cu_seqlens = torch.tensor([0, 32, 63], dtype=torch.int32)

    with pytest.raises(ValueError, match="total_tokens"):
        _validate_sm12x_gdn_prefill_inputs(
            q=q,
            k=k,
            v=v,
            gate=gate,
            beta=beta,
            output=output,
            cu_seqlens=cu_seqlens,
            initial_state=state,
            output_state=state,
            checkpoint_every_n_tokens=0,
            cu_checkpoints=None,
            output_checkpoints=None,
        )


def test_sm12x_prefill_entrypoint_requires_cuda_after_validation():
    q, k, v, gate, beta, output, cu_seqlens, state = _make_inputs()

    with pytest.raises(NotImplementedError, match="requires CUDA"):
        chunk_gated_delta_rule_sm12x(
            q=q,
            k=k,
            v=v,
            gate=gate,
            beta=beta,
            output=output,
            cu_seqlens=cu_seqlens,
            initial_state=state,
            output_state=state,
            scale=128**-0.5,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm12x_prefill_entrypoint_runs_cuda_without_initial_state():
    q, k, v, _, _, output, _, _ = _make_inputs(
        total_tokens=65,
        num_seqs=2,
    )
    q.zero_()
    k.zero_()
    v.zero_()
    output.zero_()
    gate = torch.full((65, 8), 0.5, dtype=torch.float32)
    beta = torch.ones_like(gate)
    cu_seqlens = torch.tensor([0, 1, 65], dtype=torch.int32)

    output_cuda = output.cuda()
    chunk_gated_delta_rule_sm12x(
        q=q.cuda(),
        k=k.cuda(),
        v=v.cuda(),
        gate=gate.cuda(),
        beta=beta.cuda(),
        output=output_cuda,
        cu_seqlens=cu_seqlens.cuda(),
        initial_state=None,
        output_state=None,
        scale=128**-0.5,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(output_cuda.cpu(), output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm12x_prefill_recurrent_kernel_matches_reference_gva():
    torch.manual_seed(0)
    total_tokens = 129
    num_q_heads = 1
    num_v_heads = 2
    head_size = 128
    q = (
        torch.randn(total_tokens, num_q_heads, head_size, dtype=torch.float32) * 0.05
    ).to(torch.bfloat16)
    k = (
        torch.randn(total_tokens, num_q_heads, head_size, dtype=torch.float32) * 0.05
    ).to(torch.bfloat16)
    v = (
        torch.randn(total_tokens, num_v_heads, head_size, dtype=torch.float32) * 0.05
    ).to(torch.bfloat16)
    gate = torch.rand(total_tokens, num_v_heads, dtype=torch.float32) * 0.2 + 0.75
    beta = torch.rand_like(gate) * 0.3 + 0.2
    cu_seqlens = torch.tensor([0, 64, 129], dtype=torch.int32)
    initial_state = torch.randn(2, num_v_heads, head_size, head_size) * 0.01
    output = torch.empty(total_tokens, num_v_heads, head_size, dtype=torch.bfloat16)
    output_state = torch.empty_like(initial_state)
    scale = head_size**-0.5

    expected_output, expected_state = _reference_sm12x_gdn_prefill(
        q,
        k,
        v,
        gate,
        beta,
        cu_seqlens,
        initial_state,
        scale,
    )

    output_cuda = output.cuda()
    output_state_cuda = output_state.cuda()
    chunk_gated_delta_rule_sm12x(
        q=q.cuda(),
        k=k.cuda(),
        v=v.cuda(),
        gate=gate.cuda(),
        beta=beta.cuda(),
        output=output_cuda,
        cu_seqlens=cu_seqlens.cuda(),
        initial_state=initial_state.cuda(),
        output_state=output_state_cuda,
        scale=scale,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output_cuda.cpu().float(),
        expected_output,
        rtol=1e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        output_state_cuda.cpu(),
        expected_state,
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm12x_prefill_recurrent_kernel_matches_reference_gqa_fp16():
    torch.manual_seed(1)
    total_tokens = 4
    num_q_heads = 2
    num_v_heads = 1
    head_size = 128
    q = (
        torch.randn(total_tokens, num_q_heads, head_size, dtype=torch.float32) * 0.04
    ).to(torch.float16)
    k = (
        torch.randn(total_tokens, num_v_heads, head_size, dtype=torch.float32) * 0.04
    ).to(torch.float16)
    v = (
        torch.randn(total_tokens, num_v_heads, head_size, dtype=torch.float32) * 0.04
    ).to(torch.float16)
    gate = torch.rand(total_tokens, num_q_heads, dtype=torch.float32) * 0.2 + 0.7
    beta = torch.rand_like(gate) * 0.25 + 0.1
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
    output = torch.empty(total_tokens, num_q_heads, head_size, dtype=torch.float16)
    output_state = torch.empty(1, num_q_heads, head_size, head_size)
    scale = head_size**-0.5

    expected_output, expected_state = _reference_sm12x_gdn_prefill(
        q,
        k,
        v,
        gate,
        beta,
        cu_seqlens,
        initial_state=None,
        scale=scale,
    )

    output_cuda = output.cuda()
    output_state_cuda = output_state.cuda()
    chunk_gated_delta_rule_sm12x(
        q=q.cuda(),
        k=k.cuda(),
        v=v.cuda(),
        gate=gate.cuda(),
        beta=beta.cuda(),
        output=output_cuda,
        cu_seqlens=cu_seqlens.cuda(),
        initial_state=None,
        output_state=output_state_cuda,
        scale=scale,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output_cuda.cpu().float(),
        expected_output,
        rtol=1e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        output_state_cuda.cpu(),
        expected_state,
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm12x_prefill_recurrent_kernel_writes_checkpoints():
    torch.manual_seed(2)
    total_tokens = 129
    num_heads = 1
    head_size = 128
    q = (
        torch.randn(total_tokens, num_heads, head_size, dtype=torch.float32) * 0.03
    ).to(torch.bfloat16)
    k = (
        torch.randn(total_tokens, num_heads, head_size, dtype=torch.float32) * 0.03
    ).to(torch.bfloat16)
    v = (
        torch.randn(total_tokens, num_heads, head_size, dtype=torch.float32) * 0.03
    ).to(torch.bfloat16)
    gate = torch.rand(total_tokens, num_heads, dtype=torch.float32) * 0.2 + 0.7
    beta = torch.rand_like(gate) * 0.25 + 0.1
    cu_seqlens = torch.tensor([0, 64, 129], dtype=torch.int32)
    cu_checkpoints = torch.tensor([0, 1, 2], dtype=torch.int32)
    output = torch.empty(total_tokens, num_heads, head_size, dtype=torch.bfloat16)
    output_state = torch.empty(2, num_heads, head_size, head_size)
    checkpoints = torch.empty(2, num_heads, head_size, head_size)
    scale = head_size**-0.5

    _, expected_state = _reference_sm12x_gdn_prefill(
        q,
        k,
        v,
        gate,
        beta,
        cu_seqlens,
        initial_state=None,
        scale=scale,
    )
    _, second_seq_first_checkpoint = _reference_sm12x_gdn_prefill(
        q[64:128],
        k[64:128],
        v[64:128],
        gate[64:128],
        beta[64:128],
        torch.tensor([0, 64], dtype=torch.int32),
        initial_state=None,
        scale=scale,
    )
    expected_checkpoints = torch.cat(
        [expected_state[:1], second_seq_first_checkpoint],
        dim=0,
    )

    checkpoints_cuda = checkpoints.cuda()
    chunk_gated_delta_rule_sm12x(
        q=q.cuda(),
        k=k.cuda(),
        v=v.cuda(),
        gate=gate.cuda(),
        beta=beta.cuda(),
        output=output.cuda(),
        cu_seqlens=cu_seqlens.cuda(),
        initial_state=None,
        output_state=output_state.cuda(),
        scale=scale,
        checkpoint_every_n_tokens=64,
        cu_checkpoints=cu_checkpoints.cuda(),
        output_checkpoints=checkpoints_cuda,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        checkpoints_cuda.cpu(),
        expected_checkpoints,
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm12x_public_prefill_entrypoint_matches_reference():
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires SM12x")

    torch.manual_seed(3)
    total_tokens = 33
    num_q_heads = 1
    num_v_heads = 2
    head_size = 128
    q = (
        torch.randn(total_tokens, num_q_heads, head_size, dtype=torch.float32) * 0.04
    ).to(torch.bfloat16)
    k = (
        torch.randn(total_tokens, num_q_heads, head_size, dtype=torch.float32) * 0.04
    ).to(torch.bfloat16)
    v = (
        torch.randn(total_tokens, num_v_heads, head_size, dtype=torch.float32) * 0.04
    ).to(torch.bfloat16)
    gate = torch.rand(total_tokens, num_v_heads, dtype=torch.float32) * 0.2 + 0.75
    beta = torch.rand_like(gate) * 0.25 + 0.1
    cu_seqlens = torch.tensor([0, 16, 33], dtype=torch.int64)
    scale = head_size**-0.5

    expected_output, expected_state = _reference_sm12x_gdn_prefill(
        q,
        k,
        v,
        gate,
        beta,
        cu_seqlens.to(torch.int32),
        initial_state=None,
        scale=scale,
    )

    output, output_state = chunk_gated_delta_rule(
        q.cuda(),
        k.cuda(),
        v.cuda(),
        gate.cuda(),
        beta.cuda(),
        scale,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu_seqlens.cuda(),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output.cpu().float(),
        expected_output,
        rtol=1e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        output_state.cpu(),
        expected_state,
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm12x_recurrent_prefill_cuda_graph_replay_matches_eager():
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires SM12x")

    torch.manual_seed(4)
    total_tokens = 33
    num_q_heads = 2
    num_v_heads = 1
    head_size = 128
    q = (
        torch.randn(total_tokens, num_q_heads, head_size, device="cuda") * 0.04
    ).to(torch.bfloat16)
    k = (
        torch.randn(total_tokens, num_v_heads, head_size, device="cuda") * 0.04
    ).to(torch.bfloat16)
    v = (
        torch.randn(total_tokens, num_v_heads, head_size, device="cuda") * 0.04
    ).to(torch.bfloat16)
    gate = torch.rand(total_tokens, num_q_heads, device="cuda") * 0.2 + 0.75
    beta = torch.rand_like(gate) * 0.25 + 0.1
    cu_seqlens = torch.tensor([0, 17, 33], device="cuda", dtype=torch.int32)
    scale = head_size**-0.5

    eager_output = torch.empty_like(q)
    eager_state = torch.empty(
        2,
        num_q_heads,
        head_size,
        head_size,
        device="cuda",
        dtype=torch.float32,
    )
    graph_output = torch.empty_like(eager_output)
    graph_state = torch.empty_like(eager_state)

    chunk_gated_delta_rule_sm12x(
        q=q,
        k=k,
        v=v,
        gate=gate,
        beta=beta,
        output=eager_output,
        cu_seqlens=cu_seqlens,
        initial_state=None,
        output_state=eager_state,
        scale=scale,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        chunk_gated_delta_rule_sm12x(
            q=q,
            k=k,
            v=v,
            gate=gate,
            beta=beta,
            output=graph_output,
            cu_seqlens=cu_seqlens,
            initial_state=None,
            output_state=graph_state,
            scale=scale,
        )

    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)
    torch.testing.assert_close(graph_state, eager_state, rtol=0, atol=0)


def test_sm12x_prefill_entrypoint_rejects_bad_metadata_before_stub():
    q, k, v, gate, beta, output, _, _ = _make_inputs()
    cu_seqlens = torch.tensor([0, 64, 63], dtype=torch.int32)
    state = torch.empty(2, 8, 128, 128, dtype=torch.float32)

    with pytest.raises(ValueError, match="monotonic"):
        chunk_gated_delta_rule_sm12x(
            q=q,
            k=k,
            v=v,
            gate=gate,
            beta=beta,
            output=output,
            cu_seqlens=cu_seqlens,
            initial_state=state,
            output_state=state,
            scale=128**-0.5,
        )
