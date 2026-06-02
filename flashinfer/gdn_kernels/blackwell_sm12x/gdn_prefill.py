"""
Gated Delta Net Chunked Prefill - Blackwell SM12x Adapter
=========================================================

This module owns the SM12x/GB10 GDN prefill entry point. It intentionally does
not reuse the SM100 implementation because that path depends on tcgen05/TMEM
instructions that are not the target architecture contract for GB10.

State layout: ``[N, H, V, K]``.
"""

import functools
from dataclasses import dataclass
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float16, Float32, Int32
from cutlass.cute.runtime import from_dlpack


@dataclass(frozen=True)
class Sm12xGdnPrefillConfig:
    total_tokens: int
    num_seqs: int
    num_q_heads: int
    num_k_heads: int
    num_v_heads: int
    num_o_heads: int
    head_size: int
    is_gqa: bool
    is_gva: bool
    use_initial_state: bool
    store_final_state: bool
    enable_checkpoints: bool


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _sm12x_upper_bound_chunks(
    num_seqs: int,
    total_tokens: int,
    chunk_size: int = 64,
) -> int:
    """Return the tight chunk upper bound when every sequence is non-empty."""
    if num_seqs <= 0:
        raise ValueError(f"num_seqs must be positive, got {num_seqs}")
    if total_tokens < num_seqs:
        raise ValueError(
            "total_tokens must be at least num_seqs for positive sequence "
            f"lengths, got total_tokens={total_tokens}, num_seqs={num_seqs}"
        )
    if chunk_size != 64:
        raise ValueError(f"SM12x GDN prefill requires chunk_size=64, got {chunk_size}")
    return (num_seqs - 1) + _ceil_div(total_tokens - (num_seqs - 1), chunk_size)


class Sm12xGdnPrefillMetadataKernel:
    """Build chunk metadata on device for staged SM12x GDN prefill kernels."""

    def __init__(self, chunk_size: int) -> None:
        if chunk_size != 64:
            raise ValueError(
                f"SM12x GDN prefill requires chunk_size=64, got {chunk_size}"
            )
        self.chunk_size = chunk_size
        self.num_warps = 8

    @cute.jit
    def __call__(
        self,
        cu_seqlens: cute.Tensor,
        chunk_indices: cute.Tensor,
        chunk_offsets: cute.Tensor,
        stream: cuda.CUstream,
    ):
        block = (self.num_warps * 32, 1, 1)
        self.kernel(
            cu_seqlens,
            chunk_indices,
            chunk_offsets,
        ).launch(grid=(1, 1, 1), block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        cu_seqlens: cute.Tensor,
        chunk_indices: cute.Tensor,
        chunk_offsets: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32

        num_seqs = cu_seqlens.shape[0] - 1
        num_warps = self.num_warps
        tb_size = num_warps * 32

        if tid == 0:
            chunk_offsets[0] = 0

        coarsen = cute.ceil_div(num_seqs, tb_size)
        seq_start = tid * coarsen
        num_iters = cutlass.min(seq_start + coarsen, num_seqs) - seq_start

        thread_sum = Int32(0)
        for i in range(num_iters):
            seq_id = seq_start + i
            seqlen = cu_seqlens[seq_id + 1] - cu_seqlens[seq_id]
            thread_sum += cute.ceil_div(seqlen, self.chunk_size)

        cu_num_chunks = thread_sum
        for i in cutlass.range_constexpr(5):
            offset = cutlass.const_expr(1 << i)
            lower = cute.arch.shuffle_sync_up(
                cu_num_chunks, offset=offset, mask_and_clamp=0
            )
            if lane_id >= offset:
                cu_num_chunks += lower

        smem = cutlass.utils.SmemAllocator()
        warp_num_chunks = smem.allocate_array(Int32, num_warps)
        if lane_id == 31:
            warp_num_chunks[warp_id] = cu_num_chunks
        cute.arch.sync_threads()

        for i in cutlass.range_constexpr(1, num_warps):
            if warp_id >= i:
                cu_num_chunks += warp_num_chunks[i - 1]

        chunk_start = cu_num_chunks - thread_sum

        for i in range(num_iters):
            seq_id = seq_start + i
            seqlen = cu_seqlens[seq_id + 1] - cu_seqlens[seq_id]
            num_chunks = cute.ceil_div(seqlen, self.chunk_size)
            chunk_end = chunk_start + num_chunks
            chunk_offsets[seq_id + 1] = chunk_end

            for chunk_id in range(num_chunks):
                chunk_indices[chunk_start + chunk_id, 0] = seq_id
                chunk_indices[chunk_start + chunk_id, 1] = chunk_id

            chunk_start = chunk_end


class Sm12xGdnPrefillGateCumsumKernel:
    """Build per-sequence cumulative log2(gate) values on SM12x."""

    @cute.jit
    def __call__(
        self,
        gate: cute.Tensor,
        cu_seqlens: cute.Tensor,
        g_cu: cute.Tensor,
        stream: cuda.CUstream,
    ):
        num_seqs = cu_seqlens.shape[0] - 1
        num_heads = gate.shape[1]
        self.kernel(
            gate,
            cu_seqlens,
            g_cu,
        ).launch(grid=(num_heads, num_seqs, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        gate: cute.Tensor,
        cu_seqlens: cute.Tensor,
        g_cu: cute.Tensor,
    ):
        head_id, seq_id, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()

        if tid == 0:
            start = cu_seqlens[seq_id]
            end = cu_seqlens[seq_id + 1]
            acc = Float32(0.0)
            for offset in range(end - start):
                token = start + offset
                acc += cute.math.log2(gate[token, head_id] + 1e-10, fastmath=True)
                g_cu[token, head_id] = acc


@cute.jit
def _warp_reduce_sum(
    local_value: Float32,
) -> Float32:
    """Reduce a scalar across one warp and return the sum to every lane."""
    warp_sum = local_value
    for offset in [16, 8, 4, 2, 1]:
        warp_sum += cute.arch.shuffle_sync_bfly(
            warp_sum,
            offset=offset,
            mask=-1,
            mask_and_clamp=31,
        )
    return warp_sum


class Sm12xGdnPrefillRecurrentKernel:
    """Simple SM12x-safe recurrent GDN prefill kernel.

    This is a correctness-first kernel. It avoids SM100 tcgen05/TMEM and uses
    one warp/CTA per ``(sequence, output head, value row)``. Each lane owns four
    K elements, which keeps the recurrent state in registers and reduces the
    two per-token dot products with warp shuffles only.
    """

    def __init__(
        self,
        io_dtype: type[cutlass.Numeric],
        state_dtype: type[cutlass.Numeric],
        is_gqa: bool,
        use_initial_state: bool,
        store_final_state: bool,
        enable_checkpoints: bool,
    ) -> None:
        self.io_dtype = io_dtype
        self.state_dtype = state_dtype
        self.is_gqa = is_gqa
        self.use_initial_state = use_initial_state
        self.store_final_state = store_final_state
        self.enable_checkpoints = enable_checkpoints
        self.head_size = 128
        self.keys_per_lane = 4

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        gate: cute.Tensor,
        beta: cute.Tensor,
        output: cute.Tensor,
        cu_seqlens: cute.Tensor,
        initial_state: Optional[cute.Tensor],
        output_state: Optional[cute.Tensor],
        output_checkpoints: Optional[cute.Tensor],
        cu_checkpoints: Optional[cute.Tensor],
        checkpoint_every_n_tokens: Int32,
        scale: Float32,
        stream: cuda.CUstream,
    ):
        num_seqs = cu_seqlens.shape[0] - 1
        num_o_heads = output.shape[1]
        self.kernel(
            q,
            k,
            v,
            gate,
            beta,
            output,
            cu_seqlens,
            initial_state,
            output_state,
            output_checkpoints,
            cu_checkpoints,
            checkpoint_every_n_tokens,
            scale,
        ).launch(
            grid=(num_seqs, num_o_heads, self.head_size),
            block=(32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        gate: cute.Tensor,
        beta: cute.Tensor,
        output: cute.Tensor,
        cu_seqlens: cute.Tensor,
        initial_state: Optional[cute.Tensor],
        output_state: Optional[cute.Tensor],
        output_checkpoints: Optional[cute.Tensor],
        cu_checkpoints: Optional[cute.Tensor],
        checkpoint_every_n_tokens: Int32,
        scale: Float32,
    ):
        seq_id, out_head, value_idx = cute.arch.block_idx()
        lane_id, _, _ = cute.arch.thread_idx()

        if cutlass.const_expr(self.is_gqa):
            group_size = q.shape[1] // v.shape[1]
            q_head = out_head
            k_head = out_head // group_size
            v_head = k_head
        else:
            group_size = v.shape[1] // q.shape[1]
            q_head = out_head // group_size
            k_head = q_head
            v_head = out_head

        k_indices = cute.make_rmem_tensor((self.keys_per_lane,), Int32)
        state_vals = cute.make_rmem_tensor((self.keys_per_lane,), Float32)
        for slot in cutlass.range_constexpr(self.keys_per_lane):
            k_indices[slot] = lane_id + cutlass.const_expr(slot * 32)
            state_vals[slot] = Float32(0.0)

        if cutlass.const_expr(self.use_initial_state):
            assert initial_state is not None
            for slot in cutlass.range_constexpr(self.keys_per_lane):
                state_vals[slot] = Float32(
                    initial_state[seq_id, out_head, value_idx, k_indices[slot]]
                )

        start = cu_seqlens[seq_id]
        end = cu_seqlens[seq_id + 1]
        for offset in range(end - start):
            token = start + offset
            gate_val = gate[token, out_head]
            beta_val = beta[token, out_head]

            projected_local = Float32(0.0)
            k_vals = cute.make_rmem_tensor((self.keys_per_lane,), Float32)
            for slot in cutlass.range_constexpr(self.keys_per_lane):
                k_vals[slot] = Float32(k[token, k_head, k_indices[slot]])
                state_vals[slot] *= gate_val
                projected_local += state_vals[slot] * k_vals[slot]

            projected = _warp_reduce_sum(projected_local)
            value = Float32(v[token, v_head, value_idx])
            delta = (value - projected) * beta_val

            for slot in cutlass.range_constexpr(self.keys_per_lane):
                state_vals[slot] += delta * k_vals[slot]

            output_local = Float32(0.0)
            for slot in cutlass.range_constexpr(self.keys_per_lane):
                q_val = Float32(q[token, q_head, k_indices[slot]]) * scale
                output_local += state_vals[slot] * q_val

            output_value = _warp_reduce_sum(output_local)

            if lane_id == 0:
                if cutlass.const_expr(self.io_dtype is BFloat16):
                    output[token, out_head, value_idx] = BFloat16(output_value)
                else:
                    output[token, out_head, value_idx] = Float16(output_value)

            if cutlass.const_expr(self.enable_checkpoints):
                assert output_checkpoints is not None
                assert cu_checkpoints is not None
                token_count = offset + 1
                if token_count % checkpoint_every_n_tokens == 0:
                    checkpoint_idx = (
                        cu_checkpoints[seq_id]
                        + token_count // checkpoint_every_n_tokens
                        - 1
                    )
                    if cutlass.const_expr(self.state_dtype is BFloat16):
                        for slot in cutlass.range_constexpr(self.keys_per_lane):
                            output_checkpoints[
                                checkpoint_idx,
                                out_head,
                                value_idx,
                                k_indices[slot],
                            ] = BFloat16(state_vals[slot])
                    else:
                        for slot in cutlass.range_constexpr(self.keys_per_lane):
                            output_checkpoints[
                                checkpoint_idx,
                                out_head,
                                value_idx,
                                k_indices[slot],
                            ] = state_vals[slot]

        if cutlass.const_expr(self.store_final_state):
            assert output_state is not None
            if cutlass.const_expr(self.state_dtype is BFloat16):
                for slot in cutlass.range_constexpr(self.keys_per_lane):
                    output_state[seq_id, out_head, value_idx, k_indices[slot]] = (
                        BFloat16(state_vals[slot])
                    )
            else:
                for slot in cutlass.range_constexpr(self.keys_per_lane):
                    output_state[seq_id, out_head, value_idx, k_indices[slot]] = (
                        state_vals[slot]
                    )


@functools.cache
def _compile_sm12x_metadata_kernel(chunk_size: int):
    cu_entries = cute.sym_int()
    upper_bound_chunks = cute.sym_int()

    cu_seqlens = cute.runtime.make_fake_compact_tensor(
        Int32,
        (cu_entries,),
        assumed_align=4,
    )
    chunk_indices = cute.runtime.make_fake_tensor(
        Int32,
        (upper_bound_chunks, 2),
        (2, 1),
        assumed_align=4,
    )
    chunk_offsets = cute.runtime.make_fake_compact_tensor(
        Int32,
        (cu_entries,),
        assumed_align=4,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        Sm12xGdnPrefillMetadataKernel(chunk_size),
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        stream,
        options="--enable-tvm-ffi --opt-level 2",
    )


@functools.cache
def _compile_sm12x_gate_cumsum_kernel():
    total_tokens = cute.sym_int()
    num_heads = cute.sym_int()
    cu_entries = cute.sym_int()

    gate = cute.runtime.make_fake_tensor(
        Float32,
        (total_tokens, num_heads),
        (num_heads, 1),
        assumed_align=4,
    )
    cu_seqlens = cute.runtime.make_fake_compact_tensor(
        Int32,
        (cu_entries,),
        assumed_align=4,
    )
    g_cu = cute.runtime.make_fake_tensor(
        Float32,
        (total_tokens, num_heads),
        (num_heads, 1),
        assumed_align=4,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        Sm12xGdnPrefillGateCumsumKernel(),
        gate,
        cu_seqlens,
        g_cu,
        stream,
        options="--enable-tvm-ffi --opt-level 2",
    )


@functools.cache
def _get_sm12x_recurrent_cache(
    io_dtype_str: str,
    state_dtype_str: str,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    num_o_heads: int,
    is_gqa: bool,
    use_initial_state: bool,
    store_final_state: bool,
    enable_checkpoints: bool,
) -> dict:
    return {}


def _cutlass_io_dtype(torch_dtype: torch.dtype) -> type[cutlass.Numeric]:
    if torch_dtype == torch.bfloat16:
        return BFloat16
    if torch_dtype == torch.float16:
        return Float16
    raise ValueError(f"Unsupported IO dtype {torch_dtype}")


def _cutlass_state_dtype(torch_dtype: torch.dtype) -> type[cutlass.Numeric]:
    if torch_dtype == torch.float32:
        return Float32
    if torch_dtype == torch.bfloat16:
        return BFloat16
    raise ValueError(f"Unsupported state dtype {torch_dtype}")


def _mark_token_dim_dynamic(tensor: cute.Tensor, rank: int) -> cute.Tensor:
    tensor.mark_compact_shape_dynamic(
        mode=0,
        stride_order=tuple(range(rank)),
        divisibility=1,
    )
    return tensor


def _build_sm12x_chunk_metadata(
    cu_seqlens: torch.Tensor,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build exact per-chunk metadata for the staged SM12x prefill kernels.

    Returns:
        ``chunk_indices`` with shape ``[num_chunks, 2]`` where each row is
        ``[seq_id, chunk_id_within_seq]``, and ``chunk_offsets`` with shape
        ``[num_seqs + 1]``.
    """
    if chunk_size != 64:
        raise ValueError(f"SM12x GDN prefill requires chunk_size=64, got {chunk_size}")
    if cu_seqlens.ndim != 1:
        raise ValueError(f"cu_seqlens must be 1D, got {cu_seqlens.ndim}D")
    if cu_seqlens.dtype != torch.int32:
        raise ValueError(f"cu_seqlens must be int32, got {cu_seqlens.dtype}")

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if bool((lengths < 0).any().item()):
        raise ValueError("cu_seqlens must be monotonic non-decreasing")
    if bool((lengths == 0).any().item()):
        raise ValueError("SM12x GDN prefill does not support empty sequences")

    num_seqs = int(lengths.numel())
    chunk_counts = torch.div(
        lengths + chunk_size - 1,
        chunk_size,
        rounding_mode="floor",
    ).to(torch.int32)
    chunk_offsets = torch.empty(
        num_seqs + 1,
        dtype=torch.int32,
        device=cu_seqlens.device,
    )
    chunk_offsets[0] = 0
    chunk_offsets[1:] = torch.cumsum(chunk_counts, dim=0)

    total_chunks = int(chunk_offsets[-1].item())
    chunk_indices = torch.empty(
        total_chunks,
        2,
        dtype=torch.int32,
        device=cu_seqlens.device,
    )

    write_offset = 0
    for seq_id in range(num_seqs):
        count = int(chunk_counts[seq_id].item())
        chunk_indices[write_offset : write_offset + count, 0] = seq_id
        chunk_indices[write_offset : write_offset + count, 1] = torch.arange(
            count,
            dtype=torch.int32,
            device=cu_seqlens.device,
        )
        write_offset += count

    return chunk_indices, chunk_offsets


def _validate_sm12x_gate_cumsum_inputs(
    gate: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[int, int, int]:
    if gate.ndim != 2:
        raise ValueError(f"gate must be 2D, got {gate.ndim}D")
    if gate.dtype != torch.float32:
        raise ValueError(f"gate must be float32, got {gate.dtype}")
    if not gate.is_contiguous():
        raise ValueError("gate must be contiguous")
    if cu_seqlens.ndim != 1:
        raise ValueError(f"cu_seqlens must be 1D, got {cu_seqlens.ndim}D")
    if cu_seqlens.dtype != torch.int32:
        raise ValueError(f"cu_seqlens must be int32, got {cu_seqlens.dtype}")
    if not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous")
    if gate.device != cu_seqlens.device:
        raise ValueError(
            "gate and cu_seqlens must be on the same device, got "
            f"{gate.device} and {cu_seqlens.device}"
        )

    total_tokens = gate.size(0)
    num_heads = gate.size(1)
    num_seqs = cu_seqlens.size(0) - 1
    if num_seqs <= 0:
        raise ValueError("cu_seqlens must contain at least one sequence")
    if int(cu_seqlens[0].item()) != 0:
        raise ValueError("cu_seqlens must start at 0")

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if bool((lengths < 0).any().item()):
        raise ValueError("cu_seqlens must be monotonic non-decreasing")
    if bool((lengths == 0).any().item()):
        raise ValueError("SM12x GDN prefill does not support empty sequences")
    if int(cu_seqlens[-1].item()) != total_tokens:
        raise ValueError(
            "cu_seqlens[-1] must match total_tokens, got "
            f"{int(cu_seqlens[-1].item())} and {total_tokens}"
        )

    return total_tokens, num_heads, num_seqs


def _build_sm12x_gate_cumsum(
    gate: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Build per-sequence cumulative log2 gates on the current device."""
    _, _, num_seqs = _validate_sm12x_gate_cumsum_inputs(gate, cu_seqlens)

    g_cu = torch.empty_like(gate)
    for seq_id in range(num_seqs):
        start = int(cu_seqlens[seq_id].item())
        end = int(cu_seqlens[seq_id + 1].item())
        g_cu[start:end] = torch.log2(gate[start:end] + 1e-10).cumsum(dim=0)
    return g_cu


def _build_sm12x_gate_cumsum_device(
    gate: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Build per-sequence cumulative log2 gates on CUDA with a CuTe kernel."""
    _validate_sm12x_gate_cumsum_inputs(gate, cu_seqlens)
    if not gate.is_cuda:
        raise ValueError("gate must be a CUDA tensor for device gate cumsum")

    g_cu = torch.empty_like(gate)
    _compile_sm12x_gate_cumsum_kernel()(
        gate,
        cu_seqlens,
        g_cu,
    )
    return g_cu


def _prepare_sm12x_gate_cumsum(
    gate: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Prepare cumulative log2 gate values using CUDA when possible."""
    if gate.is_cuda:
        return _build_sm12x_gate_cumsum_device(gate, cu_seqlens)
    return _build_sm12x_gate_cumsum(gate, cu_seqlens)


def _build_sm12x_chunk_metadata_device(
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build exact per-chunk metadata on CUDA using the CuTe metadata kernel."""
    if not cu_seqlens.is_cuda:
        raise ValueError("cu_seqlens must be a CUDA tensor for device metadata")
    if cu_seqlens.dtype != torch.int32:
        raise ValueError(f"cu_seqlens must be int32, got {cu_seqlens.dtype}")

    num_seqs = cu_seqlens.numel() - 1
    upper_bound_chunks = _sm12x_upper_bound_chunks(
        num_seqs,
        total_tokens,
        chunk_size,
    )
    chunk_indices = torch.empty(
        upper_bound_chunks,
        2,
        dtype=torch.int32,
        device=cu_seqlens.device,
    )
    chunk_offsets = torch.empty(
        num_seqs + 1,
        dtype=torch.int32,
        device=cu_seqlens.device,
    )

    _compile_sm12x_metadata_kernel(chunk_size)(
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
    )
    return chunk_indices, chunk_offsets


def _prepare_sm12x_chunk_metadata(
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare chunk metadata using the CUDA kernel when possible."""
    if cu_seqlens.is_cuda:
        return _build_sm12x_chunk_metadata_device(
            cu_seqlens,
            total_tokens,
            chunk_size,
        )
    return _build_sm12x_chunk_metadata(cu_seqlens, chunk_size)


def _expect_3d(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be 3D, got {tensor.ndim}D")


def _expect_shape(
    name: str,
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...],
) -> None:
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{name} shape mismatch: expected {expected_shape}, "
            f"got {tuple(tensor.shape)}"
        )


def _validate_state_tensor(
    name: str,
    tensor: torch.Tensor,
    expected_shape: tuple[int, int, int, int],
) -> None:
    _expect_shape(name, tensor, expected_shape)
    if tensor.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"{name} must be float32 or bfloat16, got {tensor.dtype}")


def _validate_sm12x_gdn_prefill_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    output_state: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
    cu_checkpoints: Optional[torch.Tensor],
    output_checkpoints: Optional[torch.Tensor],
) -> Sm12xGdnPrefillConfig:
    """Validate the initial SM12x GDN prefill kernel contract.

    The first GB10-native implementation is intentionally narrow: Qwen-style
    ``head_size=128`` with FP16/BF16 IO and FP32/BF16 recurrent state.
    """
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        _expect_3d(name, tensor)

    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"q must be float16 or bfloat16, got {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(
            f"q, k, and v must have the same dtype, got {q.dtype}, {k.dtype}, {v.dtype}"
        )

    total_tokens = q.size(0)
    num_q_heads = q.size(1)
    num_k_heads = k.size(1)
    num_v_heads = v.size(1)
    head_size = q.size(2)

    _expect_shape("k", k, (total_tokens, num_k_heads, head_size))
    _expect_shape("v", v, (total_tokens, num_v_heads, head_size))

    if head_size != 128:
        raise ValueError(f"SM12x GDN prefill requires head_size=128, got {head_size}")

    if num_q_heads >= num_v_heads:
        if num_k_heads != num_v_heads or num_q_heads % num_v_heads != 0:
            raise ValueError(
                "GQA layout requires num_q_heads to be a multiple of "
                "num_v_heads and num_k_heads == num_v_heads"
            )
        is_gqa = num_q_heads > num_v_heads
        is_gva = False
    else:
        if num_k_heads != num_q_heads or num_v_heads % num_q_heads != 0:
            raise ValueError(
                "GVA layout requires num_v_heads to be a multiple of "
                "num_q_heads and num_k_heads == num_q_heads"
            )
        is_gqa = False
        is_gva = True

    num_o_heads = max(num_q_heads, num_v_heads)
    _expect_shape("gate", gate, (total_tokens, num_o_heads))
    _expect_shape("beta", beta, (total_tokens, num_o_heads))
    if gate.dtype != torch.float32:
        raise ValueError(f"gate must be float32, got {gate.dtype}")
    if beta.dtype != torch.float32:
        raise ValueError(f"beta must be float32, got {beta.dtype}")
    _expect_shape("output", output, (total_tokens, num_o_heads, head_size))
    if output.dtype != q.dtype:
        raise ValueError(f"output dtype must be {q.dtype}, got {output.dtype}")

    if cu_seqlens.ndim != 1:
        raise ValueError(f"cu_seqlens must be 1D, got {cu_seqlens.ndim}D")
    if cu_seqlens.dtype != torch.int32:
        raise ValueError(f"cu_seqlens must be int32, got {cu_seqlens.dtype}")
    num_seqs = cu_seqlens.size(0) - 1
    if num_seqs <= 0:
        raise ValueError("cu_seqlens must contain at least one sequence")
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if bool((lengths < 0).any().item()):
        raise ValueError("cu_seqlens must be monotonic non-decreasing")
    if int(cu_seqlens[-1].item()) != total_tokens:
        raise ValueError(
            "cu_seqlens[-1] must match total_tokens, got "
            f"{int(cu_seqlens[-1].item())} and {total_tokens}"
        )

    state_shape = (num_seqs, num_o_heads, head_size, head_size)
    if initial_state is not None:
        _validate_state_tensor("initial_state", initial_state, state_shape)
    if output_state is not None:
        _validate_state_tensor("output_state", output_state, state_shape)

    enable_checkpoints = checkpoint_every_n_tokens > 0
    if enable_checkpoints:
        if checkpoint_every_n_tokens % 64 != 0:
            raise ValueError(
                "checkpoint_every_n_tokens must be a multiple of 64, "
                f"got {checkpoint_every_n_tokens}"
            )
        if cu_checkpoints is None or output_checkpoints is None:
            raise ValueError(
                "cu_checkpoints and output_checkpoints must be provided when "
                "checkpoint_every_n_tokens > 0"
            )
        if cu_checkpoints.dtype != torch.int32:
            raise ValueError(
                f"cu_checkpoints must be int32, got {cu_checkpoints.dtype}"
            )
        if cu_checkpoints.ndim != 1 or cu_checkpoints.size(0) != num_seqs + 1:
            raise ValueError(
                f"cu_checkpoints must be 1D with {num_seqs + 1} elements, "
                f"got shape {tuple(cu_checkpoints.shape)}"
            )
        if output_checkpoints.ndim != 4:
            raise ValueError(
                "output_checkpoints must be 4D "
                "[total_checkpoints, num_o_heads, head_size, head_size]"
            )
        if tuple(output_checkpoints.shape[1:]) != state_shape[1:]:
            raise ValueError(
                "output_checkpoints shape mismatch: expected "
                f"[*, {num_o_heads}, {head_size}, {head_size}], "
                f"got {tuple(output_checkpoints.shape)}"
            )
        if output_checkpoints.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(
                "output_checkpoints must be float32 or bfloat16, "
                f"got {output_checkpoints.dtype}"
            )
    elif cu_checkpoints is not None or output_checkpoints is not None:
        raise ValueError(
            "cu_checkpoints and output_checkpoints must be None when "
            "checkpoint_every_n_tokens == 0"
        )

    return Sm12xGdnPrefillConfig(
        total_tokens=total_tokens,
        num_seqs=num_seqs,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        num_o_heads=num_o_heads,
        head_size=head_size,
        is_gqa=is_gqa,
        is_gva=is_gva,
        use_initial_state=initial_state is not None,
        store_final_state=output_state is not None,
        enable_checkpoints=enable_checkpoints,
    )


def _is_cuda_graph_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _validate_sm12x_gdn_prefill_inputs_for_capture(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    output_state: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
    cu_checkpoints: Optional[torch.Tensor],
    output_checkpoints: Optional[torch.Tensor],
) -> Sm12xGdnPrefillConfig:
    """Validate the graph-capture-safe parts of the SM12x GDN contract.

    Full value checks on ``cu_seqlens`` require host reads from device tensors,
    which invalidate CUDA graph capture. Callers must warm up this exact shape
    through the normal path before capture; this helper keeps the static shape,
    dtype, and buffer-contract checks during capture.
    """
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        _expect_3d(name, tensor)

    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"q must be float16 or bfloat16, got {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(
            f"q, k, and v must have the same dtype, got {q.dtype}, {k.dtype}, {v.dtype}"
        )

    total_tokens = q.size(0)
    num_q_heads = q.size(1)
    num_k_heads = k.size(1)
    num_v_heads = v.size(1)
    head_size = q.size(2)
    if head_size != 128:
        raise ValueError(f"SM12x GDN prefill requires head_size=128, got {head_size}")
    _expect_shape("k", k, (total_tokens, num_k_heads, head_size))
    _expect_shape("v", v, (total_tokens, num_v_heads, head_size))

    if num_q_heads >= num_v_heads:
        if num_k_heads != num_v_heads or num_q_heads % num_v_heads != 0:
            raise ValueError(
                "GQA layout requires num_q_heads to be a multiple of "
                "num_v_heads and num_k_heads == num_v_heads"
            )
        is_gqa = num_q_heads > num_v_heads
        is_gva = False
    else:
        if num_k_heads != num_q_heads or num_v_heads % num_q_heads != 0:
            raise ValueError(
                "GVA layout requires num_v_heads to be a multiple of "
                "num_q_heads and num_k_heads == num_q_heads"
            )
        is_gqa = False
        is_gva = True

    num_o_heads = max(num_q_heads, num_v_heads)
    _expect_shape("gate", gate, (total_tokens, num_o_heads))
    _expect_shape("beta", beta, (total_tokens, num_o_heads))
    if gate.dtype != torch.float32:
        raise ValueError(f"gate must be float32, got {gate.dtype}")
    if beta.dtype != torch.float32:
        raise ValueError(f"beta must be float32, got {beta.dtype}")
    _expect_shape("output", output, (total_tokens, num_o_heads, head_size))
    if output.dtype != q.dtype:
        raise ValueError(f"output dtype must be {q.dtype}, got {output.dtype}")

    if cu_seqlens.ndim != 1:
        raise ValueError(f"cu_seqlens must be 1D, got {cu_seqlens.ndim}D")
    if cu_seqlens.dtype != torch.int32:
        raise ValueError(f"cu_seqlens must be int32, got {cu_seqlens.dtype}")
    num_seqs = cu_seqlens.size(0) - 1
    if num_seqs <= 0:
        raise ValueError("cu_seqlens must contain at least one sequence")

    state_shape = (num_seqs, num_o_heads, head_size, head_size)
    if initial_state is not None:
        _validate_state_tensor("initial_state", initial_state, state_shape)
    if output_state is not None:
        _validate_state_tensor("output_state", output_state, state_shape)

    enable_checkpoints = checkpoint_every_n_tokens > 0
    if enable_checkpoints:
        if checkpoint_every_n_tokens % 64 != 0:
            raise ValueError(
                "checkpoint_every_n_tokens must be a multiple of 64, "
                f"got {checkpoint_every_n_tokens}"
            )
        if cu_checkpoints is None or output_checkpoints is None:
            raise ValueError(
                "cu_checkpoints and output_checkpoints must be provided when "
                "checkpoint_every_n_tokens > 0"
            )
        if cu_checkpoints.dtype != torch.int32:
            raise ValueError(
                f"cu_checkpoints must be int32, got {cu_checkpoints.dtype}"
            )
        if cu_checkpoints.ndim != 1 or cu_checkpoints.size(0) != num_seqs + 1:
            raise ValueError(
                f"cu_checkpoints must be 1D with {num_seqs + 1} elements, "
                f"got shape {tuple(cu_checkpoints.shape)}"
            )
        if output_checkpoints.ndim != 4:
            raise ValueError(
                "output_checkpoints must be 4D "
                "[total_checkpoints, num_o_heads, head_size, head_size]"
            )
        if tuple(output_checkpoints.shape[1:]) != state_shape[1:]:
            raise ValueError(
                "output_checkpoints shape mismatch: expected "
                f"[*, {num_o_heads}, {head_size}, {head_size}], "
                f"got {tuple(output_checkpoints.shape)}"
            )
        if output_checkpoints.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(
                "output_checkpoints must be float32 or bfloat16, "
                f"got {output_checkpoints.dtype}"
            )
    elif cu_checkpoints is not None or output_checkpoints is not None:
        raise ValueError(
            "cu_checkpoints and output_checkpoints must be None when "
            "checkpoint_every_n_tokens == 0"
        )

    return Sm12xGdnPrefillConfig(
        total_tokens=total_tokens,
        num_seqs=num_seqs,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        num_o_heads=num_o_heads,
        head_size=head_size,
        is_gqa=is_gqa,
        is_gva=is_gva,
        use_initial_state=initial_state is not None,
        store_final_state=output_state is not None,
        enable_checkpoints=enable_checkpoints,
    )


def _compile_or_get_sm12x_recurrent_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    output_state: Optional[torch.Tensor],
    output_checkpoints: Optional[torch.Tensor],
    cu_checkpoints: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
    scale: float,
    config: Sm12xGdnPrefillConfig,
):
    state_torch_dtype = torch.float32
    if initial_state is not None:
        state_torch_dtype = initial_state.dtype
    elif output_state is not None:
        state_torch_dtype = output_state.dtype
    elif output_checkpoints is not None:
        state_torch_dtype = output_checkpoints.dtype

    cache = _get_sm12x_recurrent_cache(
        str(q.dtype),
        str(state_torch_dtype),
        config.num_q_heads,
        config.num_k_heads,
        config.num_v_heads,
        config.num_o_heads,
        config.is_gqa,
        config.use_initial_state,
        config.store_final_state,
        config.enable_checkpoints,
    )
    if "compiled" in cache:
        return cache["compiled"]

    recurrent = Sm12xGdnPrefillRecurrentKernel(
        io_dtype=_cutlass_io_dtype(q.dtype),
        state_dtype=_cutlass_state_dtype(state_torch_dtype),
        is_gqa=config.is_gqa,
        use_initial_state=config.use_initial_state,
        store_final_state=config.store_final_state,
        enable_checkpoints=config.enable_checkpoints,
    )

    q_cute = _mark_token_dim_dynamic(from_dlpack(q, assumed_align=16), rank=3)
    k_cute = _mark_token_dim_dynamic(from_dlpack(k, assumed_align=16), rank=3)
    v_cute = _mark_token_dim_dynamic(from_dlpack(v, assumed_align=16), rank=3)
    gate_cute = _mark_token_dim_dynamic(from_dlpack(gate, assumed_align=16), rank=2)
    beta_cute = _mark_token_dim_dynamic(from_dlpack(beta, assumed_align=16), rank=2)
    output_cute = _mark_token_dim_dynamic(
        from_dlpack(output, assumed_align=16),
        rank=3,
    )
    cu_seqlens_cute = from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic()

    initial_state_cute = None
    if initial_state is not None:
        initial_state_cute = from_dlpack(initial_state, assumed_align=16)
        initial_state_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
            mode=0,
            stride_order=(0, 1, 2, 3),
            divisibility=1,
        )

    output_state_cute = None
    if output_state is not None:
        output_state_cute = from_dlpack(output_state, assumed_align=16)
        output_state_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
            mode=0,
            stride_order=(0, 1, 2, 3),
            divisibility=1,
        )

    output_checkpoints_cute = None
    cu_checkpoints_cute = None
    if output_checkpoints is not None:
        output_checkpoints_cute = from_dlpack(output_checkpoints, assumed_align=16)
        output_checkpoints_cute.mark_layout_dynamic()
    if cu_checkpoints is not None:
        cu_checkpoints_cute = from_dlpack(
            cu_checkpoints,
            assumed_align=4,
        ).mark_layout_dynamic()

    stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)
    compiled = cute.compile(
        recurrent,
        q_cute,
        k_cute,
        v_cute,
        gate_cute,
        beta_cute,
        output_cute,
        cu_seqlens_cute,
        initial_state_cute,
        output_state_cute,
        output_checkpoints_cute,
        cu_checkpoints_cute,
        checkpoint_every_n_tokens,
        scale,
        stream,
        options="--enable-tvm-ffi --opt-level 2",
    )
    cache["compiled"] = compiled
    return compiled


def _run_sm12x_recurrent_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    output_state: Optional[torch.Tensor],
    scale: float,
    checkpoint_every_n_tokens: int,
    cu_checkpoints: Optional[torch.Tensor],
    output_checkpoints: Optional[torch.Tensor],
    config: Sm12xGdnPrefillConfig,
) -> None:
    if not q.is_cuda:
        raise NotImplementedError("SM12x GDN prefill requires CUDA tensors")

    compiled = _compile_or_get_sm12x_recurrent_kernel(
        q=q,
        k=k,
        v=v,
        gate=gate,
        beta=beta,
        output=output,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        output_state=output_state,
        output_checkpoints=output_checkpoints,
        cu_checkpoints=cu_checkpoints,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        scale=scale,
        config=config,
    )
    stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)
    compiled(
        q,
        k,
        v,
        gate,
        beta,
        output,
        cu_seqlens,
        initial_state,
        output_state,
        output_checkpoints,
        cu_checkpoints,
        checkpoint_every_n_tokens,
        scale,
        stream,
    )


def chunk_gated_delta_rule_sm12x(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    output_state: Optional[torch.Tensor],
    scale: float,
    checkpoint_every_n_tokens: int = 0,
    cu_checkpoints: Optional[torch.Tensor] = None,
    output_checkpoints: Optional[torch.Tensor] = None,
) -> None:
    """Execute the SM12x chunked GDN prefill kernel.

    This is the stable Python entry point for the upcoming GB10-native kernel.
    The implementation must use an SM12x-safe kernel strategy, such as
    warp-level MMA with SMEM/RMEM staging, rather than SM100 tcgen05/TMEM.
    """
    validator = (
        _validate_sm12x_gdn_prefill_inputs_for_capture
        if _is_cuda_graph_capturing()
        else _validate_sm12x_gdn_prefill_inputs
    )
    config = validator(
        q,
        k,
        v,
        gate,
        beta,
        output,
        cu_seqlens,
        initial_state,
        output_state,
        checkpoint_every_n_tokens,
        cu_checkpoints,
        output_checkpoints,
    )
    _run_sm12x_recurrent_kernel(
        q=q,
        k=k,
        v=v,
        gate=gate,
        beta=beta,
        output=output,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        output_state=output_state,
        scale=scale,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        cu_checkpoints=cu_checkpoints,
        output_checkpoints=output_checkpoints,
        config=config,
    )
