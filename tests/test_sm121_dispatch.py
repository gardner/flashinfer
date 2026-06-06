from types import SimpleNamespace

import pytest
import torch
from packaging.version import Version


def test_mm_fp4_auto_prefers_b12x_on_sm121_cuda13(monkeypatch):
    import flashinfer.gemm.gemm_base as gemm_base

    monkeypatch.setattr(gemm_base, "get_cuda_version", lambda: Version("13.0"))
    monkeypatch.setattr(gemm_base, "get_compute_capability", lambda device: (12, 1))
    monkeypatch.setattr(gemm_base, "CUDNN_AVAILABLE", False)

    backends = gemm_base._heuristic_func_mm_fp4(
        ["b12x", "cutlass", "cudnn"],
        SimpleNamespace(device=torch.device("cuda")),
        None,
        None,
        None,
        use_nvfp4=True,
    )

    assert backends == ["b12x", "cutlass", "cudnn"]


def test_mm_fp4_auto_does_not_select_b12x_for_mxfp4(monkeypatch):
    import flashinfer.gemm.gemm_base as gemm_base

    monkeypatch.setattr(gemm_base, "get_cuda_version", lambda: Version("13.0"))
    monkeypatch.setattr(gemm_base, "get_compute_capability", lambda device: (12, 1))
    monkeypatch.setattr(gemm_base, "CUDNN_AVAILABLE", False)

    backends = gemm_base._heuristic_func_mm_fp4(
        ["b12x", "cutlass", "cudnn"],
        SimpleNamespace(device=torch.device("cuda")),
        None,
        None,
        None,
        use_nvfp4=False,
    )

    assert backends == ["cutlass", "cudnn"]


def test_mm_fp4_auto_does_not_select_b12x_for_fp16_output(monkeypatch):
    import flashinfer.gemm.gemm_base as gemm_base

    monkeypatch.setattr(gemm_base, "get_cuda_version", lambda: Version("13.0"))
    monkeypatch.setattr(gemm_base, "get_compute_capability", lambda device: (12, 1))
    monkeypatch.setattr(gemm_base, "CUDNN_AVAILABLE", False)

    backends = gemm_base._heuristic_func_mm_fp4(
        ["b12x", "cutlass", "cudnn"],
        SimpleNamespace(device=torch.device("cuda")),
        None,
        None,
        None,
        out_dtype=torch.float16,
        use_nvfp4=True,
    )

    assert backends == ["cutlass", "cudnn"]


def test_mm_fp8_low_latency_guard_rejects_sm121(monkeypatch):
    import flashinfer.gemm.gemm_base as gemm_base

    monkeypatch.setattr(gemm_base, "get_compute_capability", lambda device: (12, 1))

    with pytest.raises(NotImplementedError, match="SM100/SM103.*121"):
        gemm_base._check_trtllm_low_latency_gemm_supported(torch.device("cuda"))


def test_fp4_quantization_backend_keeps_sm121_arch_specific(monkeypatch):
    import flashinfer.quantization.fp4_quantization as fp4_quantization

    monkeypatch.setattr(fp4_quantization, "is_cuda_version_at_least", lambda _: True)

    assert fp4_quantization._canonicalize_fp4_quantization_backend("120") == "120f"
    assert fp4_quantization._canonicalize_fp4_quantization_backend("121") == "121"


def test_aot_oai_oss_attention_sink_does_not_emit_sm90_without_sm90():
    from flashinfer.aot import gen_attention

    specs = list(
        gen_attention(
            f16_dtype_=[torch.float16],
            f8_dtype_=[],
            fa2_head_dim_=[],
            fa3_head_dim_=[],
            use_sliding_window_=[False],
            use_logits_soft_cap_=[False],
            has_sm90=False,
            has_sm100=False,
            add_gemma=False,
            add_oai_oss=True,
        )
    )

    names = [spec.name for spec in specs]
    assert any("attention_sink" in name for name in names)
    assert not any(name.endswith("_sm90") for name in names)


def test_aot_sm121_includes_selective_state_update_blackwell_modules(monkeypatch):
    import flashinfer.aot as aot

    def fake_spec(name):
        return SimpleNamespace(name=name)

    for attr_name in [
        "gen_api_log_stats_module",
        "gen_cascade_module",
        "gen_mhc_module",
        "gen_norm_module",
        "gen_page_module",
        "gen_quantization_module",
        "gen_rope_module",
        "gen_sampling_module",
        "gen_topk_module",
        "gen_fp4_kv_dequantization_module",
        "gen_fp4_kv_quantization_module",
    ]:
        monkeypatch.setattr(
            aot,
            attr_name,
            lambda attr_name=attr_name: fake_spec(attr_name),
        )

    calls = []

    def fake_ssu(kind):
        def _inner(*args):
            calls.append((kind, args))
            return fake_spec(f"selective_state_update_{kind}_{len(calls)}")

        return _inner

    monkeypatch.setattr(
        aot, "gen_selective_state_update_module", fake_ssu("generic")
    )
    monkeypatch.setattr(
        aot, "gen_selective_state_update_sm90_module", fake_ssu("sm90")
    )
    monkeypatch.setattr(
        aot, "gen_selective_state_update_sm100_module", fake_ssu("sm100")
    )

    aot.gen_all_modules(
        f16_dtype_=[],
        f8_dtype_=[],
        fa2_head_dim_=[],
        fa3_head_dim_=[],
        use_sliding_window_=[],
        use_logits_soft_cap_=[],
        sm_capabilities={"sm121": True},
        add_comm=False,
        add_gemma=False,
        add_oai_oss=False,
        add_moe=False,
        add_act=False,
        add_misc=True,
        add_xqa=False,
    )

    kinds = [kind for kind, _ in calls]
    assert kinds.count("generic") == 48
    assert kinds.count("sm100") == 48
    assert "sm90" not in kinds


def _make_gdn_prefill_tensors(dtype=torch.bfloat16):
    total_tokens = 64
    num_heads = 1
    head_size = 128
    q = torch.empty(total_tokens, num_heads, head_size, dtype=dtype)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    g = torch.empty(total_tokens, num_heads, dtype=torch.float32)
    beta = torch.empty_like(g)
    cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.int64)
    output = torch.empty_like(q)
    output_state = torch.empty(1, num_heads, head_size, head_size, dtype=torch.float32)
    return q, k, v, g, beta, cu_seqlens, output, output_state


def test_gdn_prefill_sm121_requires_native_sm12x_kernel(monkeypatch):
    import flashinfer.gdn_prefill as gdn_prefill

    monkeypatch.setattr(gdn_prefill.torch.version, "cuda", "13.0")
    monkeypatch.setattr(gdn_prefill, "get_compute_capability", lambda device: (12, 1))
    monkeypatch.setattr(gdn_prefill, "_has_sm12x_prefill", False)

    q, k, v, g, beta, cu_seqlens, output, output_state = _make_gdn_prefill_tensors()

    with pytest.raises(NotImplementedError, match="native SM12x implementation"):
        gdn_prefill.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=None,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            output=output,
            output_state=output_state,
        )


def test_gdn_prefill_sm121_dispatches_to_sm12x_kernel(monkeypatch):
    import flashinfer.gdn_prefill as gdn_prefill

    monkeypatch.setattr(gdn_prefill.torch.version, "cuda", "13.0")
    monkeypatch.setattr(gdn_prefill, "get_compute_capability", lambda device: (12, 1))
    monkeypatch.setattr(gdn_prefill, "_has_sm12x_prefill", True)

    calls = []

    def fake_sm12x_kernel(
        q,
        k,
        v,
        gate,
        beta,
        output,
        cu_seqlens,
        initial_state,
        output_state,
        scale,
        checkpoint_every_n_tokens=0,
        cu_checkpoints=None,
        output_checkpoints=None,
    ):
        calls.append(
            {
                "q": q,
                "k": k,
                "v": v,
                "gate": gate,
                "beta": beta,
                "output": output,
                "cu_seqlens_dtype": cu_seqlens.dtype,
                "initial_state": initial_state,
                "output_state": output_state,
                "scale": scale,
                "checkpoint_every_n_tokens": checkpoint_every_n_tokens,
                "cu_checkpoints": cu_checkpoints,
                "output_checkpoints": output_checkpoints,
            }
        )

    monkeypatch.setattr(gdn_prefill, "chunk_gated_delta_rule_sm12x", fake_sm12x_kernel)

    q, k, v, g, beta, cu_seqlens, output, output_state = _make_gdn_prefill_tensors()

    result = gdn_prefill.chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        output=output,
        output_state=output_state,
    )

    assert result == (output, output_state)
    assert len(calls) == 1
    assert calls[0]["q"] is q
    assert calls[0]["k"] is k
    assert calls[0]["output"] is output
    assert calls[0]["output_state"] is output_state
    assert calls[0]["cu_seqlens_dtype"] == torch.int32


def test_gdn_prefill_sm121_l2norms_q_and_k_before_dispatch(monkeypatch):
    import flashinfer.gdn_prefill as gdn_prefill

    monkeypatch.setattr(gdn_prefill.torch.version, "cuda", "13.0")
    monkeypatch.setattr(gdn_prefill, "get_compute_capability", lambda device: (12, 1))
    monkeypatch.setattr(gdn_prefill, "_has_sm12x_prefill", True)

    calls = []

    def fake_sm12x_kernel(
        q,
        k,
        v,
        gate,
        beta,
        output,
        cu_seqlens,
        initial_state,
        output_state,
        scale,
        checkpoint_every_n_tokens=0,
        cu_checkpoints=None,
        output_checkpoints=None,
    ):
        calls.append({"q": q, "k": k, "output": output, "output_state": output_state})

    monkeypatch.setattr(gdn_prefill, "chunk_gated_delta_rule_sm12x", fake_sm12x_kernel)

    q, k, v, g, beta, cu_seqlens, output, output_state = _make_gdn_prefill_tensors()
    q.copy_(torch.randn_like(q) + 1.0)
    k.copy_(torch.randn_like(k) + 1.0)
    original_q = q.clone()
    original_k = k.clone()

    result = gdn_prefill.chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        output=output,
        output_state=output_state,
        use_qk_l2norm_in_kernel=True,
    )

    assert result == (output, output_state)
    assert len(calls) == 1
    assert calls[0]["q"] is not q
    assert calls[0]["k"] is not k
    torch.testing.assert_close(
        torch.linalg.vector_norm(calls[0]["q"].float(), dim=-1),
        torch.ones_like(calls[0]["q"].float().sum(dim=-1)),
        atol=5e-3,
        rtol=5e-3,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(calls[0]["k"].float(), dim=-1),
        torch.ones_like(calls[0]["k"].float().sum(dim=-1)),
        atol=5e-3,
        rtol=5e-3,
    )
    torch.testing.assert_close(q, original_q)
    torch.testing.assert_close(k, original_k)
