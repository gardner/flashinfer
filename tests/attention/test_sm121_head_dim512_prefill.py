import math

import pytest
import torch

import flashinfer


def _is_sm12x() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


def _paged_prefill_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    *,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = qo_indptr.numel() - 1
    _, page_size, num_kv_heads, head_dim = k_cache.shape
    num_qo_heads = q.shape[1]
    group_size = num_qo_heads // num_kv_heads

    out = torch.empty_like(q)
    lse = torch.empty((q.shape[0], num_qo_heads), dtype=torch.float32, device=q.device)

    for batch_idx in range(batch_size):
        q_begin = int(qo_indptr[batch_idx].item())
        q_end = int(qo_indptr[batch_idx + 1].item())
        page_begin = int(kv_indptr[batch_idx].item())
        page_end = int(kv_indptr[batch_idx + 1].item())
        if page_begin == page_end:
            out[q_begin:q_end].zero_()
            lse[q_begin:q_end].fill_(-float("inf"))
            continue

        pages = kv_indices[page_begin:page_end].long()
        kv_len = (page_end - page_begin - 1) * page_size + int(
            kv_last_page_len[batch_idx].item()
        )
        k = k_cache[pages].reshape(-1, num_kv_heads, head_dim)[:kv_len].float()
        v = v_cache[pages].reshape(-1, num_kv_heads, head_dim)[:kv_len].float()

        for q_idx in range(q_begin, q_end):
            for qo_head in range(num_qo_heads):
                kv_head = qo_head // group_size
                logits = torch.matmul(q[q_idx, qo_head].float(), k[:, kv_head].T) * sm_scale
                lse[q_idx, qo_head] = torch.logsumexp(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1)
                out[q_idx, qo_head] = torch.matmul(probs, v[:, kv_head]).to(q.dtype)

    return out, lse / math.log(2.0)


@pytest.mark.skipif(not _is_sm12x(), reason="SM12x-only GB10 regression")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sm12x_fa2_paged_prefill_head_dim512(dtype):
    torch.manual_seed(0)

    batch_size = 2
    qo_lens = [2, 1]
    pages_per_seq = 2
    page_size = 16
    num_qo_heads = 4
    num_kv_heads = 2
    head_dim = 512
    total_pages = batch_size * pages_per_seq

    qo_indptr = torch.tensor([0, *torch.cumsum(torch.tensor(qo_lens), 0)], dtype=torch.int32)
    qo_indptr = qo_indptr.to("cuda")
    kv_indptr = (torch.arange(batch_size + 1, dtype=torch.int32, device="cuda") * pages_per_seq)
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")
    kv_last_page_len = torch.full((batch_size,), page_size, dtype=torch.int32, device="cuda")

    q = torch.randn(sum(qo_lens), num_qo_heads, head_dim, dtype=dtype, device="cuda")
    k_cache = torch.randn(
        total_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device="cuda"
    )
    v_cache = torch.randn_like(k_cache)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        backend="fa2",
    )
    sm_scale = 1.0 / math.sqrt(head_dim)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        sm_scale=sm_scale,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    out = torch.empty_like(q)
    actual, actual_lse = wrapper.run_return_lse(q, (k_cache, v_cache), out=out)
    expected, expected_lse = _paged_prefill_reference(
        q,
        k_cache,
        v_cache,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        sm_scale=sm_scale,
    )

    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=2e-2, atol=2e-2)
