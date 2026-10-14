"""CPU check of dsa_backend._ragged_verify_rows: per packed verify token, its request's pool index
and KV length seq_len + position-in-block + 1; rows past the real tokens read request 0, length 1.
Also: with every request verifying T tokens it reproduces the uniform verify's per-row lengths
(seqlens_expand: seq + T - T + 1 + j).
kpool (kpool_plan.py): fill_ragged_kpool_write_plan matches _update_kpool_write_plan_kernel's
per-request outputs (the kernel reads request b from row b * N of the repeat_interleave'd page
table), and ragged_kpool_padded_rows maps packed row (b, j) to b * N + j, padding to the dump row.

  python test/manual/dsa_ragged_rows_check.py
"""
import torch

from sglang.srt.layers.attention.dsa.kpool_fp8_index import kpool_max_closed_pools
from sglang.srt.layers.attention.dsa.kpool_plan import (
    alloc_ragged_kpool_write_plan,
    fill_ragged_kpool_write_plan,
    ragged_kpool_padded_rows,
)
from sglang.srt.layers.attention.dsa_backend import _ragged_verify_rows


def main():
    seq = torch.tensor([10, 3, 7, 50])
    pool = torch.tensor([5, 9, 2, 7])
    lens = torch.tensor([2, 0, 3, 1])  # a zero-length request in the middle
    qo = torch.zeros(5, dtype=torch.int32)
    qo[1:] = lens.cumsum(0)
    kv, rows = _ragged_verify_rows(seq_lens=seq, qo_indptr=qo, req_pool_indices=pool, num_rows=8)
    assert kv.tolist() == [11, 12, 8, 9, 10, 51, 1, 1], kv.tolist()
    assert rows.tolist() == [5, 5, 2, 2, 2, 7, 5, 5], rows.tolist()
    T, bs = 8, 3
    seq = torch.tensor([20, 4, 100])
    qo = torch.arange(bs + 1, dtype=torch.int32) * T
    kv, rows = _ragged_verify_rows(seq_lens=seq, qo_indptr=qo, req_pool_indices=torch.arange(bs), num_rows=bs * T)
    want = torch.cat([s + torch.arange(1, T + 1) for s in seq]).to(torch.int32)
    assert torch.equal(kv, want) and rows.tolist() == [i for i in range(bs) for _ in range(T)]
    print("DSA_RAGGED_ROWS_OK")


def kernel_write_loc(write_start, padded_table, N, P, spp, max_closed):
    out = []
    for b, ws in enumerate(write_start.tolist()):
        row = []
        for p in range(max_closed):
            pool_id = ws // P + p
            tpr = min(max((pool_id // spp) * P, 0), padded_table.shape[1] - 1)
            row.append(int(padded_table[b * N, tpr]) * spp + pool_id % spp)
        out.append(row)
    return out


def kpool_main():
    N, P, spp = 8, 4, 16
    seq = torch.tensor([10, 3, 70, 1029, 0])
    pool = torch.tensor([5, 9, 2, 7, 0])
    lens = torch.tensor([2, 0, 8, 5, 0])  # an empty request and a padding slot
    qo = torch.zeros(6, dtype=torch.int32)
    qo[1:] = lens.cumsum(0)
    table = torch.randint(1, 1000, (5, 40), dtype=torch.int32)
    plan = alloc_ragged_kpool_write_plan(num_slots=5, num_draft_tokens=N, pool_size=P, device=torch.device("cpu"))
    fill_ragged_kpool_write_plan(plan, write_start=seq, req_pool_indices=pool, verify_lens=qo[1:] - qo[:-1],
                                 real_page_rows=table, pool_size=P, slots_per_page=spp)
    want = kernel_write_loc(seq, table.repeat_interleave(N, dim=0), N, P, spp, kpool_max_closed_pools(N, P))
    assert plan.write_loc.tolist() == want, (plan.write_loc.tolist(), want)
    assert plan.req.tolist() == pool.tolist() and plan.write_start.tolist() == seq.tolist()
    assert plan.tail_logical_start.tolist() == [(s // P) * P for s in seq.tolist()]
    assert plan.effective_n_per_batch.tolist() == lens.tolist()
    rows = ragged_kpool_padded_rows(qo_indptr=qo, num_rows=18, num_draft_tokens=N)
    assert rows.tolist() == [0, 1] + [16 + j for j in range(8)] + [24 + j for j in range(5)] + [40] * 3, rows.tolist()
    print("DSA_RAGGED_KPOOL_OK")


if __name__ == "__main__":
    main()
    kpool_main()
