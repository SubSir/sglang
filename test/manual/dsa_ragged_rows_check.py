"""CPU check of dsa_backend._ragged_verify_rows: per packed verify token, its request's pool index
and KV length seq_len + position-in-block + 1; rows past the real tokens read request 0, length 1.
Also: with every request verifying T tokens it reproduces the uniform verify's per-row lengths
(seqlens_expand: seq + T - T + 1 + j).

  python test/manual/dsa_ragged_rows_check.py
"""
import torch

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


if __name__ == "__main__":
    main()
