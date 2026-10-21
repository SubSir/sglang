"""Compact greedy accept from packed argmax ids == accept from the strided full-vocab logits.

CPU (torch reference paths): python test/manual/compact_greedy_predict_check.py
GPU (triton paths):          python test/manual/compact_greedy_predict_check.py --cuda
"""

import sys

import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import AcceptGreedy
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    ScatterCompactToStrided,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

dev = "cuda" if "--cuda" in sys.argv else "cpu"
torch.manual_seed(0)
for trial in range(20):
    bs, T, vocab = int(torch.randint(1, 17, ())), 8, 97
    lens = torch.randint(1, T + 1, (bs,))
    n = int(lens.sum())
    pad = int(torch.randint(0, 5, ()))
    qo = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)]).to(torch.int32)
    layout = RaggedVerifyLayout(
        verify_lens=lens.to(dev), graph_num_tokens=n + pad,
        extend_start_loc=qo[:-1].to(dev), qo_indptr_device=qo.to(dev),
    )
    compact = torch.randn(n + pad, vocab, device=dev)
    # plant draft hits so accept lengths vary
    cand = torch.randint(0, vocab, (bs, T), device=dev)
    for i in range(bs):
        s0 = int(qo[i])
        for w in range(int(lens[i]) - 1):
            if torch.rand(()) < 0.7:
                cand[i, w + 1] = compact[s0 + w].argmax()
    strided = ScatterCompactToStrided.execute(
        compact=compact, layout=layout, fill_value=0.0, verify_num_draft_tokens=T
    )
    ref = AcceptGreedy.execute(
        candidates=cand, target_logits=strided, verify_num_draft_tokens=T,
        cutoff_verify_lens=layout.verify_lens,
    )
    predict = ScatterCompactToStrided.execute(
        compact=torch.argmax(compact, dim=-1, keepdim=True), layout=layout,
        fill_value=0, verify_num_draft_tokens=T,
    ).view(-1)
    got = AcceptGreedy.execute(
        candidates=cand, target_logits=None, verify_num_draft_tokens=T,
        cutoff_verify_lens=layout.verify_lens, target_predict=predict,
    )
    for r, g in zip(ref, got):
        assert torch.equal(r.long(), g.long()), (trial, r, g)
print("compact greedy predict == strided logits accept: OK (20 trials)")
