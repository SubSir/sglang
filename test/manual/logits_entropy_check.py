"""GPU check: spec_confidence_triton.logits_entropy (one fused pass) against entropy_from_logits (torch) on
draft-like logits: bf16 and fp32, peaked and flat rows, with and without per-request temperatures, vocab 151936.

  python test/manual/logits_entropy_check.py
"""
import torch

from sglang.kernels.ops.speculative.spec_confidence_triton import logits_entropy
from sglang.srt.speculative.spec_confidence import entropy_from_logits


def main():
    torch.manual_seed(0)
    ok = True
    for dtype in (torch.bfloat16, torch.float32):
        for bs, gamma in ((1, 7), (37, 7), (64, 15)):
            x = torch.randn(bs, gamma, 151936, device="cuda") * torch.rand(bs, gamma, 1, device="cuda") * 6
            x[0, 0, 123] += 60.0  # a near-one-hot row
            x = x.to(dtype)
            for temps in (None, torch.rand(bs, device="cuda") * 1.5 + 0.2):
                ref = entropy_from_logits(x, temps)
                got = logits_entropy(x, temps)
                err = float((got - ref).abs().max())
                good = err < 2e-4 and bool(torch.isfinite(got).all())
                ok &= good
                print(f"{str(dtype):15s} bs {bs:2d} gamma {gamma:2d} temps {'on ' if temps is not None else 'off'}: "
                      f"max|dH| {err:.1e} {'ok' if good else 'FAIL'}")
    print("LOGITS_ENTROPY_OK" if ok else "LOGITS_ENTROPY_FAIL")


if __name__ == "__main__":
    main()
