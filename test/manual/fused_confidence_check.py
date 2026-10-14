"""GPU check: fused_entropy_confidence (one Triton launch) against the torch reference
(entropy_from_probs + entropy_confidence) on random sparse draft distributions, one and two tables
(per-row thinking flags), gamma 7 and 15. Reports max |diff| of entropy and confidence and how many
(row, slot) calibration bins differ (only possible when a raw chain value sits within an ulp of an edge).

  python test/manual/fused_confidence_check.py
"""
import numpy as np
import torch

from sglang.kernels.ops.speculative.spec_confidence_triton import (
    fused_entropy_confidence,
    stack_calibration_tables,
)
from sglang.srt.speculative.spec_confidence import (
    EntropyCalibrationTable,
    EntropyCalibrator,
    entropy_confidence,
    entropy_from_probs,
    raw_survival_chain,
)


def table(gamma, rng, bins=20):
    edges, values = [], []
    for k in range(gamma):
        e = np.unique(np.sort(rng.uniform(0, 1, bins + 1) ** (1 + k / 3)))
        if k == gamma - 1:  # a constant slot, as fitted tables can have
            e = e[:1]
        edges.append([float(x) for x in e])
        values.append([float(x) for x in np.sort(rng.uniform(0, 1, max(len(e) - 1, 1)))])
    return EntropyCalibrationTable(edges=edges, values=values)


def main():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    ok = True
    for gamma in (7, 15):
        cal1 = EntropyCalibrator(table=table(gamma, rng), gamma=gamma, device="cuda")
        cal2 = EntropyCalibrator(table=table(gamma, rng), gamma=gamma, device="cuda")
        for bs in (1, 13, 64, 200):
            logits = torch.randn(bs, gamma, 16, device="cuda") * torch.rand(bs, gamma, 1, device="cuda") * 8
            q = torch.softmax(logits, -1) * torch.rand(bs, gamma, 1, device="cuda").clamp_min(0.3)  # sparse mass < 1
            q[0, 0] = 0  # an all-zero row
            q = q.to(torch.bfloat16)
            think = torch.rand(bs, device="cuda") < 0.5
            for two in (False, True):
                ref_e = entropy_from_probs(q)
                ref_c = entropy_confidence(entropy=ref_e, calibrator=cal1, calibrator_nothink=cal2 if two else None,
                                           thinking=think if two else None)
                tabs = stack_calibration_tables([cal1, cal2] if two else [cal1])
                e, c = fused_entropy_confidence(q, tabs, think if two else None)
                de, dc = float((e - ref_e).abs().max()), float((c - ref_c).abs().max())
                raw_r, raw_f = raw_survival_chain(torch.exp(-ref_e)), raw_survival_chain(torch.exp(-e))
                flips = int(((cal1.survival(raw_r) != cal1.survival(raw_f))).sum()) if not two else -1
                good = de < 1e-5 and dc < 1e-5
                ok &= good
                print(f"gamma {gamma:2d} bs {bs:3d} tables {2 if two else 1}: max|d entropy| {de:.1e}, "
                      f"max|d confidence| {dc:.1e}, survival flips {flips} {'ok' if good else 'FAIL'}")
    print("FUSED_CONFIDENCE_OK" if ok else "FUSED_CONFIDENCE_FAIL")


if __name__ == "__main__":
    main()
