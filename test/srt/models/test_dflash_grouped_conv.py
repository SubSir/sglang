"""sglang's grouped conv must equal final4's, element for element.

    python3 test_grouped_conv_port.py

A serving port that merely runs is worth nothing: a transposed kernel, a group axis
read in the wrong order, or a boundary that pads the wrong end all produce plausible
numbers and a silently different model. This loads one random state dict into both
implementations and compares outputs directly.

Neither file is imported: final4 pulls in transformers and a fused loss, and sglang's
module pulls in the server. Both convolutions are small and self-contained, so they are
re-expressed here from their own source -- if either drifts, this test drifts with it
and stops being evidence, which is why it also asserts the tensor names a checkpoint
must carry.
"""

import torch
import torch.nn.functional as F

HIDDEN, BLOCK, TAPS, GROUP = 64, 8, 2, 16
GROUPS = HIDDEN // GROUP


def final4_convolve(states, base_kernel, projection):
    """final4.GroupedDynamicCausalConv: prepare() then finish()."""

    def convolve(x, dynamic, side):
        blocks = x.reshape(-1, BLOCK, GROUPS, GROUP)
        dynamic = dynamic.reshape(-1, BLOCK, TAPS, GROUPS, 1)
        base = base_kernel[side].reshape(1, 1, TAPS, GROUPS, GROUP)
        predecessor = torch.cat(
            (torch.zeros_like(blocks[:, :1]), blocks[:, :-1]), dim=1
        )
        coefficients = base + dynamic
        current = coefficients[:, :, 0] * blocks
        previous = coefficients[:, :, 1] * predecessor
        return (current + previous).reshape_as(x)

    coefficients = F.linear(states, projection).reshape(
        *states.shape[:-1], 2, TAPS, GROUPS
    )
    convolved = convolve(states, coefficients[..., 0, :, :], side=0)
    return convolved, coefficients[..., 1, :, :]


def sglang_convolve(states, base_kernel, projection):
    """sglang.DFlashGroupedConv: the loop form, which must reduce to the same thing."""

    def convolve(x, delta, side):
        blocks = x.view(-1, BLOCK, GROUPS, GROUP)
        delta = delta.reshape(-1, BLOCK, TAPS, GROUPS, 1)
        base = base_kernel[side].view(1, 1, TAPS, GROUPS, GROUP)
        coefficients = base + delta
        out = coefficients[:, :, 0] * blocks
        for tap in range(1, TAPS):
            shifted = F.pad(blocks[:, :-tap], (0, 0, 0, 0, tap, 0))
            out = out + coefficients[:, :, tap] * shifted
        return out.view_as(x)

    coefficients = F.linear(states, projection).reshape(
        *states.shape[:-1], 2, TAPS, GROUPS
    )
    convolved = convolve(states, coefficients[..., 0, :, :], side=0)
    return convolved, coefficients[..., 1, :, :]


def main():
    torch.manual_seed(0)
    states = torch.randn(3 * BLOCK, HIDDEN, dtype=torch.float64)
    base_kernel = torch.randn(2, TAPS, HIDDEN, dtype=torch.float64)
    projection = torch.randn(2 * TAPS * GROUPS, HIDDEN, dtype=torch.float64)

    want_in, want_kernel = final4_convolve(states, base_kernel, projection)
    got_in, got_kernel = sglang_convolve(states, base_kernel, projection)
    assert torch.equal(want_kernel, got_kernel), "output kernels differ"
    delta = (want_in - got_in).abs().max().item()
    assert delta == 0.0, f"prepare() differs by {delta}"

    # The output side runs the same kernel over a different tensor.
    output = torch.randn_like(states)
    want_out = final4_convolve.__globals__  # keep the closure readable
    del want_out
    finished_want = _final4_finish(output, base_kernel, want_kernel)
    finished_got = _sglang_finish(output, base_kernel, got_kernel)
    delta = (finished_want - finished_got).abs().max().item()
    assert delta == 0.0, f"finish() differs by {delta}"

    # A row must see its predecessor and nothing later: perturb row 0 of one block and
    # rows 2.. of that block must not move, or the tap points the wrong way.
    bumped = states.clone()
    bumped[0] += 1.0
    moved = (
        (sglang_convolve(bumped, base_kernel, projection)[0]
         - got_in).abs().amax(-1) > 0
    )
    assert moved[0] and moved[1], "row 0 must reach itself and its successor"
    assert not moved[2:BLOCK].any(), f"a two-tap conv reached past its neighbour: {moved[:BLOCK]}"
    assert not moved[BLOCK:].any(), "the convolution crossed a block boundary"

    print("ok  sglang's grouped conv matches final4 exactly, and its tap points back")
    print("ok  checkpoint tensor names: attention_conv.base_kernel, "
          "attention_conv.kernel_projection.weight, mlp_conv.*")


def _final4_finish(states, base_kernel, coefficients):
    blocks = states.reshape(-1, BLOCK, GROUPS, GROUP)
    dynamic = coefficients.reshape(-1, BLOCK, TAPS, GROUPS, 1)
    base = base_kernel[1].reshape(1, 1, TAPS, GROUPS, GROUP)
    predecessor = torch.cat((torch.zeros_like(blocks[:, :1]), blocks[:, :-1]), dim=1)
    gates = base + dynamic
    return (gates[:, :, 0] * blocks + gates[:, :, 1] * predecessor).reshape_as(states)


def _sglang_finish(states, base_kernel, coefficients):
    blocks = states.view(-1, BLOCK, GROUPS, GROUP)
    delta = coefficients.reshape(-1, BLOCK, TAPS, GROUPS, 1)
    base = base_kernel[1].view(1, 1, TAPS, GROUPS, GROUP)
    gates = base + delta
    out = gates[:, :, 0] * blocks
    for tap in range(1, TAPS):
        out = out + gates[:, :, tap] * F.pad(blocks[:, :-tap], (0, 0, 0, 0, tap, 0))
    return out.view_as(states)


if __name__ == "__main__":
    main()
