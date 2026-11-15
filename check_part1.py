"""Small, independent checks for every Part 1 exercise.

From this folder, run:
    python3 check_part1.py 03
    python3 check_part1.py all

Expected results below are worked out for tiny inputs. This file does not
implement the exercise functions.
"""

from __future__ import annotations

import argparse
import math
import time

import torch

try:
    import part1_pytorch as exercises
except ModuleNotFoundError as error:
    if error.name != "part1_pytorch":
        raise
    import problems as exercises


def expect_tensor(label: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not isinstance(actual, torch.Tensor):
        raise AssertionError(f"{label}: expected a tensor, got {type(actual).__name__}")
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{label}: expected shape {tuple(expected.shape)}, got {tuple(actual.shape)}"
        )
    if expected.dtype == torch.bool:
        if actual.dtype != torch.bool or not torch.equal(actual, expected):
            raise AssertionError(f"{label}: expected\n{expected}\ngot\n{actual}")
        return
    if not torch.isfinite(actual).all():
        raise AssertionError(f"{label}: output contains Inf or NaN; expected\n{expected}")
    try:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    except AssertionError as error:
        raise AssertionError(f"{label}: expected\n{expected}\ngot\n{actual}\n{error}") from error


def expect_prediction(
    label: str,
    actual: tuple[torch.Tensor, torch.Tensor],
    expected_probs: torch.Tensor,
    expected_id: int,
) -> None:
    if not isinstance(actual, tuple) or len(actual) != 2:
        raise AssertionError(f"{label}: expected (probabilities, token_id)")
    probs, token_id = actual
    expect_tensor(f"{label} probabilities", probs, expected_probs)
    if not isinstance(token_id, torch.Tensor) or token_id.shape != torch.Size([]):
        raise AssertionError(f"{label}: token ID must be a scalar tensor")
    if token_id.dtype != torch.long or token_id.item() != expected_id:
        raise AssertionError(f"{label}: expected torch.long token ID {expected_id}, got {token_id}")


def check_01() -> None:
    """Check known Q-K scores, including zero/negative values and two D sizes."""
    q = torch.tensor(
        [[1.0, 2.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]
    )  # [L=2, D=4]
    k = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [-1.0, 0.0, 0.0, 0.0]]
    )  # [S=3, D=4]
    expected = torch.tensor([[0.5, 1.0, -0.5], [0.0, 0.5, 0.0]])
    actual = exercises.qk_scores_one_head(q, k)
    assert actual.shape == (2, 3), f"D=4: expected shape (2, 3), got {tuple(actual.shape)}"
    assert torch.isfinite(actual).all(), "D=4: zero dot products should produce finite zero scores"
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

    # D=9 gives a different scale. This catches a hard-coded D=4 solution.
    q9 = torch.zeros(2, 9)
    k9 = torch.zeros(2, 9)
    q9[0, 0], q9[1, 1] = 3.0, 3.0
    k9[0, 0], k9[1, 1] = 2.0, 2.0
    expected9 = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    actual9 = exercises.qk_scores_one_head(q9, k9)
    assert actual9.shape == (2, 2), f"D=9: expected shape (2, 2), got {tuple(actual9.shape)}"
    assert torch.isfinite(actual9).all(), "D=9: zero dot products should produce finite zero scores"
    torch.testing.assert_close(actual9, expected9, rtol=0, atol=1e-6)


def check_02() -> None:
    """Check a weighted vector sum and the S=1 edge case."""
    weights = torch.tensor([[0.25, 0.75, 0.0], [0.0, 0.5, 0.5]])
    v = torch.tensor([[2.0, 10.0], [4.0, 20.0], [8.0, 30.0]])
    expected = torch.tensor([[3.5, 17.5], [6.0, 25.0]])
    expect_tensor("two queries, three values", exercises.weighted_values(weights, v), expected)

    one_key_weights = torch.ones(2, 1)
    one_key_v = torch.tensor([[-3.0, 2.0, 7.0]])
    expected_one_key = torch.tensor([[-3.0, 2.0, 7.0], [-3.0, 2.0, 7.0]])
    expect_tensor(
        "one visible key", exercises.weighted_values(one_key_weights, one_key_v), expected_one_key
    )


def check_03() -> None:
    """Check causal visibility, window boundaries, and the first token."""
    full = torch.tensor(
        [[True, True, True, False, False],
         [True, True, True, True, False],
         [True, True, True, True, True]]
    )
    expect_tensor(
        "two past plus three new, full context",
        exercises.causal_window_mask(2, 3, device="cpu"),
        full,
    )

    window_two = torch.tensor(
        [[False, True, True, False, False],
         [False, False, True, True, False],
         [False, False, False, True, True]]
    )
    expect_tensor(
        "window includes current token and one prior token",
        exercises.causal_window_mask(2, 3, window=2),
        window_two,
    )
    expect_tensor(
        "first token sees itself",
        exercises.causal_window_mask(0, 1, window=1),
        torch.tensor([[True]]),
    )


def check_04() -> None:
    """Check causal attention, sliding window, and nonuniform softmax."""
    q = torch.zeros(3, 4)
    k = torch.zeros(5, 4)
    v = torch.tensor([[0.0, 0.0], [2.0, 10.0], [4.0, 20.0], [6.0, 30.0], [8.0, 40.0]])
    expect_tensor(
        "causal prefix means",
        exercises.causal_attention_one_head(q, k, v, past_len=2),
        torch.tensor([[2.0, 10.0], [3.0, 15.0], [4.0, 20.0]]),
    )
    expect_tensor(
        "two-token sliding window means",
        exercises.causal_attention_one_head(q, k, v, past_len=2, window=2),
        torch.tensor([[3.0, 15.0], [5.0, 25.0], [7.0, 35.0]]),
    )

    q_nonuniform = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
    k_nonuniform = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    v_nonuniform = torch.tensor([[0.0], [10.0]])
    expect_tensor(
        "nonuniform scaled scores and softmax",
        exercises.causal_attention_one_head(
            q_nonuniform, k_nonuniform, v_nonuniform, past_len=1
        ),
        torch.tensor([[7.310586]]),
    )


def check_05() -> None:
    """Check head isolation, per-head softmax, causal and window masking."""
    q = torch.zeros(2, 2, 4)
    k = torch.zeros(4, 2, 4)
    v = torch.tensor(
        [[[0.0, 0.0], [10.0, 100.0]],
         [[2.0, 4.0], [20.0, 200.0]],
         [[4.0, 8.0], [30.0, 300.0]],
         [[6.0, 12.0], [40.0, 400.0]]]
    )
    expect_tensor(
        "two independent heads, full context",
        exercises.multi_head_attention(q, k, v, past_len=2),
        torch.tensor([[[2.0, 4.0], [20.0, 200.0]], [[3.0, 6.0], [25.0, 250.0]]]),
    )
    expect_tensor(
        "two independent heads, sliding window",
        exercises.multi_head_attention(q, k, v, past_len=2, window=2),
        torch.tensor([[[3.0, 6.0], [25.0, 250.0]], [[5.0, 10.0], [35.0, 350.0]]]),
    )

    q_nonuniform = torch.zeros(1, 2, 4)
    q_nonuniform[0, 0, 0], q_nonuniform[0, 1, 0] = 2.0, -2.0
    k_nonuniform = torch.zeros(2, 2, 4)
    k_nonuniform[1, :, 0] = 1.0
    v_nonuniform = torch.tensor([[[0.0], [100.0]], [[10.0], [0.0]]])
    expect_tensor(
        "each head has its own nonuniform softmax",
        exercises.multi_head_attention(
            q_nonuniform, k_nonuniform, v_nonuniform, past_len=1
        ),
        torch.tensor([[[7.310586], [73.10586]]]),
    )


def check_06() -> None:
    """Check shared K/V, per-query-head softmax, causal masking, and windowing."""
    q = torch.zeros(2, 3, 4)
    k = torch.zeros(4, 1, 4)
    v = torch.tensor([[[0.0, 0.0]], [[2.0, 20.0]], [[4.0, 40.0]], [[6.0, 60.0]]])
    expect_tensor(
        "MQA full causal context",
        exercises.multi_query_attention(q, k, v, past_len=2),
        torch.tensor([[[2.0, 20.0]] * 3, [[3.0, 30.0]] * 3]),
    )
    expect_tensor(
        "MQA two-token window",
        exercises.multi_query_attention(q, k, v, past_len=2, window=2),
        torch.tensor([[[3.0, 30.0]] * 3, [[5.0, 50.0]] * 3]),
    )

    q = torch.tensor([[[2.0, 0.0, 0.0, 0.0],
                       [-2.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0]]])
    k = torch.tensor([[[0.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]])
    v = torch.tensor([[[0.0]], [[10.0]]])
    expect_tensor(
        "MQA heads have separate softmax distributions",
        exercises.multi_query_attention(q, k, v, past_len=1),
        torch.tensor([[[7.310586], [2.689414], [5.0]]]),
    )


def check_07() -> None:
    """Check contiguous GQA groups, two head-count ratios, and edge cases."""
    q = torch.zeros(2, 4, 4)
    k = torch.zeros(4, 2, 4)
    v = torch.tensor([[[0.0], [100.0]], [[2.0], [120.0]],
                      [[4.0], [140.0]], [[6.0], [160.0]]])
    expect_tensor(
        "GQA 4:2 full causal context",
        exercises.grouped_query_attention(q, k, v, past_len=2),
        torch.tensor([[[2.0], [2.0], [120.0], [120.0]],
                      [[3.0], [3.0], [130.0], [130.0]]]),
    )
    expect_tensor(
        "GQA 4:2 two-token window",
        exercises.grouped_query_attention(q, k, v, past_len=2, window=2),
        torch.tensor([[[3.0], [3.0], [130.0], [130.0]],
                      [[5.0], [5.0], [150.0], [150.0]]]),
    )

    q = torch.tensor([[[2.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0],
                       [0.0, 2.0, 0.0, 0.0], [0.0, -2.0, 0.0, 0.0]]])
    k = torch.zeros(2, 2, 4)
    k[1, 0, 0] = 1.0
    k[1, 1, 1] = 1.0
    v = torch.tensor([[[0.0], [100.0]], [[10.0], [0.0]]])
    expect_tensor(
        "GQA query heads within each group keep their own scores",
        exercises.grouped_query_attention(q, k, v, past_len=1),
        torch.tensor([[[7.310586], [2.689414], [26.89414], [73.10586]]]),
    )

    q = torch.zeros(1, 16, 4)
    k = torch.zeros(1, 2, 4)
    v = torch.tensor([[[3.0], [30.0]]])
    expect_tensor(
        "GQA 16:2 maps eight adjacent Q heads to each KV head",
        exercises.grouped_query_attention(q, k, v, past_len=0),
        torch.tensor([[[3.0]] * 8 + [[30.0]] * 8]),
    )
    expect_tensor(
        "GQA MQA edge case (Hkv=1)",
        exercises.grouped_query_attention(torch.zeros(1, 3, 4), torch.zeros(1, 1, 4),
                                          torch.tensor([[[7.0]]]), past_len=0),
        torch.tensor([[[7.0]] * 3]),
    )
    expect_tensor(
        "GQA MHA edge case (Hq=Hkv)",
        exercises.grouped_query_attention(torch.zeros(1, 2, 4), torch.zeros(1, 2, 4),
                                          torch.tensor([[[7.0], [70.0]]]), past_len=0),
        torch.tensor([[[7.0], [70.0]]]),
    )

    try:
        exercises.grouped_query_attention(torch.zeros(1, 3, 4), torch.zeros(1, 2, 4),
                                          torch.zeros(1, 2, 1), past_len=0)
    except (AssertionError, RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("GQA should reject Hq=3, Hkv=2: the group size is not integral")


def check_04a() -> None:
    """Check vocabulary probabilities and greedy ID after single-head attention."""
    q = torch.zeros(1, 4)
    k = torch.zeros(2, 4)
    v = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    w_vocab = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    e2, e1 = math.exp(2), math.e
    expect_prediction(
        "single head: attention mean then vocabulary softmax",
        exercises.predict_next_token_single_head(q, k, v, w_vocab, past_len=1),
        torch.tensor([e2, e1, 1.0]) / (e2 + e1 + 1),
        0,
    )
    expect_prediction(
        "single head: one-token attention window changes prediction",
        exercises.predict_next_token_single_head(q, k, v, w_vocab, past_len=1, window=1),
        torch.tensor([1.0, e2, 1.0]) / (2 + e2),
        1,
    )


def check_07b() -> None:
    """Check GQA head flattening, output projection, residual, and vocab softmax."""
    q = torch.zeros(1, 4, 4)
    k = torch.zeros(1, 2, 4)
    v = torch.tensor([[[1.0], [2.0]]])
    w_o = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0], [0.0, 0.0, 0.5]])
    residual = torch.tensor([[1.0, -0.5, 0.0]])
    w_vocab = torch.eye(3)
    weights = torch.tensor([math.exp(2), math.exp(0.5), math.exp(3)])
    expect_prediction(
        "GQA: four heads become one hidden vector and three vocab logits",
        exercises.predict_next_token_gqa(q, k, v, w_o, residual, w_vocab, past_len=0),
        weights / weights.sum(),
        2,
    )

    q = torch.tensor([[[2.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0],
                       [2.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0]]])
    k = torch.zeros(2, 2, 4)
    k[1, :, 0] = 1.0
    v = torch.tensor([[[0.0], [10.0]], [[2.0], [0.0]]])
    w_o = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    e = math.e
    logits = [2*e/(1+e), 2/(1+e), 10/(1+e)]
    weights = torch.tensor([math.exp(x) for x in logits])
    expect_prediction(
        "GQA: different query heads change final vocabulary probabilities",
        exercises.predict_next_token_gqa(q, k, v, w_o, torch.zeros(1, 3),
                                         torch.eye(3), past_len=1),
        weights / weights.sum(),
        2,
    )

    q = torch.zeros(2, 4, 4)
    k = torch.zeros(2, 2, 4)
    v = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    w_o = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0], [0.0, 0.0, 0.5]])
    residual = torch.tensor([[20.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    weights = torch.tensor([math.exp(2), math.exp(3), math.exp(4.5)])
    expect_prediction(
        "GQA: each token has its own residual; predict from the last row",
        exercises.predict_next_token_gqa(q, k, v, w_o, residual,
                                         torch.eye(3), past_len=0),
        weights / weights.sum(),
        2,
    )


def check_08() -> None:
    """Check in-place append, preservation of old and unused rows, and capacity."""
    k_cache = torch.full((5, 2, 2), -99.0)
    v_cache = torch.full((5, 2, 1), -88.0)
    k_cache[:2] = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
                                [[5.0, 6.0], [7.0, 8.0]]])
    v_cache[:2] = torch.tensor([[[10.0], [20.0]], [[30.0], [40.0]]])
    old_k, old_v = k_cache[:2].clone(), v_cache[:2].clone()
    k_new = torch.tensor([[[9.0, 10.0], [11.0, 12.0]],
                          [[13.0, 14.0], [15.0, 16.0]]])
    v_new = torch.tensor([[[50.0], [60.0]], [[70.0], [80.0]]])
    used = exercises.append_contiguous_kv(k_cache, v_cache, 2, k_new, v_new)
    assert used == 4, f"append: expected used=4, got {used}"
    expect_tensor("old K rows remain unchanged", k_cache[:2], old_k)
    expect_tensor("old V rows remain unchanged", v_cache[:2], old_v)
    expect_tensor("new K rows occupy positions 2 and 3", k_cache[2:4], k_new)
    expect_tensor("new V rows occupy positions 2 and 3", v_cache[2:4], v_new)
    expect_tensor("unused K row remains untouched", k_cache[4], torch.full((2, 2), -99.0))
    expect_tensor("unused V row remains untouched", v_cache[4], torch.full((2, 1), -88.0))

    used = exercises.append_contiguous_kv(k_cache, v_cache, used, k_new[:1], v_new[:1])
    assert used == 5, f"second append: expected used=5, got {used}"
    expect_tensor("second append fills the last row", k_cache[4:5], k_new[:1])
    try:
        exercises.append_contiguous_kv(k_cache, v_cache, used, k_new[:1], v_new[:1])
    except (AssertionError, IndexError, RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("append beyond capacity should raise an error")


def check_09() -> None:
    """Check one-token GQA output and the exact cache rows written."""
    def run(window: int | None, expected: torch.Tensor) -> None:
        q = torch.zeros(4, 4)
        k_new = torch.tensor([[9.0, 0.0, 0.0, 0.0], [0.0, 9.0, 0.0, 0.0]])
        v_new = torch.tensor([[6.0], [160.0]])
        k_cache = torch.full((5, 2, 4), -99.0)
        v_cache = torch.full((5, 2, 1), -88.0)
        k_cache[:2] = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                                    [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]])
        v_cache[:2] = torch.tensor([[[0.0], [100.0]], [[3.0], [130.0]]])
        old_k, old_v = k_cache[:2].clone(), v_cache[:2].clone()
        out, used = exercises.decode_one_token(q, k_new, v_new, k_cache, v_cache, 2,
                                                window=window)
        assert used == 3, f"decode: expected used=3, got {used}"
        expect_tensor("decode output", out, expected)
        expect_tensor("decode preserves old K", k_cache[:2], old_k)
        expect_tensor("decode preserves old V", v_cache[:2], old_v)
        expect_tensor("decode writes new K", k_cache[2], k_new)
        expect_tensor("decode writes new V", v_cache[2], v_new)
        expect_tensor("decode leaves unused K alone", k_cache[3:], torch.full((2, 2, 4), -99.0))
        expect_tensor("decode leaves unused V alone", v_cache[3:], torch.full((2, 2, 1), -88.0))

    run(None, torch.tensor([[3.0], [3.0], [130.0], [130.0]]))
    run(2, torch.tensor([[4.5], [4.5], [145.0], [145.0]]))

    q = torch.tensor([[2.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0],
                      [2.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0]])
    k_cache = torch.zeros(2, 2, 4)
    v_cache = torch.tensor([[[0.0], [100.0]], [[-88.0], [-88.0]]])
    k_new = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    v_new = torch.tensor([[10.0], [0.0]])
    out, used = exercises.decode_one_token(q, k_new, v_new, k_cache, v_cache, 1)
    assert used == 2, f"nonuniform decode: expected used=2, got {used}"
    expect_tensor(
        "decode uses scaled scores independently per grouped Q head",
        out,
        torch.tensor([[7.310586], [2.689414], [26.89414], [73.10586]]),
    )
    expect_tensor("nonuniform decode writes new K", k_cache[1], k_new)
    expect_tensor("nonuniform decode writes new V", v_cache[1], v_new)


def check_10() -> None:
    """Check causal verification, window limits, group mapping, and no mutation."""
    q = torch.zeros(3, 4, 4)
    k_past = torch.zeros(2, 2, 4)
    v_past = torch.tensor([[[0.0], [100.0]], [[2.0], [120.0]]])
    k_new = torch.zeros(3, 2, 4)
    v_new = torch.tensor([[[4.0], [140.0]], [[6.0], [160.0]], [[8.0], [180.0]]])
    inputs = (q, k_past, v_past, k_new, v_new)
    snapshots = tuple(x.clone() for x in inputs)
    expect_tensor(
        "verify block sees old tokens and only its causal new prefix",
        exercises.verify_block(*inputs),
        torch.tensor([[[2.0], [2.0], [120.0], [120.0]],
                      [[3.0], [3.0], [130.0], [130.0]],
                      [[4.0], [4.0], [140.0], [140.0]]]),
    )
    expect_tensor(
        "verify block respects a two-token window",
        exercises.verify_block(*inputs, window=2),
        torch.tensor([[[3.0], [3.0], [130.0], [130.0]],
                      [[5.0], [5.0], [150.0], [150.0]],
                      [[7.0], [7.0], [170.0], [170.0]]]),
    )
    for name, actual, before in zip(("Q", "past K", "past V", "new K", "new V"),
                                    inputs, snapshots):
        expect_tensor(f"verify must not mutate {name}", actual, before)

    q = torch.zeros(2, 4, 4)
    q[1, :, 0] = torch.tensor([2.0, -2.0, 2.0, -2.0])
    k_new = torch.zeros(2, 2, 4)
    k_new[1, :, 0] = 1.0
    v_new = torch.tensor([[[2.0], [20.0]], [[10.0], [100.0]]])
    e = math.e
    expect_tensor(
        "verify no-past case uses scaled scores independently per head",
        exercises.verify_block(q, torch.empty(0, 2, 4), torch.empty(0, 2, 1),
                               k_new, v_new),
        torch.tensor([[[2.0], [2.0], [20.0], [20.0]],
                      [[(2 + 10*e)/(1 + e)], [(2*e + 10)/(1 + e)],
                       [(20 + 100*e)/(1 + e)], [(20*e + 100)/(1 + e)]]]),
    )


def check_11() -> None:
    """Check accepted-prefix commit, zero acceptance, and untouched cache rows."""
    k_cache = torch.full((6, 2, 2), -9.0)
    v_cache = torch.full((6, 2, 1), -8.0)
    k_cache[:2] = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
                                [[5.0, 6.0], [7.0, 8.0]]])
    v_cache[:2] = torch.tensor([[[10.0], [20.0]], [[30.0], [40.0]]])
    old_k, old_v = k_cache[:2].clone(), v_cache[:2].clone()
    k_candidates = torch.arange(12.0).reshape(3, 2, 2) + 20
    v_candidates = torch.arange(6.0).reshape(3, 2, 1) + 50
    used = exercises.commit_accepted_prefix(
        k_cache, v_cache, 2, k_candidates, v_candidates, accepted_count=2
    )
    assert used == 4, f"accepted prefix: expected used=4, got {used}"
    expect_tensor("commit preserves old K", k_cache[:2], old_k)
    expect_tensor("commit preserves old V", v_cache[:2], old_v)
    expect_tensor("commit writes accepted K only", k_cache[2:4], k_candidates[:2])
    expect_tensor("commit writes accepted V only", v_cache[2:4], v_candidates[:2])
    expect_tensor("commit excludes rejected K", k_cache[4:], torch.full((2, 2, 2), -9.0))
    expect_tensor("commit excludes rejected V", v_cache[4:], torch.full((2, 2, 1), -8.0))

    before_k, before_v = k_cache.clone(), v_cache.clone()
    used = exercises.commit_accepted_prefix(
        k_cache, v_cache, used, k_candidates, v_candidates, accepted_count=0
    )
    assert used == 4, f"zero accepted: expected used=4, got {used}"
    expect_tensor("zero acceptance leaves K unchanged", k_cache, before_k)
    expect_tensor("zero acceptance leaves V unchanged", v_cache, before_v)

    all_k = torch.full((4, 2, 2), -9.0)
    all_v = torch.full((4, 2, 1), -8.0)
    used = exercises.commit_accepted_prefix(
        all_k, all_v, 0, k_candidates, v_candidates, accepted_count=3
    )
    assert used == 3, f"all accepted: expected used=3, got {used}"
    expect_tensor("all candidates become visible K", all_k[:3], k_candidates)
    expect_tensor("all candidates become visible V", all_v[:3], v_candidates)
    expect_tensor("unused row stays untouched", all_k[3], torch.full((2, 2), -9.0))


def check_12() -> None:
    """Check causal GQA prefill over several chunk boundaries and nonuniform scores."""
    q = torch.zeros(5, 4, 4)
    k = torch.zeros(5, 2, 4)
    v = torch.tensor([[[0.0], [100.0]], [[2.0], [120.0]],
                      [[4.0], [140.0]], [[6.0], [160.0]], [[8.0], [180.0]]])
    expected = torch.tensor([
        [[0.0], [0.0], [100.0], [100.0]],
        [[1.0], [1.0], [110.0], [110.0]],
        [[2.0], [2.0], [120.0], [120.0]],
        [[3.0], [3.0], [130.0], [130.0]],
        [[4.0], [4.0], [140.0], [140.0]],
    ])
    for chunk_size in (1, 2, 3, 5, 8):
        expect_tensor(
            f"causal prefill with chunk_size={chunk_size}",
            exercises.chunked_prefill(q, k, v, chunk_size), expected,
        )

    q = torch.zeros(3, 4, 4)
    q[2, :, 0] = torch.tensor([2.0, -2.0, 2.0, -2.0])
    k = torch.zeros(3, 2, 4)
    k[2, :, 0] = 1.0
    v = torch.tensor([[[0.0], [100.0]], [[2.0], [120.0]], [[10.0], [0.0]]])
    e = math.e
    expected = torch.tensor([
        [[0.0], [0.0], [100.0], [100.0]],
        [[1.0], [1.0], [110.0], [110.0]],
        [[(2 + 10*e)/(2 + e)], [(2 + 10/e)/(2 + 1/e)],
         [220/(2 + e)], [220/(2 + 1/e)]],
    ])
    expect_tensor(
        "later chunk uses its own Q-K scores and all earlier keys",
        exercises.chunked_prefill(q, k, v, chunk_size=2), expected,
    )


def check_13() -> None:
    """Check independent request lengths, ignored padding, and GQA head mapping."""
    q = torch.zeros(2, 4, 4)
    k = torch.zeros(2, 4, 2, 4)
    v = torch.tensor([
        [[[0.0], [100.0]], [[2.0], [120.0]],
         [[999.0], [999.0]], [[999.0], [999.0]]],
        [[[10.0], [200.0]], [[20.0], [220.0]],
         [[30.0], [240.0]], [[40.0], [260.0]]],
    ])
    expect_tensor(
        "different request lengths do not expose padding or other requests",
        exercises.batched_gqa_decode(q, k, v, torch.tensor([2, 4])),
        torch.tensor([[[1.0], [1.0], [110.0], [110.0]],
                      [[25.0], [25.0], [230.0], [230.0]]]),
    )

    q = torch.tensor([[[2.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0],
                       [2.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0]]])
    k = torch.zeros(1, 2, 2, 4)
    k[0, 1, :, 0] = 1.0
    v = torch.tensor([[[[0.0], [100.0]], [[10.0], [0.0]]]])
    expect_tensor(
        "batched decode keeps independent Q-head softmax distributions",
        exercises.batched_gqa_decode(q, k, v, torch.tensor([2])),
        torch.tensor([[[7.310586], [2.689414], [26.89414], [73.10586]]]),
    )


def check_14() -> None:
    """Check shuffled page tables, page boundaries, and page size one."""
    rows = exercises.paged_cache_rows(
        torch.tensor([2, 0, 3]), torch.tensor([0, 3, 4, 7, 8, 10]), page_size=4
    )
    assert rows.dtype in (torch.int32, torch.int64), "cache row indices must be integers"
    expect_tensor("logical positions map through shuffled pages", rows,
                  torch.tensor([8, 11, 0, 3, 12, 14], dtype=rows.dtype))
    one_rows = exercises.paged_cache_rows(
        torch.tensor([3, 1, 0]), torch.tensor([0, 1, 2]), 1
    )
    assert one_rows.dtype in (torch.int32, torch.int64), "cache row indices must be integers"
    expect_tensor(
        "page_size=1 still uses the page table",
        one_rows, torch.tensor([3, 1, 0], dtype=one_rows.dtype),
    )


def check_15() -> None:
    """Check paged K/V reads in logical order without mutating storage."""
    k_pages = torch.arange(24.0).reshape(3, 2, 2, 2)
    v_pages = torch.arange(12.0).reshape(3, 2, 2, 1) + 100
    page_table = torch.tensor([2, 0, 1])
    before_k, before_v = k_pages.clone(), v_pages.clone()
    actual = exercises.read_paged_kv(k_pages, v_pages, page_table, length=5)
    assert isinstance(actual, tuple) and len(actual) == 2, "paged read must return (K, V)"
    k, v = actual
    expect_tensor("paged K follows logical token order", k,
                  torch.stack([k_pages[2, 0], k_pages[2, 1], k_pages[0, 0],
                               k_pages[0, 1], k_pages[1, 0]]))
    expect_tensor("paged V follows logical token order", v,
                  torch.stack([v_pages[2, 0], v_pages[2, 1], v_pages[0, 0],
                               v_pages[0, 1], v_pages[1, 0]]))
    expect_tensor("paged read preserves K storage", k_pages, before_k)
    expect_tensor("paged read preserves V storage", v_pages, before_v)

    k, v = exercises.read_paged_kv(k_pages, v_pages, page_table, length=1)
    expect_tensor("one-token read K", k, k_pages[2, :1])
    expect_tensor("one-token read V", v, v_pages[2, :1])


def check_16() -> None:
    """Check paged writes across page boundaries and untouched physical rows."""
    k_pages = torch.full((3, 2, 2, 2), -9.0)
    v_pages = torch.full((3, 2, 2, 1), -8.0)
    page_table = torch.tensor([2, 0, 1])
    k_new = torch.arange(12.0).reshape(3, 2, 2) + 1
    v_new = torch.arange(6.0).reshape(3, 2, 1) + 20
    expected_k, expected_v = k_pages.clone(), v_pages.clone()
    expected_k[2, 1], expected_k[0, 0], expected_k[0, 1] = k_new
    expected_v[2, 1], expected_v[0, 0], expected_v[0, 1] = v_new
    result = exercises.write_paged_kv(k_pages, v_pages, page_table, 1, k_new, v_new)
    assert result is None, "paged write should return None"
    expect_tensor("cross-page K write and untouched rows", k_pages, expected_k)
    expect_tensor("cross-page V write and untouched rows", v_pages, expected_v)

    one_k = torch.full((3, 1, 2, 2), -9.0)
    one_v = torch.full((3, 1, 2, 1), -8.0)
    page_table = torch.tensor([2, 0, 1])
    exercises.write_paged_kv(one_k, one_v, page_table, 1, k_new[:2], v_new[:2])
    expect_tensor("page_size=1 K writes", one_k[0, 0], k_new[0])
    expect_tensor("page_size=1 V writes", one_v[0, 0], v_new[0])
    expect_tensor("page_size=1 second K write", one_k[1, 0], k_new[1])
    expect_tensor("page_size=1 second V write", one_v[1, 0], v_new[1])
    expect_tensor("page_size=1 untouched K page", one_k[2], torch.full((1, 2, 2), -9.0))


def check_17() -> None:
    """Check adjacent-pair rotation, per-token positions, heads, and theta."""
    x = torch.zeros(2, 2, 2, 4)
    x[:, :, 0, 0] = x[:, :, 0, 2] = 1.0
    x[:, :, 1, 1] = x[:, :, 1, 3] = 1.0
    positions = torch.tensor([[0, 1], [2, 5]])
    expected = torch.empty_like(x)
    for b in range(2):
        for l in range(2):
            p = positions[b, l].item()
            c0, s0 = math.cos(p), math.sin(p)
            c1, s1 = math.cos(p / 100), math.sin(p / 100)
            expected[b, l, 0] = torch.tensor([c0, s0, c1, s1])
            expected[b, l, 1] = torch.tensor([-s0, c0, -s1, c1])
    actual = exercises.rope_interleaved(x, positions, theta=10000.0)
    expect_tensor("RoPE uses each token position and each adjacent pair", actual, expected)

    x = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
    expect_tensor(
        "theta changes the second pair's frequency",
        exercises.rope_interleaved(x, torch.tensor([[1]]), theta=16.0),
        torch.tensor([[[[math.cos(1), math.sin(1), math.cos(0.25), math.sin(0.25)]]]]),
    )


def check_18a() -> None:
    """Check per-head int8 scales, zero heads, range, and reconstruction error."""
    x = torch.zeros(3, 3, 2)
    x[:, 0] = torch.tensor([[-1.0, 0.0], [0.5, 1.0], [0.0, 0.0]])
    x[:, 1] = torch.tensor([[-4.0, 0.0], [2.0, 4.0], [0.0, 0.0]])
    original = x.clone()
    actual = exercises.quantize_kv_int8_per_head(x)
    assert isinstance(actual, tuple) and len(actual) == 2, "quantize must return (codes, scales)"
    codes, scales = actual
    assert isinstance(codes, torch.Tensor) and codes.shape == x.shape, "codes need shape [S,Hkv,D]"
    assert codes.dtype == torch.int8, "codes must be torch.int8"
    assert isinstance(scales, torch.Tensor) and scales.shape == (3,), "scales need shape [Hkv]"
    assert torch.is_floating_point(scales) and torch.isfinite(scales).all(), "scales must be finite floats"
    assert (scales > 0).all(), "even an all-zero head needs a positive scale"
    assert codes.min().item() >= -127 and codes.max().item() <= 127, "codes must lie in [-127,127]"
    expect_tensor("nonzero heads use their own maximum magnitude", scales[:2],
                  torch.tensor([1 / 127, 4 / 127], dtype=scales.dtype))
    expect_tensor("all-zero head has zero codes", codes[:, 2], torch.zeros((3, 2), dtype=torch.int8))
    assert codes[0, 0, 0].item() == -127 and codes[1, 0, 1].item() == 127
    assert codes[0, 1, 0].item() == -127 and codes[1, 1, 1].item() == 127
    reconstructed = codes.float() * scales[None, :, None]
    error = (reconstructed[:, :2] - original[:, :2]).abs()
    assert (error <= scales[None, :2, None] / 2 + 1e-6).all(), "quantization error exceeds half a step"


def check_18b() -> None:
    """Check code-to-float reconstruction with different head scales."""
    codes = torch.tensor([[[-2, 4], [3, -1]], [[0, 127], [-127, 2]]], dtype=torch.int8)
    scales = torch.tensor([0.5, 2.0])
    actual = exercises.dequantize_kv_int8_per_head(codes, scales)
    assert isinstance(actual, torch.Tensor) and torch.is_floating_point(actual), "dequantized values must be float"
    expect_tensor("each KV head uses its own scale", actual,
                  torch.tensor([[[-1.0, 2.0], [6.0, -2.0]],
                                [[0.0, 63.5], [-254.0, 4.0]]]))


def check_19() -> None:
    """Check both K and V, all layers, zero tokens, and a large cache."""
    cases = [
        ((2, 3, 4, 8, 2), 768),
        ((80, 8192, 8, 128, 1), 1_342_177_280),
        ((4, 0, 2, 64, 2), 0),
        ((1, 100, 1, 64, 1), 12_800),
    ]
    for arguments, expected in cases:
        actual = exercises.kv_cache_bytes(*arguments)
        assert isinstance(actual, int) and not isinstance(actual, bool), "cache bytes must be an int"
        assert actual == expected, f"cache bytes {arguments}: expected {expected}, got {actual}"


def check_20() -> None:
    """Check CPU timing units, warmup/repeat counts, and finite median latency."""
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1

    latency = exercises.median_latency_ms(operation, warmup=2, repeats=5, device="cpu")
    assert calls == 7, f"expected 2 warmups + 5 timed calls, got {calls}"
    assert isinstance(latency, (int, float)) and math.isfinite(latency) and latency >= 0, (
        f"median latency must be a finite number of milliseconds, got {latency}"
    )

    slow_calls = 0

    def slow_operation() -> None:
        nonlocal slow_calls
        slow_calls += 1
        time.sleep(0.004)

    slow_ms = exercises.median_latency_ms(slow_operation, warmup=1, repeats=5, device="cpu")
    assert slow_calls == 6, f"expected 1 warmup + 5 timed calls, got {slow_calls}"
    assert isinstance(slow_ms, (int, float)) and 1 < slow_ms < 1000, (
        f"a 4 ms operation should report milliseconds, got {slow_ms}"
    )


def main() -> None:
    checks = {
        "01": check_01,
        "02": check_02,
        "03": check_03,
        "04": check_04,
        "04a": check_04a,
        "05": check_05,
        "06": check_06,
        "07": check_07,
        "07b": check_07b,
        "08": check_08,
        "09": check_09,
        "10": check_10,
        "11": check_11,
        "12": check_12,
        "13": check_13,
        "14": check_14,
        "15": check_15,
        "16": check_16,
        "17": check_17,
        "18a": check_18a,
        "18b": check_18b,
        "19": check_19,
        "20": check_20,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exercise", choices=[*checks, "all"])
    args = parser.parse_args()
    selected = checks if args.exercise == "all" else {args.exercise: checks[args.exercise]}
    failed = False
    for number, check in selected.items():
        try:
            check()
        except Exception as error:
            failed = True
            print(f"FAIL exercise {number}: {type(error).__name__}: {error}")
        else:
            print(f"PASS exercise {number}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
