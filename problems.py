"""Part 1: inference engineering exercises in plain PyTorch.

Implement each function yourself. The docstrings give the contract, not a solution.
Use small float32 tensors on CPU first. No model weights or CUDA kernels are needed.
See README.md for the order and the later lab outline.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


# 01-07a: attention math. No batch axis yet.


# Starter functions for the public workbook. Worked solutions stay in ignored part1_pytorch.py.

def qk_scores_one_head(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """01. Compute scaled Q-K scores for one head.

    q: [L, D] new query vectors; k: [S, D] key vectors.

    Return: [L, S]. Each entry compares one query with one key.
    Scale each dot product by 1 / sqrt(D). No masking or softmax yet.
    """
    raise NotImplementedError("exercise 01")


def weighted_values(weights: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """02. Turn attention weights into output vectors.

    weights: [L, S], already normalized across S; v: [S, Dv].

    Return: [L, Dv]. For each query, sum the S value vectors using
    that query's S scalar weights. There is no head or batch axis yet.

    Try L=2, S=3, Dv=2.

    weights
                   S
          tok0 tok1 tok2 tok3 tok4
    tok2  0.1  0.2  0.7  0.0  0.0   <- softmaxed
  L tok3  ...
    tok4

    v
               dv
    tok0  [1, 4, 3, 5, 4, 8, ...]
  S tok1  ...
    tok2
    tok3
    tok4

    """
    raise NotImplementedError("exercise 02")


def causal_window_mask(
    past_len: int,
    q_len: int,
    window: int | None = None,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """03. Decide which keys each new query token may see.

    There are past_len old keys followed by q_len new keys, so S =
    past_len + q_len.

    Return a bool mask [L=q_len, S]. True means visible.

    Query j is at absolute position past_len+j and sees keys through itself.
    If window is set, it sees at most window keys, including itself.

    Example: past_len=2, q_len=3, window=None gives rows with 3, 4, 5
    visible keys. Avoid a row with zero visible keys.
    """
    raise NotImplementedError("exercise 03")


def causal_attention_one_head(
    q_new: torch.Tensor,
    k_all: torch.Tensor,
    v_all: torch.Tensor,
    past_len: int,
    window: int | None = None,
) -> torch.Tensor:
    """04. Compose scores, causal masking, softmax, and weighted V.

    q_new: [L, D]; k_all: [S, D]; v_all: [S, Dv], where S=past_len+L.

    Return: [L, Dv]. Mask invisible scores before softmax across S.
    The first new query can see its own new key, but not later new keys.

    Available helpers: qk_scores_one_head (01), weighted_values (02),
    and causal_window_mask (03). You may call them here.

    Try past_len=2, L=3, D=4, Dv=2.
    """
    raise NotImplementedError("exercise 04")


def predict_next_token_single_head(
    q_new: torch.Tensor,
    k_all: torch.Tensor,
    v_all: torch.Tensor,
    w_vocab: torch.Tensor,
    past_len: int,
    window: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """04a. Turn one head's attention output into next-token probabilities.

    This is a tiny final-layer decoder: the supplied Q/K/V already come
    from the model, and attention's output acts as its final hidden vector.

    q_new: [1, D]; k_all: [S, D]; v_all: [S, Dv], S=past_len+1.
    w_vocab: [Dv, V], where V is the vocabulary size.

    Use causal attention, then project [Dv] to V vocabulary logits.
    Softmax over V, not over the S cached tokens. Return (probabilities [V],
    greedy_next_token_id []), where the ID is a scalar torch.long tensor.
    Vocabulary indices stand for tokens; no tokenizer is needed.

    Available helper: causal_attention_one_head (04).

    Try S=2, D=4, Dv=2, V=3 and write down both softmax axes.
    """
    """
    a: [L, dv]


                   dv
          [1, 4, 3, 5, 4, 8, ...]
          ...
    tok2
  L tok3
    tok4

    w_vocab: [dV, V] where V is the vocab size

               dv
          [1, 4, 3, 5, 4, 8, ...]
           ...
    tok_a
  V tok_b
    tok_c
    ...

    The dot product tells you how close the dv vector for each token l is to each token in the vocab.

    Then we use softmax to get probabilities.
    """
    raise NotImplementedError("exercise 04a")


def multi_head_attention(
    q_new: torch.Tensor,
    k_all: torch.Tensor,
    v_all: torch.Tensor,
    past_len: int,
    window: int | None = None,
) -> torch.Tensor:
    """05. Extend causal attention to ordinary multi-head attention.

    q_new: [L, H, D]; k_all: [S, H, D]; v_all: [S, H, Dv].

    Return: [L, H, Dv]. Query head h uses only K/V head h.
    Normalize each head's scores over S, never over H.

    Available helpers: causal_attention_one_head (04), if you loop over
    heads, and causal_window_mask (03), if you work on all heads at once.

    Start with a loop over H if that makes the mapping easier to see.
    Try past_len=2, L=2, S=4, H=2, D=4, Dv=3.
    """
    """
    There are H of these

                       S
          tok0 tok1 tok2 tok3 tok4
    tok2  0.1  0.2  0.7  0.0  0.0
  L tok3  ...
    tok4
    """
    """
    For a given l and s, the value from mask is broadcasted to all h in H.
    """
    raise NotImplementedError("exercise 05")


def multi_query_attention(
    q_new: torch.Tensor,
    k_all: torch.Tensor,
    v_all: torch.Tensor,
    past_len: int,
    window: int | None = None,
) -> torch.Tensor:
    """06. Implement MQA: many Q heads share one K/V head.

    q_new: [L, Hq, D]; k_all: [S, 1, D]; v_all: [S, 1, Dv].

    Return: [L, Hq, Dv]. Every query head has its own scores and
    softmax distribution, even though all query heads read the same K/V.

    Available helpers: causal_attention_one_head (04), if you loop over
    query heads, and causal_window_mask (03) for a vectorized version.

    Try past_len=2, L=2, S=4, Hq=3, D=4, Dv=3.
    """
    raise NotImplementedError("exercise 06")


def grouped_query_attention(
    q_new: torch.Tensor,
    k_all: torch.Tensor,
    v_all: torch.Tensor,
    past_len: int,
    window: int | None = None,
) -> torch.Tensor:
    """07a. Implement GQA without mixing heads or requests.

    q_new: [L, Hq, D]; k_all: [S, Hkv, D]; v_all: [S, Hkv, Dv].

    Return: [L, Hq, Dv]. Require Hq to be divisible by Hkv.
    Contiguous groups of Hq/Hkv query heads use one K/V head each.
    Check both Hq=4,Hkv=2 and Hq=16,Hkv=2. MHA and MQA are the
    Hq=Hkv and Hkv=1 edge cases of the same head-sharing rule.

    Available helpers: causal_attention_one_head (04), if you loop over
    query heads, and causal_window_mask (03) for a vectorized version.

    For a first run, use past_len=2, L=2, S=4, Hq=4, Hkv=2, D=4.
    """
    """
    Scores:

    There are Hkv * q_heads_per_kv_head of these

                       S
          tok0 tok1 tok2 tok3 tok4
    tok2  0.1  0.2  0.7  0.0  0.0
  L tok3  ...
    tok4
    """
    raise NotImplementedError("exercise 07a")


def predict_next_token_gqa(
    q_new: torch.Tensor,
    k_all: torch.Tensor,
    v_all: torch.Tensor,
    w_o: torch.Tensor,
    residual: torch.Tensor,
    w_vocab: torch.Tensor,
    past_len: int,
    window: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """07b. Predict a token after a final GQA attention layer.

    q_new: [L, Hq, D]; k_all: [S, Hkv, D]; v_all: [S, Hkv, Dv].
    w_o: [Hq*Dv, D_model]; residual: [L, D_model];
    w_vocab: [D_model, V], with S=past_len+L.

    Compute causal GQA, flatten its Hq heads into [L, Hq*Dv], apply the
    attention output projection, and add the residual for each token.
    Treat the result as final hidden vectors [L, D_model], project to
    vocabulary logits [L, V], then return the last token's
    (probabilities [V], greedy_next_token_id []). The ID is
    a scalar torch.long tensor. This toy final layer omits FFN and final
    normalization so you can focus on the shape change and vocab softmax.

    Available helper: grouped_query_attention (07a).

    Try Hq=4, Hkv=2, D=4, Dv=1, D_model=3, V=3.
    """
    """
    Purpose of output projection w_o
    - converts back to dmodel so we can add the residual
    - learns to combine information across different heads
    """
    """
    Why are dmodel, d, and dv different?

    D: width of each Q,K head. Q and K need the same width for the dot product.
    Dv: width of each V head. The information returned by attention. This is a different purpose than queries.
    Dmodel: Wdith of the models hidden / learned state - ex. Wq, Wk, Wv, Wo, etc.
    """
    raise NotImplementedError("exercise 07b")


def append_contiguous_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    used: int,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
) -> int:
    """08. Append new K/V rows to a preallocated cache in place.

    k_cache: [capacity, Hkv, D]; v_cache: [capacity, Hkv, Dv].
    k_new: [L, Hkv, D]; v_new: [L, Hkv, Dv].

    Rows [0:used] belong to old tokens and must remain unchanged.
    Write the L new rows starting at used; return the new used length.
    Decide what to do if the cache lacks capacity.
    """
    raise NotImplementedError("exercise 08")


def decode_one_token(
    q_new: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    used: int,
    window: int | None = None,
) -> tuple[torch.Tensor, int]:
    """09. Perform one GQA decode step with a contiguous KV cache.

    q_new: [Hq, D]; k_new: [Hkv, D]; v_new: [Hkv, Dv].

    Cache shapes are as in 08. Append this token's K/V, attend to the
    permitted old keys plus the new key, and return ([Hq, Dv], used+1).
    The old cache rows must not be recomputed or overwritten.

    Available helpers: append_contiguous_kv (08) and
    grouped_query_attention (07a).
    """
    raise NotImplementedError("exercise 09")


def verify_block(
    q_new: torch.Tensor,
    k_past: torch.Tensor,
    v_past: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    window: int | None = None,
) -> torch.Tensor:
    """10. Verify L distinct draft tokens in one causal attention pass.

    q_new: [L, Hq, D]; k_past: [S_old, Hkv, D];
    v_past: [S_old, Hkv, Dv]; k_new: [L, Hkv, D];
    v_new: [L, Hkv, Dv].

    Return: [L, Hq, Dv].
    Query j sees old keys and new keys 0..j, never j+1..L-1.
    Do not mutate a cache here; acceptance is decided outside this step.

    Available helper: grouped_query_attention (07a).
    """
    raise NotImplementedError("exercise 10")


def commit_accepted_prefix(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    used: int,
    k_candidates: torch.Tensor,
    v_candidates: torch.Tensor,
    accepted_count: int,
) -> int:
    """11. Retain only an accepted prefix of speculative tokens.

    k_candidates/v_candidates contain L target-model K/V rows computed
    while verifying the draft's candidate tokens. The external verifier
    has accepted the first accepted_count tokens, where 0 <= count <= L.

    Append only those rows; leave rejected suffix rows out of the cache.
    Return used+accepted_count. Existing cache rows are immutable.

    Available helper: append_contiguous_kv (08).
    """
    raise NotImplementedError("exercise 11")


def chunked_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """12. Compute full causal GQA prefill in consecutive chunks.

    q: [T, Hq, D]; k: [T, Hkv, D]; v: [T, Hkv, Dv].

    Return: [T, Hq, Dv]. Process queries in chunks of at most chunk_size.
    A query in a later chunk sees all prior chunks and its causal prefix
    inside the current chunk. The answer should equal unchunked causal
    attention, aside from normal floating-point rounding.

    Available helper: grouped_query_attention (07a).
    """
    raise NotImplementedError("exercise 12")


def batched_gqa_decode(
    q_new: torch.Tensor,
    k_all: torch.Tensor,
    v_all: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """13. Decode one token for each of B independent requests.

    q_new: [B, Hq, D]; k_all: [B, Smax, Hkv, D];
    v_all: [B, Smax, Hkv, Dv]; lengths: [B].

    Return: [B, Hq, Dv]. For request b, valid keys are positions
    0..lengths[b]-1, and the last valid key is its new token.
    Padding and other requests must not affect its softmax.
    No cache mutation is needed in this exercise.

    Available helper: grouped_query_attention (07a), called for each
    request's valid prefix if you choose a loop.
    """
    raise NotImplementedError("exercise 13")


def paged_cache_rows(
    page_table: torch.Tensor,
    logical_positions: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """14. Map logical token positions to flat physical cache rows.

    page_table: [max_pages], giving a physical page for each logical page.
    logical_positions: [N], valid positions for one request.

    Return: [N] flat row indices into a [physical_pages*page_size, ...]
    cache view.

    Test positions on both sides of a page boundary and a
    shuffled page table. Do not assume physical pages are contiguous.
    """
    raise NotImplementedError("exercise 14")


def read_paged_kv(
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    page_table: torch.Tensor,
    length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """15. Gather one request's logical K/V sequence from pages.

    k_pages: [physical_pages, page_size, Hkv, D];
    v_pages: [physical_pages, page_size, Hkv, Dv].
    page_table: [max_pages].

    Return logical K [length,Hkv,D] and
    V [length,Hkv,Dv] in token order, even if physical pages are shuffled.
    Read only positions 0..length-1; page size comes from the cache shape.

    Available helper: paged_cache_rows (14).
    """
    raise NotImplementedError("exercise 15")


def write_paged_kv(
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    page_table: torch.Tensor,
    start_position: int,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
) -> None:
    """16. Write L new logical K/V positions through a page table.

    k_new: [L,Hkv,D]; v_new: [L,Hkv,Dv].

    Mutate k_pages and v_pages at logical positions
    start_position..start_position+L-1.
    A write may cross one or more page boundaries. Preserve all other
    physical cache rows. Eviction and page allocation are out of scope.

    Available helper: paged_cache_rows (14).
    """
    raise NotImplementedError("exercise 16")


def rope_interleaved(
    x: torch.Tensor,
    positions: torch.Tensor,
    theta: float,
) -> torch.Tensor:
    """17. Rotate adjacent pairs within every Q or K head.

    x: [B,L,H,D] with even D; positions: [B,L].

    Return x's shape.
    Pair i is coordinates (2*i,2*i+1), with angle at position p equal
    to p / theta^(2*i/D). Both coordinates use the same angle.
    Reuse each token's angles across heads. Do not rotate V.
    """
    raise NotImplementedError("exercise 17")


def quantize_kv_int8_per_head(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """18a. Toy cache quantization, not the challenge's FP8 format.

    x: float [S,Hkv,D].

    Return quantized int8 [S,Hkv,D] and positive scale [Hkv].
    Use one symmetric scale per KV head so each nonzero head's largest
    magnitude maps to 127. Keep codes in [-127,127]. Handle an all-zero
    head without division by zero.
    Work on CPU first; this teaches scale ownership and error analysis.
    """
    raise NotImplementedError("exercise 18a")


def dequantize_kv_int8_per_head(
    codes: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """18b. Reconstruct approximate float K/V values.

    codes: int8 [S,Hkv,D]; scales: [Hkv].

    Return float [S,Hkv,D].
    Compare the result with the original values and measure error.
    """
    raise NotImplementedError("exercise 18b")


def kv_cache_bytes(
    num_layers: int,
    allocated_tokens: int,
    hkv: int,
    head_dim: int,
    bytes_per_value: int,
) -> int:
    """19. Estimate K+V cache payload size, excluding allocator overhead.

    allocated_tokens is across all requests and allocated pages.

    Count both K and V for every layer. Compare MHA, GQA, and MQA
    while holding query-head count fixed. This one is arithmetic, not
    a tensor operation; write down the units at each multiplication.
    """
    raise NotImplementedError("exercise 19")


def median_latency_ms(
    operation: Callable[[], object],
    warmup: int = 10,
    repeats: int = 50,
    device: torch.device | str = "cpu",
) -> float:
    """20. Build a small, honest PyTorch inference microbenchmark.

    Warm up, time repeated calls, and return median milliseconds.

    Synchronize the device around each measurement if it is async
    (CUDA or MPS); CPU needs no device synchronization. Do not include
    input creation in the measured operation. Compare attention at
    several S values and report latency together with shapes and dtype.
    Mac CPU is enough for methodology; MPS results are optional and
    should not be treated as NVIDIA CUDA performance predictions.
    """
    raise NotImplementedError("exercise 20")
