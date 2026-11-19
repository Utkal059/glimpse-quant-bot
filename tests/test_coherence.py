"""Cross-topic refinement, Frank-Wolfe Bregman projection and buy-only cover arbitrage."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from glimpse_quant_bot.coherence import (
    cover_arbitrage,
    frank_wolfe_projection,
    linked_groups,
    refine,
)
from glimpse_quant_bot.lmsr import LmsrBook, book_from_prices
from glimpse_quant_bot.types import AssetType, Bucket, TopicMeta

END = 1_800_003_600


def make_topic(topic_id: int, width: float, asset: AssetType = AssetType.BTC) -> TopicMeta:
    edges = np.arange(1000.0, 2000.0 + width / 2, width)
    buckets = tuple(
        Bucket(topic_id * 1000 + i, f"{lo:.0f}-{hi:.0f}", float(lo), float(hi))
        for i, (lo, hi) in enumerate(itertools.pairwise(edges))
    )
    return TopicMeta(topic_id, f"batch{topic_id}", "t", asset, END, buckets)


def make_book(topic: TopicMeta, probabilities: np.ndarray, b: float = 30.0) -> LmsrBook:
    return book_from_prices(
        topic.topic_id,
        [bucket.option_id for bucket in topic.buckets],
        list(100.0 * probabilities),
        liquidity_b=b,
        alpha=0.02,
    )


FINE = np.array([0.02, 0.03, 0.05, 0.10, 0.30, 0.25, 0.12, 0.07, 0.04, 0.02])


def coarse_of(fine: np.ndarray) -> np.ndarray:
    return fine.reshape(-1, 2).sum(axis=1)


def test_refinement_and_groups() -> None:
    hourly, daily = make_topic(1, 100.0), make_topic(2, 200.0)
    refinement = refine([hourly, daily])
    assert refinement.n_cells == 10
    assert refinement.membership[0].shape == (10, 10)
    assert np.array_equal(refinement.membership[0], np.eye(10, dtype=bool))
    assert refinement.membership[1].sum(axis=1).tolist() == [2] * 5
    other = make_topic(3, 100.0, AssetType.ETH)
    groups = linked_groups({1: hourly, 2: daily, 3: other})
    assert groups == [[1, 2]]


def test_coherent_books_have_no_divergence_and_no_cover() -> None:
    hourly, daily = make_topic(1, 100.0), make_topic(2, 200.0)
    books = [make_book(hourly, FINE), make_book(daily, coarse_of(FINE))]
    projection = frank_wolfe_projection([hourly, daily], books)
    assert projection.divergence_points < 1e-3
    np.testing.assert_allclose(projection.cells, FINE, atol=2e-3)
    assert cover_arbitrage([hourly, daily], books, fee_rate=0.02, max_cost_points=1e5) is None


def test_incoherent_books_yield_an_exact_cover() -> None:
    hourly, daily = make_topic(1, 100.0), make_topic(2, 200.0)
    cheap = coarse_of(FINE).copy()
    cheap[2] = 0.08  # daily prices 1400-1600 at 8% while hourly says 55%
    cheap = cheap / cheap.sum()
    books = [make_book(hourly, FINE), make_book(daily, cheap)]
    projection = frank_wolfe_projection([hourly, daily], books)
    assert projection.divergence_points > 1.0
    assert projection.gap < 1e-3 * projection.divergence_points + 1e-6

    arb = cover_arbitrage([hourly, daily], books, fee_rate=0.02, max_cost_points=1e5)
    assert arb is not None
    assert set(arb.legs) == {1, 2}
    assert arb.unit_cost_points < 98.0
    # Every refinement cell pays at least `scale` contracts.
    refinement = refine([hourly, daily])
    payoff = np.zeros(refinement.n_cells)
    for t, (topic, book) in enumerate(zip([hourly, daily], books, strict=True)):
        held = dict(arb.legs.get(topic.topic_id, ()))
        x = np.array([held.get(o, 0.0) for o in book.option_ids])
        payoff += refinement.membership[t].T.astype(float) @ x
    assert float(np.min(payoff)) >= arb.scale - 1e-9
    # All-in is the exact LS-LMSR cost plus commission, group by group.
    exact = 0.0
    for book in books:
        legs = list(arb.legs[book.topic_id])
        raw = int(book.buy_cost(legs)) / 1000.0
        exact += raw + max(1.0, 0.02 * raw)
    assert arb.all_in_points == pytest.approx(exact, abs=0.01)
    assert arb.guaranteed_profit_points == pytest.approx(98.0 * arb.scale - exact, abs=0.01)
    assert arb.guaranteed_profit_points > 0.0


def test_cover_respects_the_budget() -> None:
    hourly, daily = make_topic(1, 100.0), make_topic(2, 200.0)
    cheap = coarse_of(FINE).copy()
    cheap[2] = 0.08
    books = [make_book(hourly, FINE), make_book(daily, cheap / cheap.sum())]
    small = cover_arbitrage([hourly, daily], books, fee_rate=0.02, max_cost_points=200.0)
    assert small is not None
    assert small.all_in_points <= 200.0 * 1.0 + 5.0  # rounding up to 0.01 contracts + fees


def test_uncovered_cells_mean_no_cover() -> None:
    hourly = make_topic(1, 100.0)
    shifted = TopicMeta(
        2,
        "b2",
        "t",
        AssetType.BTC,
        END,
        tuple(Bucket(2000 + i, "x", 3000.0 + 100 * i, 3100.0 + 100 * i) for i in range(5)),
    )
    books = [make_book(hourly, FINE), make_book(shifted, np.full(5, 0.2))]
    # The gap between 2000 and 3000 is a cell nobody pays in.
    assert cover_arbitrage([hourly, shifted], books, fee_rate=0.02, max_cost_points=1e5) is None
