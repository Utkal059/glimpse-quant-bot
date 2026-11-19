"""Multi-level OFI, its online drift model, VPIN with exact buy volume, toxicity scaling."""

from __future__ import annotations

import math

import numpy as np
import pytest

from glimpse_quant_bot.external_feeds import Candle, OrderBook
from glimpse_quant_bot.flow_toxicity import (
    NU_MIN,
    BookLevels,
    OfiState,
    OfiTracker,
    book_levels,
    level_ofi,
    ofi_drift,
    toxic_tail_index,
    toxic_variance_multiplier,
    toxicity_from_candles,
    vpin_series,
)
from glimpse_quant_bot.forecaster import microstructure_cdf


def levels(
    bid: float, ask: float, bid_sz: list[float], ask_sz: list[float], ts: float
) -> BookLevels:
    m = len(bid_sz)
    return BookLevels(
        bid_px=bid - np.arange(m, dtype=np.float64),
        bid_sz=np.asarray(bid_sz, dtype=np.float64),
        ask_px=ask + np.arange(m, dtype=np.float64),
        ask_sz=np.asarray(ask_sz, dtype=np.float64),
        ts=ts,
    )


def test_level_ofi_matches_cont_kukanov_stoikov() -> None:
    prev = levels(100.0, 101.0, [5.0, 5.0], [5.0, 5.0], 0.0)
    # Same prices, bid size +3 at L1, ask size +2 at L2.
    same = levels(100.0, 101.0, [8.0, 5.0], [5.0, 7.0], 1.0)
    np.testing.assert_allclose(level_ofi(prev, same), [3.0, -2.0])
    # Bid price ticks up: the whole new bid queue counts as buying pressure.
    up = levels(100.5, 101.0, [4.0, 4.0], [5.0, 5.0], 1.0)
    np.testing.assert_allclose(level_ofi(prev, up), [4.0, 4.0])
    # Ask price ticks up (ask depleted): the old ask queue counts as buying pressure.
    lifted = levels(100.0, 101.5, [5.0, 5.0], [6.0, 6.0], 1.0)
    np.testing.assert_allclose(level_ofi(prev, lifted), [5.0, 5.0])


def test_book_levels_requires_depth() -> None:
    book = OrderBook(bids=((100.0, 1.0),), asks=((101.0, 1.0),))
    assert book_levels(book, 0.0, levels=5) is None
    deep = OrderBook(
        bids=tuple((100.0 - i, 1.0 + i) for i in range(6)),
        asks=tuple((101.0 + i, 2.0 + i) for i in range(6)),
    )
    snapshot = book_levels(deep, 0.0, levels=5)
    assert snapshot is not None
    assert snapshot.mid == pytest.approx(100.5)
    assert snapshot.bid_sz.shape == (5,)


def simulate(beta_true: float, n: int, seed: int = 3) -> OfiTracker:
    """Books whose next mid move is ``beta_true * OFI`` plus noise."""
    rng = np.random.default_rng(seed)
    tracker = OfiTracker(levels=3, forgetting=0.999, ridge=1.0)
    mid = 100.0
    bid_sz = np.full(3, 10.0)
    ask_sz = np.full(3, 10.0)
    pending = 0.0
    for t in range(n):
        mid *= math.exp(pending + rng.normal(0.0, 2e-5))
        shock = rng.normal(0.0, 3.0, 3)
        bid_sz = np.maximum(bid_sz + shock, 1.0)
        snapshot = BookLevels(
            bid_px=np.full(3, mid - 0.5) - np.arange(3),
            bid_sz=bid_sz.copy(),
            ask_px=np.full(3, mid + 0.5) + np.arange(3),
            ask_sz=ask_sz.copy(),
            ts=float(10 * t),
        )
        tracker.update(snapshot)
        state = tracker.state(min_obs=1)
        pending = beta_true * (state.integrated if state is not None else 0.0)
    return tracker


def test_tracker_learns_a_predictive_ofi_coefficient() -> None:
    tracker = simulate(beta_true=2e-4, n=600)
    state = tracker.state()
    assert state is not None
    assert state.beta == pytest.approx(2e-4, rel=0.25)
    assert state.dt == pytest.approx(10.0)
    assert sum(abs(w) for w in state.weights) == pytest.approx(1.0)


def test_tracker_finds_no_signal_in_noise() -> None:
    state = simulate(beta_true=0.0, n=600).state()
    assert state is not None
    assert abs(state.beta) < 5e-5


def test_tracker_waits_for_data_and_resets_on_gaps() -> None:
    tracker = OfiTracker(levels=2)
    tracker.update(levels(100.0, 101.0, [5, 5], [5, 5], 0.0))
    tracker.update(levels(100.0, 101.0, [6, 5], [5, 5], 10.0))
    assert tracker.state() is None  # fewer than min_obs pairs
    tracker.update(levels(100.0, 101.0, [6, 5], [5, 5], 10_000.0))  # stale gap
    assert tracker.state(min_obs=0) is None


def test_ofi_drift_closed_form_and_cap() -> None:
    state = OfiState(integrated=1.0, beta=1e-4, phi=0.5, dt=10.0, n_obs=100, weights=(1.0,))
    # n -> infinity: beta x / (1 - phi).
    assert ofi_drift(state, 3600.0, sigma=1.0, cap_sigmas=1.0) == pytest.approx(2e-4)
    # Two intervals: beta x (1 + phi).
    assert ofi_drift(state, 20.0, sigma=1.0, cap_sigmas=1.0) == pytest.approx(1.5e-4)
    # Capped at cap * sigma.
    assert ofi_drift(state, 3600.0, sigma=1e-4, cap_sigmas=0.25) == pytest.approx(2.5e-5)
    no_memory = OfiState(1.0, 1e-4, 0.0, 10.0, 100, (1.0,))
    assert ofi_drift(no_memory, 3600.0, 1.0, 1.0) == pytest.approx(1e-4)


def test_vpin_is_exact_on_equal_volume_buckets() -> None:
    volume = np.full(100, 10.0)
    assert np.allclose(vpin_series(volume, volume, 50.0, 5), 1.0)  # all buying
    assert np.allclose(vpin_series(volume, 0.5 * volume, 50.0, 5), 0.0)  # balanced
    # A bucket straddling bars takes each bar's buy share pro rata.
    vol = np.array([10.0, 10.0])
    buy = np.array([10.0, 0.0])
    series = vpin_series(vol, buy, 5.0, 1)
    np.testing.assert_allclose(series, [1.0, 1.0, 1.0, 1.0])
    series = vpin_series(vol, buy, 20.0, 1)
    np.testing.assert_allclose(series, [0.0])
    assert vpin_series(vol, buy, 100.0, 1).size == 0


def candles(buy_share: np.ndarray) -> list[Candle]:
    return [
        Candle(
            ts=60 * i,
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=10.0,
            taker_buy_volume=10.0 * float(s),
        )
        for i, s in enumerate(buy_share)
    ]


def test_toxicity_flags_one_sided_flow() -> None:
    rng = np.random.default_rng(1)
    calm = np.clip(0.5 + rng.normal(0.0, 0.05, 900), 0.0, 1.0)
    toxic = np.concatenate([calm, np.full(250, 0.95)])
    quiet = toxicity_from_candles(candles(calm))
    loud = toxicity_from_candles(candles(toxic))
    assert quiet is not None
    assert loud is not None
    assert loud.vpin > 0.8
    assert loud.excess == pytest.approx(1.0)
    assert loud.vpin > quiet.vpin
    no_taker = [Candle(60 * i, 1, 1, 1, 1, 10.0) for i in range(500)]
    assert toxicity_from_candles(no_taker) is None


def test_toxicity_fattens_tails_within_bounds() -> None:
    assert toxic_tail_index(30.0, 0.0, 1.0) == pytest.approx(30.0)
    assert NU_MIN < toxic_tail_index(30.0, 1.0, 1.0) < 30.0
    assert toxic_tail_index(30.0, 1.0, 50.0) == pytest.approx(NU_MIN, abs=1e-9)
    assert toxic_tail_index(3.0, 0.5, 1.0) == NU_MIN
    assert toxic_variance_multiplier(1.0, 9.0) == pytest.approx(1.5)
    assert toxic_variance_multiplier(-1.0, 0.2) == pytest.approx(1.0)

    prices = 65_000.0 * np.exp(np.array([-0.03, 0.03]))
    base = microstructure_cdf(prices, 65_000.0, 1e-4, 3600.0, 1.0, 0.5, None, None)
    toxic = microstructure_cdf(
        prices, 65_000.0, 1e-4, 3600.0, 1.0, 0.5, None, None, toxicity_excess=1.0, tail_kappa=3.0
    )
    assert toxic[0] > base[0]  # more mass beyond -3 sigma
    assert 1.0 - toxic[1] > 1.0 - base[1]


def test_flow_drift_overrides_static_imbalance() -> None:
    prices = np.array([65_000.0])
    up = microstructure_cdf(prices, 65_000.0, 1e-4, 3600.0, 1.0, 0.5, -1.0, None, flow_drift=0.002)
    down = microstructure_cdf(
        prices, 65_000.0, 1e-4, 3600.0, 1.0, 0.5, 1.0, None, flow_drift=-0.002
    )
    assert up[0] < 0.5 < down[0]
