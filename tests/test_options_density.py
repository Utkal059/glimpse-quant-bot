"""Breeden-Litzenberger density: normalisation, accuracy, and response to skew and IV."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from scipy.special import ndtr

from glimpse_quant_bot.external_feeds import (
    OptionQuote,
    parse_deribit_chain,
    parse_deribit_instrument,
)
from glimpse_quant_bot.options_density import (
    YEAR_SECONDS,
    VolSurface,
    black76_call,
    bucket_probabilities_rnd,
    fit_smile_slice,
    risk_neutral_cdf,
    risk_neutral_density,
)

NOW = 1_800_000_000.0
FORWARD = 65_000.0
ONE_DAY = 86_400


def make_chain(
    smile: Callable[[float], float], expiry_seconds: int = ONE_DAY, forward: float = FORWARD
) -> list[OptionQuote]:
    """A Deribit-like chain: calls and puts at every $500 strike, IV from ``smile(z)``."""
    t_years = expiry_seconds / YEAR_SECONDS
    atm = smile(0.0)
    quotes: list[OptionQuote] = []
    for strike in np.arange(forward * 0.80, forward * 1.20 + 1.0, 500.0):
        z = math.log(float(strike) / forward) / (atm * math.sqrt(t_years))
        quotes.extend(
            OptionQuote(
                instrument=f"BTC-X-{int(strike)}-{'C' if is_call else 'P'}",
                expiry_ts=int(NOW) + expiry_seconds,
                strike=float(strike),
                is_call=is_call,
                mark_iv=smile(z),
                forward=forward,
                open_interest=10.0,
            )
            for is_call in (True, False)
        )
    return quotes


def surface_for(smile: Callable[[float], float], expiry_seconds: int = ONE_DAY) -> VolSurface:
    surface = VolSurface.from_chain(make_chain(smile, expiry_seconds), NOW)
    assert surface is not None
    return surface


def buckets(width: float = 200.0) -> tuple[np.ndarray, np.ndarray]:
    lows = np.arange(25_000.0, 125_000.0, width)
    return lows, lows + width


def flat(level: float) -> Callable[[float], float]:
    return lambda _z: level


def test_bucket_probabilities_sum_to_one() -> None:
    lows, highs = buckets()
    surface = surface_for(lambda z: 0.50 - 0.03 * z + 0.02 * z * z)
    probabilities = bucket_probabilities_rnd(lows, highs, FORWARD, 0.5 / 365.0, surface)
    assert probabilities.shape == lows.shape
    assert float(np.min(probabilities)) >= 0.0
    assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-12)


def test_flat_smile_matches_the_lognormal_closed_form() -> None:
    """With flat volatility Breeden-Litzenberger must reproduce Black-76's N(-d2)."""
    lows, highs = buckets()
    vol, t_years = 0.50, 0.5 / 365.0
    probabilities = bucket_probabilities_rnd(lows, highs, FORWARD, t_years, surface_for(flat(vol)))

    sigma = vol * math.sqrt(t_years)

    def cdf(strikes: np.ndarray) -> np.ndarray:
        return np.asarray(ndtr((np.log(strikes / FORWARD) + 0.5 * sigma**2) / sigma))

    exact = cdf(highs) - cdf(lows)
    exact = exact / exact.sum()
    assert float(np.abs(probabilities - exact).max()) < 2e-5


def test_density_is_non_negative_and_integrates_to_one() -> None:
    surface = surface_for(lambda z: 0.55 - 0.04 * z + 0.02 * z * z)
    t_years = 1.0 / 365.0
    strikes = np.linspace(FORWARD * 0.7, FORWARD * 1.3, 6001)
    density = risk_neutral_density(strikes, FORWARD, t_years, surface)
    assert float(np.min(density)) >= 0.0
    integral = float(np.sum(0.5 * (density[1:] + density[:-1]) * np.diff(strikes)))
    assert integral == pytest.approx(1.0, abs=5e-3)


def test_cdf_is_monotone_between_zero_and_one() -> None:
    surface = surface_for(lambda z: 0.55 - 0.04 * z + 0.02 * z * z)
    strikes = np.linspace(1.0, FORWARD * 3.0, 4001)
    cdf = risk_neutral_cdf(strikes, FORWARD, 1.0 / 365.0, surface)
    assert cdf[0] == 0.0
    assert cdf[-1] == pytest.approx(1.0, abs=1e-6)
    assert float(np.min(np.diff(cdf))) >= -1e-6


def test_negative_skew_moves_mass_to_the_downside() -> None:
    """Put skew (higher IV for low strikes) fattens the left tail and thins the right."""
    lows, highs = buckets()
    t_years = 1.0 / 365.0
    symmetric = bucket_probabilities_rnd(
        lows, highs, FORWARD, t_years, surface_for(lambda z: 0.50 + 0.02 * z * z)
    )
    skewed = bucket_probabilities_rnd(
        lows, highs, FORWARD, t_years, surface_for(lambda z: 0.50 - 0.06 * z + 0.02 * z * z)
    )
    sigma_move = FORWARD * 0.50 * math.sqrt(t_years)
    far_left = highs <= FORWARD - 2.0 * sigma_move
    far_right = lows >= FORWARD + 2.0 * sigma_move
    assert float(skewed[far_left].sum()) > float(symmetric[far_left].sum()) * 1.2
    assert float(skewed[far_right].sum()) < float(symmetric[far_right].sum()) * 0.8


def test_higher_implied_volatility_flattens_the_distribution() -> None:
    lows, highs = buckets()
    t_years = 1.0 / 365.0
    calm = bucket_probabilities_rnd(lows, highs, FORWARD, t_years, surface_for(flat(0.30)))
    wild = bucket_probabilities_rnd(lows, highs, FORWARD, t_years, surface_for(flat(0.90)))
    centre = int(np.argmax(calm))
    assert float(wild[centre]) < float(calm[centre]) * 0.5
    tails = (highs <= FORWARD * 0.97) | (lows >= FORWARD * 1.03)
    assert float(wild[tails].sum()) > float(calm[tails].sum()) * 5.0
    assert float(calm.sum()) == pytest.approx(1.0)
    assert float(wild.sum()) == pytest.approx(1.0)


def test_smile_fit_recovers_the_generating_quadratic() -> None:
    fitted = fit_smile_slice(make_chain(lambda z: 0.50 - 0.03 * z + 0.02 * z * z), NOW)
    assert fitted is not None
    a, b, c = fitted.coeffs
    assert a == pytest.approx(0.50, abs=1e-6)
    assert b == pytest.approx(-0.03, abs=1e-6)
    assert c == pytest.approx(0.02, abs=1e-6)
    assert fitted.forward == pytest.approx(FORWARD)


def test_total_variance_is_interpolated_between_expiries() -> None:
    chain = make_chain(flat(0.40), ONE_DAY) + make_chain(flat(0.60), 3 * ONE_DAY)
    surface = VolSurface.from_chain(chain, NOW)
    assert surface is not None
    assert len(surface.slices) == 2
    t_mid = 2.0 / 365.0
    expected = math.sqrt(0.5 * (0.40**2 * 1.0 + 0.60**2 * 3.0) / 2.0)
    assert surface.scale_vol(t_mid) == pytest.approx(expected, rel=1e-9)
    # Below the first expiry the first slice's volatility is carried flat.
    assert surface.scale_vol(0.1 / 365.0) == pytest.approx(0.40)
    at_the_money = surface.vol(np.array([FORWARD]), FORWARD, t_mid)
    assert float(at_the_money[0]) == pytest.approx(expected, rel=1e-9)


def test_expiries_too_close_to_now_are_ignored() -> None:
    chain = make_chain(flat(0.9), 600) + make_chain(flat(0.5), ONE_DAY)
    surface = VolSurface.from_chain(chain, NOW)
    assert surface is not None
    assert [round(s.scale_vol, 2) for s in surface.slices] == [0.5]
    assert VolSurface.from_chain(make_chain(flat(0.9), 600), NOW) is None


def test_black76_put_call_parity() -> None:
    strikes = np.array([60_000.0, 65_000.0, 70_000.0])
    vol = np.full(3, 0.6)
    calls = black76_call(FORWARD, strikes, vol, 7.0 / 365.0)
    # Call minus put equals forward minus strike; a put is a call on the mirrored payoff.
    sigma = 0.6 * math.sqrt(7.0 / 365.0)
    d1 = (np.log(FORWARD / strikes) + 0.5 * sigma**2) / sigma
    puts = strikes * ndtr(-(d1 - sigma)) - FORWARD * ndtr(-d1)
    assert np.allclose(calls - puts, FORWARD - strikes, atol=1e-6)


def test_deribit_instrument_parsing() -> None:
    expiry, strike, is_call = parse_deribit_instrument("BTC-20NOV26-64000-P")
    assert (strike, is_call) == (64_000.0, False)
    # 20 Nov 2026 08:00:00 UTC
    assert expiry == 1_795_161_600
    chain = parse_deribit_chain(
        [
            {
                "instrument_name": "BTC-20NOV26-64000-P",
                "mark_iv": 49.51,
                "underlying_price": 64900.13,
            },
            {
                "instrument_name": "BTC-20NOV26-66000-C",
                "mark_iv": None,
                "underlying_price": 64900.13,
            },
            {"instrument_name": "garbage", "mark_iv": 50.0, "underlying_price": 64900.13},
        ]
    )
    assert len(chain) == 1
    assert chain[0].mark_iv == pytest.approx(0.4951)
    assert chain[0].forward == pytest.approx(64900.13)
