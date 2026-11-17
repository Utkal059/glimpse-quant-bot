"""Walk-forward replay on a synthetic world with a known data-generating process."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from glimpse_quant_bot.backtester import (
    Backtester,
    BacktestResult,
    load_dataset,
    performance_metrics,
    save_dataset,
    synthetic_dataset,
)
from glimpse_quant_bot.config import Settings
from glimpse_quant_bot.forecaster import (
    FastInputs,
    SkewStudentT,
    build_forecast,
    fit_gjr_garch,
    fit_har,
    fit_slow,
    garman_klass_variance,
    har_variance_path,
    jensen_shannon,
)
from glimpse_quant_bot.types import RecordKind

N_TOPICS = 8


@pytest.fixture(scope="module")
def backtest_settings() -> Settings:
    # 20 sats is the liquidity-calibrated floor; fewer simulation paths keep the test fast.
    return Settings(_env_file=None, min_expected_gain_sats=20.0, garch_paths=800)


async def run(settings: Settings, mispriced: bool, seed: int = 11) -> BacktestResult:
    data = synthetic_dataset(seed=seed, n_topics=N_TOPICS, mispriced=mispriced)
    return await Backtester(settings, data).run()


async def test_replay_profits_from_mispriced_buckets(
    backtest_settings: Settings, tmp_path: Path
) -> None:
    """Stale, over-wide LS-LMSR quotes: positive P&L and ROI after all fees.

    The recording is written to disk and read back through the SDK's
    ``MarketRecorder.load`` so the test covers the real NDJSON path.
    """
    data = synthetic_dataset(seed=11, n_topics=N_TOPICS, mispriced=True)
    paths = save_dataset(data, tmp_path)
    loaded = load_dataset(paths["prices"], paths["meta"], paths["candles"], paths["spot"])
    assert set(loaded.topics) == set(data.topics)
    assert sum(len(e) for e in loaded.price_events.values()) == sum(
        len(e) for e in data.price_events.values()
    )

    result = await Backtester(backtest_settings, loaded).run()

    assert result.entries >= 20
    assert int(result.pnl_millisats) > 0
    assert result.roi > 0.0
    assert int(result.ending_equity_millisats) > int(result.starting_millisats)
    assert result.win_rate > 0.5
    assert result.sharpe > 0.0
    assert result.sortino > 0.0
    assert 0.0 <= result.max_drawdown < backtest_settings.daily_drawdown_limit
    assert int(result.commission_millisats) > 0, "fees must be charged"
    # Convergence exits fire: most positions are closed before binary settlement.
    assert result.exits > result.settlements
    # P&L is fully explained by the ledger rows (nothing is marked to model at the end).
    realised = sum(int(record.realized_pnl_millisats) for record in result.records)
    assert realised == int(result.pnl_millisats)


async def test_profit_is_robust_across_random_worlds(backtest_settings: Settings) -> None:
    outcomes = [await run(backtest_settings, mispriced=True, seed=seed) for seed in (3, 29)]
    assert all(int(result.pnl_millisats) > 0 for result in outcomes)
    assert all(result.roi > 0.0 for result in outcomes)


async def test_no_edge_means_almost_no_trading(backtest_settings: Settings) -> None:
    """Control: when the market is correctly priced the bot must mostly stand aside."""
    mispriced = await run(backtest_settings, mispriced=True)
    control = await run(backtest_settings, mispriced=False)
    assert control.entries <= max(3, mispriced.entries // 20)
    assert abs(control.roi) < 0.01


async def test_gain_floor_is_respected(make_settings: Callable[..., Settings]) -> None:
    """With a 250-sat floor every fill must have had at least 250 sats of expected gain."""
    settings = make_settings(garch_paths=800, min_expected_gain_sats=250.0)
    data = synthetic_dataset(seed=11, n_topics=4, mispriced=True, width_inflation=3.0)
    result = await Backtester(settings, data).run()
    entries = [record for record in result.records if record.kind is RecordKind.ENTRY]
    for record in entries:
        assert record.model_probability is not None
        assert record.exec_price is not None
        edge_points = 98.0 * record.model_probability - float(record.exec_price)
        assert edge_points * record.contracts >= 250.0 - 1e-6


async def test_replay_is_deterministic(backtest_settings: Settings) -> None:
    first = await run(backtest_settings, mispriced=True)
    second = await run(backtest_settings, mispriced=True)
    assert first.pnl_millisats == second.pnl_millisats
    assert first.equity_curve == second.equity_curve


def test_performance_metrics() -> None:
    flat = [(i * 300, 100_000_000) for i in range(10)]
    assert performance_metrics(flat, 300) == (0.0, 0.0, 0.0)

    curve = [(0, 100), (300, 110), (600, 99), (900, 121)]
    sharpe, sortino, max_drawdown = performance_metrics(curve, 300)
    returns = np.array([0.10, -0.10, 121 / 99 - 1])
    annualise = math.sqrt(365 * 86400 / 300)
    assert sharpe == pytest.approx(returns.mean() / returns.std() * annualise)
    assert sortino == pytest.approx(returns.mean() / 0.10 * annualise)
    assert max_drawdown == pytest.approx(0.10)


# --- forecaster components exercised by the replay ---------------------------


def test_har_forecast_tracks_the_volatility_level() -> None:
    data = synthetic_dataset(seed=5, n_topics=2, hourly_vol=0.005)
    candles = next(iter(data.candles.values()))
    variance = garman_klass_variance(candles)
    fit = fit_har(variance, day=24, week=168)
    assert not fit.fallback
    path = har_variance_path(fit, 24)
    hourly_sigma = math.sqrt(float(path.mean()))
    assert hourly_sigma == pytest.approx(0.005, rel=0.25)
    # Too little history -> exponentially weighted fallback, still a sane level.
    short = fit_har(variance[:100], day=24, week=168)
    assert short.fallback
    assert math.sqrt(float(har_variance_path(short, 5).mean())) == pytest.approx(0.005, rel=0.35)


def test_gjr_garch_fit_is_stationary_and_standardises_residuals() -> None:
    rng = np.random.default_rng(1)
    returns = rng.standard_t(6, 900) * 0.004
    fit = fit_gjr_garch(returns)
    assert 0.0 <= fit.persistence < 1.0
    assert fit.omega > 0.0
    assert fit.next_variance > 0.0
    assert float(fit.residuals.mean()) == pytest.approx(0.0, abs=1e-9)
    assert float(fit.residuals.std()) == pytest.approx(1.0, abs=1e-9)


def test_skew_student_t_is_a_standardised_distribution() -> None:
    grid = np.linspace(-40.0, 40.0, 400_001)
    for gamma in (0.8, 1.0, 1.25):
        cdf = SkewStudentT(nu=6.0, gamma=gamma).cdf(grid)
        assert float(np.min(np.diff(cdf))) >= 0.0
        assert cdf[0] == pytest.approx(0.0, abs=1e-4)
        assert cdf[-1] == pytest.approx(1.0, abs=1e-4)
        density = np.gradient(cdf, grid)
        mean = float(np.sum(grid * density) * (grid[1] - grid[0]))
        variance = float(np.sum((grid - mean) ** 2 * density) * (grid[1] - grid[0]))
        assert mean == pytest.approx(0.0, abs=5e-3)
        assert variance == pytest.approx(1.0, abs=3e-2)
    # gamma < 1 is left-skewed: more mass far below the mean than far above it.
    left = SkewStudentT(nu=6.0, gamma=0.8)
    assert float(left.cdf(np.array([-2.5]))[0]) > 1.0 - float(left.cdf(np.array([2.5]))[0])


def test_ensemble_probabilities_sum_to_one_and_reweight(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings(garch_paths=800)
    data = synthetic_dataset(seed=5, n_topics=1)
    topic = next(iter(data.topics.values())).meta
    candles = data.candles[topic.asset]
    now = float(topic.end_time_utc - 2 * 3600)
    closed = [bar for bar in candles if bar.ts + 3600 <= now]
    slow = fit_slow(topic.asset, closed, settings, now)
    spot = closed[-1].close
    forecast = build_forecast(
        topic, slow, FastInputs(spot, 0.4, 0.0005, None, ("test",), now), now, settings
    )
    # BTC has listed options, but no surface was supplied: the weights renormalise 60/40.
    assert forecast.weights == pytest.approx({"har_microstructure": 0.6, "gjr_garch": 0.4})
    assert sum(forecast.probabilities.values()) == pytest.approx(1.0, abs=1e-12)
    assert min(forecast.probabilities.values()) >= 0.0
    assert 0.0 <= forecast.disagreement <= 1.0
    assert settings.kelly_disagreement_floor <= forecast.kelly_multiplier <= 1.0
    # The distribution is centred on spot with roughly the right width.
    mids = {b.option_id: 0.5 * (b.lo + b.hi) for b in topic.buckets}
    mean = sum(mids[oid] * p for oid, p in forecast.probabilities.items())
    assert mean == pytest.approx(spot, rel=0.002)
    assert forecast.sigma == pytest.approx(0.005 * math.sqrt(2.0), rel=0.3)


def test_jensen_shannon_bounds() -> None:
    p = np.array([0.5, 0.5, 0.0, 0.0])
    q = np.array([0.0, 0.0, 0.5, 0.5])
    assert jensen_shannon([p, p], [0.5, 0.5]) == pytest.approx(0.0, abs=1e-12)
    assert jensen_shannon([p, q], [0.5, 0.5]) == pytest.approx(1.0)
    assert jensen_shannon([p], [1.0]) == 0.0
