"""Online Brier weights (fixed share over sleeping experts) and temperature recalibration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from scipy.stats import norm

from glimpse_quant_bot.calibration import (
    OnlineCalibrator,
    PendingForecast,
    apply_temperature,
    brier_score,
    fit_temperature,
)
from glimpse_quant_bot.config import Settings

N = 60
EDGES = np.linspace(-6.0, 6.0, N + 1)
IDS = tuple(range(N))


def gaussian(mu: float, sd: float) -> npt.NDArray[np.float64]:
    p: npt.NDArray[np.float64] = np.diff(np.asarray(norm.cdf(EDGES, mu, sd), dtype=np.float64))
    normalised: npt.NDArray[np.float64] = p / float(p.sum())
    return normalised


def sparse(p: np.ndarray) -> dict[int, float]:
    return {i: float(v) for i, v in enumerate(p) if v > 1e-12}


def pending(components: dict[str, np.ndarray], pooled: np.ndarray) -> PendingForecast:
    return PendingForecast(
        key="BTC:hourly",
        components={k: sparse(v) for k, v in components.items()},
        pooled=sparse(pooled),
        option_ids=IDS,
        end_time_utc=0,
        batch_id="hourly",
    )


def bucket_of(x: float) -> int:
    return int(np.clip(np.searchsorted(EDGES, x) - 1, 0, N - 1))


def test_brier_score_matches_the_dense_formula() -> None:
    p = gaussian(0.3, 1.0)
    y = np.zeros(N)
    y[25] = 1.0
    assert brier_score(sparse(p), 25) == pytest.approx(float(((p - y) ** 2).sum()))


def test_weights_move_to_the_better_model_but_keep_a_floor() -> None:
    rng = np.random.default_rng(0)
    # Brier gap ~0.09/round, eta=0.25: fixed share 0.005 settles near 0.87 / 0.13.
    cal = OnlineCalibrator(prior={"good": 0.5, "bad": 0.5}, share=0.005, min_resolved=5)
    for topic_id in range(150):
        good, bad = gaussian(0.0, 1.0), gaussian(2.5, 1.0)
        cal.record(topic_id, pending({"good": good, "bad": bad}, 0.5 * (good + bad)))
        cal.resolve(topic_id, bucket_of(rng.normal(0.0, 1.0)))
    view = cal.view("BTC:hourly")
    assert view.weights is not None
    assert view.weights["good"] > 0.8
    assert view.weights["bad"] >= 0.005 * 0.5 - 1e-12  # fixed share floor
    assert sum(view.weights.values()) == pytest.approx(1.0)


def test_sleeping_component_keeps_its_mass() -> None:
    cal = OnlineCalibrator(prior={"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, share=0.0, min_resolved=1)
    p = gaussian(0.0, 1.0)
    cal.record(1, pending({"a": p, "b": p, "c": p}, p))
    cal.resolve(1, 30)
    cal.record(2, pending({"a": p, "b": gaussian(3.0, 1.0)}, p))  # c is asleep
    cal.resolve(2, 30)
    view = cal.view("BTC:hourly")
    assert view.weights is not None
    assert view.weights["c"] == pytest.approx(1 / 3)
    assert view.weights["a"] > view.weights["b"]


@pytest.mark.parametrize(("forecast_sd", "direction"), [(1.5, 1), (0.7, -1)])
def test_temperature_corrects_over_and_under_dispersion(forecast_sd: float, direction: int) -> None:
    rng = np.random.default_rng(2)
    p = gaussian(0.0, forecast_sd)
    records = [(p, bucket_of(rng.normal(0.0, 1.0))) for _ in range(400)]
    tau = fit_temperature(records, discount=1.0, ridge=0.0)
    # Gaussian: p^tau rescales the sd by 1/sqrt(tau), so tau* ~ forecast_sd^2.
    assert (tau - 1.0) * direction > 0.0
    assert tau == pytest.approx(forecast_sd**2, rel=0.15)
    sharpened = apply_temperature(p, tau)
    assert sharpened.sum() == pytest.approx(1.0)
    assert apply_temperature(p, 1.0) is p


def test_view_waits_for_evidence_and_shrinks_tau() -> None:
    cal = OnlineCalibrator(prior={"m": 1.0}, min_resolved=3, shrink_n0=30.0)
    assert cal.view("BTC:hourly").weights is None
    p = gaussian(0.0, 2.0)
    for topic_id in range(3):
        cal.record(topic_id, pending({"m": p}, p))
        cal.resolve(topic_id, 30)
    view = cal.view("BTC:hourly")
    assert view.weights is not None
    assert 1.0 < view.temperature < 2.0  # 3 / 33 of the way to the fitted value


def test_unknown_winner_and_expiry_are_ignored() -> None:
    cal = OnlineCalibrator(prior={"m": 1.0})
    p = gaussian(0.0, 1.0)
    cal.record(1, pending({"m": p}, p))
    assert cal.resolve(1, 10_000) is None
    cal.record(2, pending({"m": p}, p))
    cal.expire(now=10.0**9)
    assert cal.pending == {}


def test_state_round_trips_through_json(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    cal = OnlineCalibrator(prior={"a": 0.5, "b": 0.5}, min_resolved=1, path=path)
    for topic_id in range(5):
        cal.record(topic_id, pending({"a": gaussian(0, 1), "b": gaussian(1, 1)}, gaussian(0, 1.5)))
        cal.resolve(topic_id, 30)
    cal.save()
    again = OnlineCalibrator(prior={"a": 0.5, "b": 0.5}, min_resolved=1, path=path)
    assert again.view("BTC:hourly") == cal.view("BTC:hourly")


def test_build_forecast_uses_learned_weights_and_temperature(
    make_settings: Callable[..., Settings],
) -> None:
    from glimpse_quant_bot.backtester import synthetic_dataset
    from glimpse_quant_bot.calibration import CalibrationView
    from glimpse_quant_bot.forecaster import FastInputs, build_forecast, fit_slow

    settings = make_settings(garch_paths=400)
    data = synthetic_dataset(seed=5, n_topics=1)
    topic = next(iter(data.topics.values())).meta
    now = float(topic.end_time_utc - 2 * 3600)
    closed = [bar for bar in data.candles[topic.asset] if bar.ts + 3600 <= now]
    slow = fit_slow(topic.asset, closed, settings, now)
    fast = FastInputs(closed[-1].close, None, None, None, ("test",), now)
    view = CalibrationView({"har_microstructure": 0.9, "gjr_garch": 0.1}, 1.3, 50)
    forecast = build_forecast(topic, slow, fast, now, settings, view)
    assert forecast.weights == pytest.approx({"har_microstructure": 0.9, "gjr_garch": 0.1})
    assert forecast.diagnostics["temperature"] == pytest.approx(1.3)
    assert sum(forecast.probabilities.values()) == pytest.approx(1.0)
    assert forecast.ambiguity == pytest.approx(settings.ambiguity_floor)
