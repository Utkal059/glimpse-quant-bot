"""Online, exponentially weighted Brier recalibration of the ensemble.

Each resolved topic yields one multi-class Brier loss per component model:

    BS_m = sum_i (p_{m,i} - y_i)^2 = 1 - 2 p_{m,win} + sum_i p_{m,i}^2     in [0, 2]

Weights (fixed-share exponential weights over sleeping experts)
---------------------------------------------------------------
``BS`` is ``eta``-exp-concave on the simplex for ``eta <= 1 / (2 max ||p - y||^2)
= 1/4``, so the exponentially weighted *average* of the component vectors has
regret at most ``ln(M) / eta = 4 ln M`` against the best single component
(Kivinen-Warmuth / Cesa-Bianchi-Lugosi). Regimes change, so after each update
a fraction ``alpha`` of the mass is shared back toward the configured prior
weights (Herbster-Warmuth 1998 fixed share), which bounds regret against the
best *sequence* of components with ``S`` switches by
``(ln M + S ln(T / S)) / eta`` up to constants::

    v_m <- w_m exp(-eta BS_m)                     (awake components only)
    v   <- v * (sum_awake w) / (sum_awake v)       (sleeping experts keep their mass)
    w   <- (1 - alpha) v / sum v + alpha prior

Temperature (sharpness recalibration)
-------------------------------------
A linear pool of calibrated forecasts is under-confident (Ranjan and Gneiting
2010). The pooled vector is recalibrated as ``p_i^tau / sum_j p_j^tau`` with
``tau`` minimising the exponentially discounted log score over a ring buffer::

    L(tau) = sum_j d^{age_j} [ -tau ln p_{j,y_j} + ln sum_i p_{j,i}^tau ] + kappa (tau - 1)^2

``L`` is convex (``L'' = sum d^age Var_{p^tau}(ln p) + 2 kappa``), so Newton
converges in a few steps; ``tau`` is clipped to ``[0.5, 2]`` and shrunk toward 1
as ``n / (n + n0)`` while evidence accumulates. For a Gaussian bucket vector,
``tau`` is exactly a ``1 / sqrt(tau)`` rescaling of the forecast's spread.

State is keyed by ``"<ASSET>:<batch_id>"`` (hourly and daily markets of one
asset learn separately) and persisted as JSON.
"""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from glimpse_quant_bot.config import Settings

FloatArray = npt.NDArray[np.float64]

TAU_BOUNDS = (0.5, 2.0)
PROBABILITY_FLOOR = 1.0e-6


def calibration_prior(settings: Settings) -> dict[str, float]:
    """Configured pool weights: the fixed-share target of the calibrator."""
    return {
        "options_rnd": settings.weight_options,
        "har_microstructure": settings.weight_microstructure,
        "gjr_garch": settings.weight_timeseries,
        "timesfm": settings.weight_timeseries,
    }


def brier_score(probabilities: Mapping[int, float], winner: int) -> float:
    """Multi-class Brier score of a sparse probability vector."""
    squares = sum(value * value for value in probabilities.values())
    return float(1.0 - 2.0 * probabilities.get(winner, 0.0) + squares)


def apply_temperature(probabilities: FloatArray, tau: float) -> FloatArray:
    """``p^tau / sum p^tau`` (zeros stay zero)."""
    if abs(tau - 1.0) < 1.0e-12:
        return probabilities
    positive = probabilities > 0.0
    log_p = np.where(positive, np.log(np.where(positive, probabilities, 1.0)), -np.inf)
    z = tau * log_p
    z = z - float(np.max(z[positive])) if bool(np.any(positive)) else z
    powered = np.where(positive, np.exp(z), 0.0)
    total = float(powered.sum())
    result: FloatArray = powered / total if total > 0.0 else probabilities
    return result


def fit_temperature(records: list[tuple[FloatArray, int]], discount: float, ridge: float) -> float:
    """Newton on the convex discounted log score (``records`` oldest first)."""
    if not records:
        return 1.0
    n = len(records)
    decay = discount ** np.arange(n - 1, -1, -1, dtype=np.float64)
    logs = [np.log(np.maximum(p, PROBABILITY_FLOOR)) for p, _ in records]
    tau = 1.0
    for _ in range(30):
        grad = 2.0 * ridge * (tau - 1.0)
        hess = 2.0 * ridge
        for weight, log_p, (_, y) in zip(decay, logs, records, strict=True):
            z = tau * log_p
            z = z - float(np.max(z))
            q = np.exp(z)
            q = q / float(q.sum())
            mean = float(q @ log_p)
            grad += weight * (-float(log_p[y]) + mean)
            hess += weight * max(float(q @ (log_p * log_p)) - mean * mean, 0.0)
        if hess <= 0.0:
            break
        step = grad / hess
        tau = float(np.clip(tau - step, *TAU_BOUNDS))
        if abs(step) < 1.0e-8:
            break
    return tau


@dataclass
class _KeyState:
    weights: dict[str, float] = field(default_factory=dict)
    n: int = 0
    buffer: deque[tuple[FloatArray, int]] = field(default_factory=deque)
    tau: float = 1.0
    brier: dict[str, float] = field(default_factory=dict)
    """EW average Brier per component (diagnostic)."""


@dataclass(frozen=True, slots=True)
class CalibrationView:
    """What ``build_forecast`` consumes."""

    weights: Mapping[str, float] | None
    temperature: float
    n_resolved: int


@dataclass(frozen=True, slots=True)
class PendingForecast:
    key: str
    components: Mapping[str, Mapping[int, float]]
    pooled: Mapping[int, float]
    option_ids: tuple[int, ...]
    end_time_utc: int
    batch_id: str


class OnlineCalibrator:
    def __init__(
        self,
        prior: Mapping[str, float] | None = None,
        *,
        eta: float = 0.25,
        share: float = 0.02,
        discount: float = 0.98,
        buffer_size: int = 300,
        min_resolved: int = 20,
        shrink_n0: float = 30.0,
        ridge: float = 1.0,
        path: Path | None = None,
    ) -> None:
        self._prior = dict(prior or {})
        self._eta = min(max(eta, 0.0), 0.25)
        self._share = float(np.clip(share, 0.0, 1.0))
        self._discount = discount
        self._buffer_size = buffer_size
        self._min_resolved = min_resolved
        self._n0 = shrink_n0
        self._ridge = ridge
        self._path = path
        self._states: dict[str, _KeyState] = {}
        self.pending: dict[int, PendingForecast] = {}
        if path is not None and path.exists():
            self._load(path)

    @staticmethod
    def key_for(asset: str, batch_id: str) -> str:
        return f"{asset}:{batch_id}"

    def _state(self, key: str) -> _KeyState:
        state = self._states.get(key)
        if state is None:
            state = _KeyState(buffer=deque(maxlen=self._buffer_size))
            self._states[key] = state
        return state

    def _prior_for(self, names: list[str]) -> dict[str, float]:
        raw = {name: max(self._prior.get(name, 0.0), 0.0) for name in names}
        total = sum(raw.values())
        if total <= 0.0:
            return {name: 1.0 / len(names) for name in names}
        return {name: value / total for name, value in raw.items()}

    # --- consumption ------------------------------------------------------

    def view(self, key: str) -> CalibrationView:
        state = self._states.get(key)
        if state is None or state.n < self._min_resolved:
            return CalibrationView(None, 1.0, 0 if state is None else state.n)
        shrink = state.n / (state.n + self._n0)
        tau = 1.0 + (state.tau - 1.0) * shrink
        return CalibrationView(dict(state.weights), tau, state.n)

    # --- learning ----------------------------------------------------------

    def record(self, topic_id: int, pending: PendingForecast) -> None:
        """Remember (overwrite) the forecast that will be scored when ``topic_id`` resolves."""
        self.pending[topic_id] = pending

    def resolve(self, topic_id: int, winner: int) -> dict[str, float] | None:
        """Score a resolved topic; returns the component Brier scores, or ``None``."""
        pending = self.pending.pop(topic_id, None)
        if pending is None or winner not in pending.option_ids:
            return None
        state = self._state(pending.key)
        names = sorted(pending.components)
        if not names:
            return None
        scores = {name: brier_score(pending.components[name], winner) for name in names}
        prior = self._prior_for(sorted(set(names) | set(state.weights)))
        for name in prior:
            state.weights.setdefault(name, prior[name])
        awake_mass = sum(state.weights[name] for name in names)
        updated = {
            name: state.weights[name] * math.exp(-self._eta * scores[name]) for name in names
        }
        total = sum(updated.values())
        if total > 0.0 and awake_mass > 0.0:
            for name in names:
                state.weights[name] = updated[name] * awake_mass / total
        grand = sum(state.weights.values())
        state.weights = {
            name: (1.0 - self._share) * value / grand + self._share * prior[name]
            for name, value in state.weights.items()
        }
        for name, score in scores.items():
            previous = state.brier.get(name)
            state.brier[name] = score if previous is None else 0.95 * previous + 0.05 * score
        dense = np.array([pending.pooled.get(option_id, 0.0) for option_id in pending.option_ids])
        total_mass = float(dense.sum())
        if total_mass > 0.0:
            dense = dense / total_mass
            state.buffer.append((dense, pending.option_ids.index(winner)))
            state.tau = fit_temperature(list(state.buffer), self._discount, self._ridge)
        state.n += 1
        return scores

    def expire(self, now: float, grace_seconds: float = 86400.0) -> None:
        """Drop pending forecasts whose resolution never arrived."""
        for topic_id in [
            tid for tid, p in self.pending.items() if now - p.end_time_utc > grace_seconds
        ]:
            del self.pending[topic_id]

    # --- persistence ---------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            key: {
                "weights": state.weights,
                "n": state.n,
                "tau": state.tau,
                "brier": state.brier,
                "buffer": [
                    {
                        "y": y,
                        "n": int(p.size),
                        "sparse": {str(i): float(p[i]) for i in np.flatnonzero(p > 1.0e-7)},
                    }
                    for p, y in state.buffer
                ],
            }
            for key, state in self._states.items()
        }

    def save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.snapshot()), encoding="utf-8")
        tmp.replace(self._path)

    def _load(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for key, raw in data.items():
            state = self._state(key)
            state.weights = {str(k): float(v) for k, v in raw.get("weights", {}).items()}
            state.n = int(raw.get("n", 0))
            state.tau = float(raw.get("tau", 1.0))
            state.brier = {str(k): float(v) for k, v in raw.get("brier", {}).items()}
            for item in raw.get("buffer", []):
                dense = np.zeros(int(item["n"]))
                for index, value in item["sparse"].items():
                    dense[int(index)] = float(value)
                state.buffer.append((dense, int(item["y"])))
