"""Multi-level order-flow imbalance (L1-L5) and VPIN flow toxicity.

Integrated OFI (Cont, Kukanov and Stoikov 2014; Cont, Cucuringu and Zhang 2023)
--------------------------------------------------------------------------------
Between two book snapshots, level ``m`` contributes (bid ``b``, ask ``a``,
price ``P``, size ``q``)::

    e_m^b = q_m^b(t) 1[P_m^b(t) >= P_m^b(t-1)] - q_m^b(t-1) 1[P_m^b(t) <= P_m^b(t-1)]
    e_m^a = q_m^a(t-1) 1[P_m^a(t) >= P_m^a(t-1)] - q_m^a(t) 1[P_m^a(t) <= P_m^a(t-1)]
    OFI_m = e_m^b + e_m^a,          ofi_m = OFI_m / Qbar,  Qbar = EW mean depth per level

and the integrated signal is ``x_t = w' ofi_t`` with ``w`` the first principal
component of the EW covariance of ``ofi`` (normalised so ``sum |w| = 1`` and
``sum w > 0``). Cont-Cucuringu-Zhang find this PC explains > 80% of the
multi-level variance and beats best-level OFI out of sample.

Drift (closed form)
-------------------
OFI's *contemporaneous* impact is large but its *predictive* power decays in
minutes. Rather than hard-coding a coefficient, the tracker fits online (EW
ridge least squares, forgetting factor ``lambda_f``)::

    r_{t+1} = beta x_t + e_t,        x_{t+1} = phi x_t + u_t

so the expected log return over a horizon of ``n = h / dt`` snapshot intervals is::

    mu_h = beta x_t (1 - phi^n) / (1 - phi)        (-> beta x_t / (1 - phi) as n -> inf)

clipped to ``+/- cap * sigma_h``. If OFI does not predict returns on this venue,
``beta -> 0`` and so does the drift: the signal earns its weight from data.

VPIN (Easley, Lopez de Prado and O'Hara 2012), exact buy volume
---------------------------------------------------------------
Binance klines report taker-buy volume per bar, so the bulk-volume
classification that Andersen and Bondarenko (2014) criticise is unnecessary.
Equal-volume buckets of size ``V`` are cut on the cumulative-volume curve and
the buy volume inside each bucket is read off by linear interpolation of the
cumulative-buy curve (exact under uniform intra-bar flow)::

    B(v) = interp(v, cumV, cumBuy),   VPIN = (1/n) sum_{last n buckets} |2 dB_tau - V| / V

The current VPIN is ranked against its own recent history, ``u = F_hat(VPIN)``,
and only the upper tail counts as toxic::

    s = clip((u - u_0) / (1 - u_0), 0, 1)            (u_0 = 0.8)
    nu_tox = nu_min + (nu - nu_min) exp(-kappa_nu s)  (fatter tails, nu_min = 4.2)
    var_tox = var (1 + kappa_var s)                   (kappa_var <= 0.5)

Andersen and Bondarenko find no *incremental* volatility forecasting power for
VPIN once volume and volatility are controlled for, so ``kappa_var`` defaults
small and the main effect is on tail shape.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from glimpse_quant_bot.external_feeds import Candle, OrderBook

FloatArray = npt.NDArray[np.float64]

NU_MIN = 4.2
"""Smallest Student-t tail index used anywhere: keeps the fourth moment finite."""


@dataclass(frozen=True, slots=True)
class BookLevels:
    """Top ``M`` price levels of one snapshot."""

    bid_px: FloatArray
    bid_sz: FloatArray
    ask_px: FloatArray
    ask_sz: FloatArray
    ts: float

    @property
    def mid(self) -> float:
        return 0.5 * (float(self.bid_px[0]) + float(self.ask_px[0]))


def book_levels(book: OrderBook, ts: float, levels: int = 5) -> BookLevels | None:
    """Pad-free top-``levels`` arrays; ``None`` if either side is too shallow."""
    if len(book.bids) < levels or len(book.asks) < levels:
        return None
    bids = np.asarray(book.bids[:levels], dtype=np.float64)
    asks = np.asarray(book.asks[:levels], dtype=np.float64)
    return BookLevels(bids[:, 0], bids[:, 1], asks[:, 0], asks[:, 1], ts)


def level_ofi(prev: BookLevels, curr: BookLevels) -> FloatArray:
    """Cont-Kukanov-Stoikov order-flow imbalance per level (signed size units)."""
    bid = curr.bid_sz * (curr.bid_px >= prev.bid_px) - prev.bid_sz * (curr.bid_px <= prev.bid_px)
    ask = prev.ask_sz * (curr.ask_px >= prev.ask_px) - curr.ask_sz * (curr.ask_px <= prev.ask_px)
    result: FloatArray = bid + ask
    return result


@dataclass(frozen=True, slots=True)
class OfiState:
    integrated: float
    """Latest integrated, depth-normalised OFI ``x_t``."""
    beta: float
    """Predictive coefficient of the next-interval log return on ``x_t``."""
    phi: float
    """AR(1) persistence of ``x``."""
    dt: float
    """EW mean seconds between snapshots."""
    n_obs: int
    weights: tuple[float, ...]


class OfiTracker:
    """Online integrated-OFI state for one asset (one instance per asset)."""

    def __init__(
        self,
        levels: int = 5,
        forgetting: float = 0.995,
        ridge: float = 5.0,
        max_phi: float = 0.95,
        max_gap_seconds: float = 120.0,
    ) -> None:
        self.levels = levels
        self._lam = forgetting
        self._ridge = ridge
        self._max_phi = max_phi
        self._max_gap = max_gap_seconds
        self._prev: BookLevels | None = None
        self._x_prev: float | None = None
        self._depth = 0.0
        self._cov = np.zeros((levels, levels))
        self._cov_n = 0.0
        self._sxx = 0.0
        self._sxr = 0.0
        self._lag_xx = 0.0
        self._lag_xy = 0.0
        self._dt = 0.0
        self.n_obs = 0
        self._weights = np.full(levels, 1.0 / levels)

    def _pc_weights(self) -> FloatArray:
        if self._cov_n < self.levels + 5:
            return np.full(self.levels, 1.0 / self.levels)
        _, vectors = np.linalg.eigh(self._cov / self._cov_n)
        first = vectors[:, -1]
        if float(first.sum()) < 0.0:
            first = -first
        total = float(np.abs(first).sum())
        if total <= 0.0:
            return np.full(self.levels, 1.0 / self.levels)
        weights: FloatArray = first / total
        return weights

    def update(self, snapshot: BookLevels) -> None:
        prev = self._prev
        self._prev = snapshot
        depth = float(np.mean(np.concatenate([snapshot.bid_sz, snapshot.ask_sz])))
        self._depth = depth if self._depth <= 0.0 else 0.95 * self._depth + 0.05 * depth
        if prev is None:
            return
        gap = snapshot.ts - prev.ts
        if gap <= 0.0 or gap > self._max_gap:
            # A stale gap breaks the event sequence: restart the pairing.
            self._x_prev = None
            return
        lam = self._lam
        self._dt = gap if self._dt <= 0.0 else lam * self._dt + (1.0 - lam) * gap
        ofi = level_ofi(prev, snapshot) / max(self._depth, 1.0e-12)
        self._cov = lam * self._cov + np.outer(ofi, ofi)
        self._cov_n = lam * self._cov_n + 1.0
        self._weights = self._pc_weights()
        x = float(self._weights @ ofi)
        ret = math.log(snapshot.mid / prev.mid)
        if self._x_prev is not None:
            # r_t is the return over (t-1, t]; x_prev was known at t-1.
            self._sxx = lam * self._sxx + self._x_prev**2
            self._sxr = lam * self._sxr + self._x_prev * ret
            self._lag_xx = lam * self._lag_xx + self._x_prev**2
            self._lag_xy = lam * self._lag_xy + self._x_prev * x
            self.n_obs += 1
        self._x_prev = x

    def state(self, min_obs: int = 20) -> OfiState | None:
        if self.n_obs < min_obs or self._x_prev is None or self._dt <= 0.0:
            return None
        beta = self._sxr / (self._sxx + self._ridge)
        phi = (
            float(np.clip(self._lag_xy / self._lag_xx, 0.0, self._max_phi))
            if self._lag_xx > 0.0
            else 0.0
        )
        return OfiState(
            integrated=self._x_prev,
            beta=beta,
            phi=phi,
            dt=self._dt,
            n_obs=self.n_obs,
            weights=tuple(float(w) for w in self._weights),
        )


def ofi_drift(state: OfiState, horizon_seconds: float, sigma: float, cap_sigmas: float) -> float:
    """``beta x (1 - phi^n) / (1 - phi)``, clipped to ``+/- cap_sigmas * sigma``."""
    n = max(horizon_seconds / state.dt, 0.0)
    multiplier = min(n, 1.0) if state.phi <= 0.0 else (1.0 - state.phi**n) / (1.0 - state.phi)
    drift = state.beta * state.integrated * multiplier
    limit = cap_sigmas * sigma
    return float(np.clip(drift, -limit, limit))


# ---------------------------------------------------------------------------
# VPIN
# ---------------------------------------------------------------------------


def vpin_series(
    volume: FloatArray, buy_volume: FloatArray, bucket_volume: float, window: int
) -> FloatArray:
    """Rolling VPIN over equal-volume buckets (empty if fewer than ``window`` buckets)."""
    if bucket_volume <= 0.0 or volume.size == 0:
        return np.zeros(0)
    vol = np.maximum(volume, 0.0)
    buy = np.clip(buy_volume, 0.0, vol)
    cum_v = np.concatenate([[0.0], np.cumsum(vol)])
    cum_b = np.concatenate([[0.0], np.cumsum(buy)])
    n_buckets = int(cum_v[-1] // bucket_volume)
    if n_buckets < window:
        return np.zeros(0)
    edges = bucket_volume * np.arange(n_buckets + 1)
    buys = np.diff(np.interp(edges, cum_v, cum_b))
    imbalance = np.abs(2.0 * buys - bucket_volume) / bucket_volume
    series: FloatArray = np.convolve(imbalance, np.full(window, 1.0 / window), mode="valid")
    return series


@dataclass(frozen=True, slots=True)
class ToxicityState:
    vpin: float
    percentile: float
    """Empirical CDF of the current VPIN within its own recent history."""
    excess: float
    """Upper-tail toxicity score ``s`` in ``[0, 1]``."""


def toxicity_from_candles(
    candles: Sequence[Candle],
    bucket_minutes: float = 10.0,
    window: int = 20,
    threshold: float = 0.8,
) -> ToxicityState | None:
    """VPIN from 1-minute bars carrying taker-buy volume; ``None`` if unavailable."""
    rows = [bar for bar in candles if bar.taker_buy_volume is not None and bar.volume > 0.0]
    if len(rows) < 120:
        return None
    volume = np.array([bar.volume for bar in rows])
    buy = np.array([float(bar.taker_buy_volume or 0.0) for bar in rows])
    bar_seconds = float(np.median(np.diff([bar.ts for bar in rows]))) or 60.0
    bucket_volume = float(volume.mean()) * bucket_minutes * 60.0 / bar_seconds
    series = vpin_series(volume, buy, bucket_volume, window)
    if series.size < 10:
        return None
    current = float(series[-1])
    percentile = float(np.mean(series <= current))
    span = max(1.0 - threshold, 1.0e-9)
    excess = float(np.clip((percentile - threshold) / span, 0.0, 1.0))
    return ToxicityState(vpin=current, percentile=percentile, excess=excess)


def toxic_tail_index(nu: float, excess: float, kappa: float) -> float:
    """``nu_min + (nu - nu_min) exp(-kappa s)``."""
    if nu <= NU_MIN:
        return NU_MIN
    return NU_MIN + (nu - NU_MIN) * math.exp(-max(kappa, 0.0) * float(np.clip(excess, 0.0, 1.0)))


def toxic_variance_multiplier(excess: float, kappa: float) -> float:
    """``1 + kappa s`` with ``kappa`` clipped to ``[0, 0.5]``."""
    return 1.0 + float(np.clip(kappa, 0.0, 0.5)) * float(np.clip(excess, 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class FlowSignal:
    """Per-asset flow inputs handed to the forecaster."""

    ofi: OfiState | None
    toxicity: ToxicityState | None
