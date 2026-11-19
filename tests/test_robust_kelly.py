"""KL-robust Kelly: closed forms against brute force, and the corridor basket optimiser."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from glimpse_markets.money import PriceUnits
from scipy.optimize import minimize

from glimpse_quant_bot.lmsr import LmsrBook, cost
from glimpse_quant_bot.lmsr_optimizer import kelly_fraction
from glimpse_quant_bot.robust_kelly import (
    aumann_shapley_costs,
    kl_bernoulli,
    kl_lower_probability,
    optimize_basket,
    robust_kelly_fraction,
    robust_log_growth,
    sharpe_shrinkage,
)
from glimpse_quant_bot.types import TopicMeta

PI = 98.0


@pytest.mark.parametrize(("p", "eps"), [(0.3, 1e-3), (0.05, 2e-4), (0.9, 0.01)])
def test_kl_lower_probability_sits_on_the_ball(p: float, eps: float) -> None:
    q = kl_lower_probability(p, eps)
    assert q < p
    assert kl_bernoulli(q, p) == pytest.approx(eps, rel=1e-9)


def test_kl_lower_probability_edges() -> None:
    assert kl_lower_probability(0.3, 0.0) == 0.3
    assert kl_lower_probability(0.3, 10.0) == 0.0
    assert kl_lower_probability(0.0, 0.1) == 0.0


def test_robust_kelly_equals_brute_force_worst_case() -> None:
    p, c, eps = 0.32, 25.0, 2e-3
    b = (PI - c) / c
    qs = np.linspace(1e-6, 1 - 1e-6, 20001)
    ball = qs[np.array([kl_bernoulli(float(q), p) for q in qs]) <= eps]
    fs = np.linspace(0.0, 0.5, 5001)
    worst = np.array(
        [float(np.min(ball * math.log1p(f * b) + (1 - ball) * math.log1p(-f))) for f in fs]
    )
    brute = float(fs[int(np.argmax(worst))])
    assert robust_kelly_fraction(p, c, PI, eps) == pytest.approx(brute, abs=2e-4)


def test_robust_kelly_shrinks_and_reduces_to_kelly() -> None:
    p, c = 0.30, 24.0
    plain = (p * PI - c) / (PI - c)
    assert robust_kelly_fraction(p, c, PI, 0.0) == pytest.approx(plain)
    assert robust_kelly_fraction(p, c, PI, 1e-3) < plain
    assert robust_kelly_fraction(p, 29.0, PI, 0.05) == 0.0  # edge not robust
    assert kelly_fraction(p, PriceUnits(c), 0.02) == pytest.approx(plain)


def test_sharpe_shrinkage_approximates_the_exact_ratio() -> None:
    p, c, eps = 0.30, 22.0, 2e-4
    exact = robust_kelly_fraction(p, c, PI, eps) / robust_kelly_fraction(p, c, PI, 0.0)
    assert sharpe_shrinkage(p, c, PI, eps) == pytest.approx(exact, rel=0.03)


def test_dual_matches_the_primal_worst_case() -> None:
    probs = np.array([0.2, 0.3, 0.5])
    wealth = np.array([1.4, 0.9, 1.05])
    eps = 0.02
    value, lam = robust_log_growth(probs, wealth, eps)
    log_w = np.log(wealth)

    def objective(x: np.ndarray) -> float:
        return float(x @ log_w)

    constraints = [
        {"type": "eq", "fun": lambda x: float(x.sum()) - 1.0},
        {"type": "ineq", "fun": lambda x: eps - float((x * np.log(x / probs)).sum())},
    ]
    primal = minimize(
        objective, probs, constraints=constraints, bounds=[(1e-9, 1.0)] * 3, method="SLSQP"
    )
    assert value == pytest.approx(float(primal.fun), abs=1e-6)
    assert value < float(probs @ log_w)
    assert 0.0 < lam < math.inf
    assert robust_log_growth(probs, wealth, 0.0)[0] == pytest.approx(float(probs @ log_w))


def test_aumann_shapley_split_is_exact(book: LmsrBook) -> None:
    legs = [(198, 10.0), (199, 20.0), (205, 5.0)]
    shares = aumann_shapley_costs(book, legs)
    after = book.shares.copy()
    for option_id, k in legs:
        after[book.index_of(option_id)] += k
    exact = cost(after, book.alpha) - cost(book.shares, book.alpha)
    assert float(shares.sum()) == pytest.approx(exact, rel=1e-12)
    assert np.all(shares > 0.0)
    # Joint pricing is cheaper than pricing each leg against the untouched book.
    separate = sum(int(book.buy_cost([leg])) for leg in legs) / 1000.0
    assert exact < separate


def model_tilted(topic: TopicMeta, book: LmsrBook) -> dict[int, float]:
    """Model that moves mass from the low side of the traded range to the high side.

    Market weights at indices 197-201 are ~(0.05, 0.21, 0.36, 0.28, 0.10); the model
    says (0.01, 0.07, 0.38, 0.36, 0.18): +6.7 and +7.2 points of edge on 200 and 201.
    """
    m = np.zeros(len(topic.buckets))
    m[197:202] = [0.01, 0.07, 0.38, 0.36, 0.18]
    return {bucket.option_id: float(m[i]) for i, bucket in enumerate(topic.buckets)}


def test_basket_finds_a_robust_corridor(topic: TopicMeta, book: LmsrBook) -> None:
    model = model_tilted(topic, book)
    kwargs: dict[str, Any] = {
        "sub_bankroll_points": 35_000.0,
        "max_cost_points": 20_000.0,
        "fee_rate": 0.02,
        "max_legs": 12,
    }
    nominal = optimize_basket(book, model, ambiguity=0.0, **kwargs)
    robust = optimize_basket(book, model, ambiguity=2e-3, **kwargs)
    assert nominal is not None
    assert robust is not None
    assert len(nominal.legs) >= 2
    legs = {leg.option_id for leg in nominal.legs}
    assert legs <= {200, 201, 202}  # option_id = index + 1
    assert robust.all_in_points < nominal.all_in_points
    assert 0.0 < robust.robust_growth <= robust.nominal_growth
    assert nominal.expected_profit_points > 0.0
    # Reported cost is the exact LS-LMSR cost of the legs.
    assert robust.cost_points == pytest.approx(
        int(book.buy_cost(robust.leg_tuples)) / 1000.0, abs=2e-3
    )
    assert sum(leg.cost_points for leg in robust.legs) == pytest.approx(
        robust.cost_points, rel=1e-6
    )


def test_basket_respects_the_budget_and_declines_without_edge(
    topic: TopicMeta, book: LmsrBook
) -> None:
    model = model_tilted(topic, book)
    tight = optimize_basket(
        book,
        model,
        sub_bankroll_points=35_000.0,
        max_cost_points=300.0,
        fee_rate=0.02,
        ambiguity=0.0,
    )
    assert tight is not None
    assert tight.cost_points * 1.02 <= 300.0 + 1e-6
    fair = {b.option_id: float(p) for b, p in zip(topic.buckets, book.probabilities(), strict=True)}
    assert (
        optimize_basket(
            book,
            fair,
            sub_bankroll_points=35_000.0,
            max_cost_points=1e4,
            fee_rate=0.02,
            ambiguity=0.0,
        )
        is None
    )


def test_basket_is_a_local_optimum(topic: TopicMeta, book: LmsrBook) -> None:
    model = model_tilted(topic, book)
    solution = optimize_basket(
        book, model, sub_bankroll_points=35_000.0, max_cost_points=1e5, fee_rate=0.02, ambiguity=0.0
    )
    assert solution is not None
    ids = [leg.option_id for leg in solution.legs]
    p = np.array([model[o] for o in ids])
    w0 = solution.sub_bankroll_points
    base = cost(book.shares, book.alpha)

    def growth(delta: np.ndarray) -> float:
        after = book.shares.copy()
        for o, d in zip(ids, delta, strict=True):
            after[book.index_of(o)] += d
        spend = 1.02 * (cost(after, book.alpha) - base)
        w = np.append(w0 - spend + PI * delta, w0 - spend)
        return float(np.append(p, 1.0 - p.sum()) @ np.log(w))

    delta = np.array([leg.contracts for leg in solution.legs])
    best = growth(delta)
    for i in range(delta.size):
        for step in (-0.5, 0.5):
            moved = delta.copy()
            moved[i] = max(moved[i] + step, 0.0)
            assert growth(moved) <= best + 1e-7
