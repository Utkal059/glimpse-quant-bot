"""LS-LMSR replica, slippage-ladder optimisation and the 0.001 BTC micro-bankroll caps."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest
from glimpse_markets.money import Millisats, PriceUnits

from glimpse_quant_bot.config import LiveTradingNotArmedError, Settings
from glimpse_quant_bot.ledger import Ledger
from glimpse_quant_bot.lmsr import (
    GLIMPSE_ALPHA,
    Leg,
    LmsrBook,
    alpha_for,
    book_from_prices,
    buy_commission,
    cost,
    max_contracts_below_price,
    sell_commission,
)
from glimpse_quant_bot.lmsr_optimizer import (
    CostQuote,
    LocalEstimator,
    SlippageOptimizer,
    evaluate_exit,
    kelly_fraction,
    ladder_point,
    settlement_value,
)
from glimpse_quant_bot.risk_manager import RiskManager
from glimpse_quant_bot.types import (
    AssetType,
    BucketSignal,
    Position,
    TopicMeta,
    classify_asset,
    millisats_to_price_points,
    price_points_to_millisats,
)

from .conftest import NOW

BANKROLL = Millisats(100_000_000)  # 0.001 BTC


class CountingEstimator(LocalEstimator):
    """A stand-in for ``/trades/estimate`` that records every size it is asked for."""

    def __init__(self, book: LmsrBook, settings: Settings) -> None:
        super().__init__(
            lambda _topic_id: book,
            settings.fee_rate,
            Millisats(settings.min_buy_commission_millisats),
        )
        self.sizes: list[float] = []

    async def quote_buy(self, topic_id: int, legs: Sequence[Leg]) -> CostQuote:
        self.sizes.append(legs[0][1])
        return await super().quote_buy(topic_id, legs)


def signal_for(
    topic: TopicMeta, book: LmsrBook, option_id: int, probability: float
) -> BucketSignal:
    bucket = topic.bucket(option_id)
    assert bucket is not None
    market = book.price(option_id)
    return BucketSignal(
        topic_id=topic.topic_id,
        option_id=option_id,
        name=bucket.name,
        lo=bucket.lo,
        hi=bucket.hi,
        market_price=market,
        model_probability=probability,
        edge_points=PriceUnits(100.0 * probability - float(market)),
    )


# --- LS-LMSR replica ---------------------------------------------------------


def test_replica_reproduces_the_whitepaper_example() -> None:
    """500 ranges, v = 2, 500 contracts per range -- every figure Glimpse publishes."""
    book = LmsrBook.from_shares(1, list(range(1, 501)), [500.0] * 500)
    assert book.alpha == pytest.approx(2.0 / (500 * np.log(500)))
    assert book.liquidity == pytest.approx(160.9, abs=0.05)
    assert cost(book.shares, book.alpha) == pytest.approx(150_000.0)  # operator's max loss
    assert float(book.prices().sum()) == pytest.approx(300.0)
    assert float(book.price(1)) == pytest.approx(0.60)
    # Figure 1: 900 contracts on one range cost 7,169 and move the price to 34.84.
    assert int(book.buy_cost([(1, 900.0)])) / 1000 == pytest.approx(7169.4, abs=0.05)
    assert float(book.price_after_buy(1, 900.0)) == pytest.approx(34.84, abs=0.005)
    # Section 5: 200 contracts on each of 3 ranges cost 475.9.
    assert int(book.buy_cost([(100, 200.0), (101, 200.0), (102, 200.0)])) / 1000 == pytest.approx(
        475.9, abs=0.05
    )
    # Section 4: Alice pays 1,126 including the 2% fee for 200 contracts on 7 ranges.
    alice = int(book.buy_cost([(i, 200.0) for i in range(75, 82)]))
    assert alice * 1.02 / 1000 == pytest.approx(1126.0, abs=0.5)


def test_production_subsidy_is_twelve_thousand_sats() -> None:
    book = LmsrBook.from_shares(1, list(range(1, 501)), [40.0] * 500)
    assert cost(book.shares, book.alpha) == pytest.approx(12_000.0)
    assert alpha_for(500) == pytest.approx(0.00064364476, rel=1e-6)


def test_alpha_is_one_venue_wide_constant() -> None:
    """Glimpse runs alpha = 2 / (500 ln 500) on every market, whatever its bucket count.

    The live 650-bucket Solana markets (40 seeded contracts per bucket) report a
    ``subsidy_amount`` of 14,839,060 millisats; only the fixed alpha reproduces it."""
    assert pytest.approx(alpha_for(500)) == GLIMPSE_ALPHA
    solana = LmsrBook.from_shares(1, list(range(1, 651)), [40.0] * 650)
    assert solana.alpha == GLIMPSE_ALPHA
    assert cost(solana.shares, solana.alpha) * 1000 == pytest.approx(14_839_060, abs=1.0)
    wrong = cost(solana.shares, alpha_for(650)) * 1000
    assert wrong == pytest.approx(12_000_000, abs=1.0)


def test_cost_is_path_independent_and_round_trips(book: LmsrBook) -> None:
    whole = int(book.buy_cost([(200, 20.0)]))
    first = int(book.buy_cost([(200, 8.0)]))
    second = int(book.applied([(200, 8.0)]).buy_cost([(200, 12.0)]))
    assert whole == pytest.approx(first + second, abs=2)  # millisat rounding only
    after = book.applied([(200, 20.0)])
    assert int(after.sell_proceeds([(200, 20.0)])) == pytest.approx(whole, abs=1)


def test_marginal_price_rises_with_size(book: LmsrBook) -> None:
    prices = [float(book.price_after_buy(200, k)) for k in (0.001, 5.0, 20.0, 60.0)]
    assert prices == sorted(prices)
    average_20 = int(book.buy_cost([(200, 20.0)])) / 20_000
    assert float(book.price(200)) < average_20 < float(book.price_after_buy(200, 20.0))


def test_commission_schedule() -> None:
    assert buy_commission(Millisats(18_034), 0.02, Millisats(1000)) == 1000  # 1 sat minimum
    assert buy_commission(Millisats(443_035), 0.02, Millisats(1000)) == 8861
    assert sell_commission(Millisats(84_503), 0.02) == 1690


def test_max_contracts_below_price_is_the_marginal_break_even(book: LmsrBook) -> None:
    limit = PriceUnits(float(book.price(200)) + 6.0)
    size = max_contracts_below_price(book, 200, limit)
    assert size > 0
    assert float(book.price_after_buy(200, size)) == pytest.approx(float(limit), abs=1e-3)
    assert max_contracts_below_price(book, 200, PriceUnits(float(book.price(200)) - 1.0)) == 0.0


def test_book_from_prices_recovers_the_quoted_distribution(book: LmsrBook) -> None:
    # Recorded quotes are the forecast 100 * w (the uniform spread rounds away).
    quotes = [100.0 * float(w) for w in book.probabilities()]
    rebuilt = book_from_prices(1, book.option_ids, quotes, book.liquidity)
    assert rebuilt.liquidity == pytest.approx(book.liquidity, rel=1e-6)
    top = int(np.argmax(book.probabilities()))
    assert float(rebuilt.probabilities()[top]) == pytest.approx(
        float(book.probabilities()[top]), abs=0.01
    )
    assert float(np.min(rebuilt.shares)) >= 0.0


# --- unit separation ---------------------------------------------------------


def test_price_points_and_millisats_never_mix() -> None:
    # One price point on one contract is one sat.
    assert price_points_to_millisats(PriceUnits(25.0), 4.0) == 100_000
    assert float(millisats_to_price_points(Millisats(100_000), 4.0)) == pytest.approx(25.0)
    with pytest.raises(ValueError, match="positive"):
        millisats_to_price_points(Millisats(1), 0.0)
    point = ladder_point(
        5.0, CostQuote(Millisats(94_168), Millisats(1_883), "server"), 0.30, fee_rate=0.02
    )
    assert int(point.all_in_millisats) == 96_051
    assert float(point.exec_price) == pytest.approx(19.2102)
    # Fair value 100 * 0.30 * 0.98 = 29.4 points per contract.
    assert float(point.net_edge_points) == pytest.approx(29.4 - 19.2102)
    assert int(point.expected_profit_millisats) == 5 * 29_400 - 96_051


def test_kelly_fraction_formula() -> None:
    assert float(settlement_value(0.30, 0.02)) == pytest.approx(29.4)
    # f* = (p * pi - c) / (pi - c) with pi = 98.
    assert kelly_fraction(0.30, PriceUnits(20.0), 0.02) == pytest.approx((29.4 - 20.0) / 78.0)
    assert kelly_fraction(0.10, PriceUnits(20.0), 0.02) == 0.0
    assert kelly_fraction(0.99, PriceUnits(99.0), 0.02) == 0.0


# --- slippage optimiser ------------------------------------------------------


async def test_optimizer_picks_the_profit_maximising_size(
    make_settings: Callable[..., Settings], topic: TopicMeta, book: LmsrBook
) -> None:
    settings = make_settings(min_expected_gain_sats=20.0)
    estimator = CountingEstimator(book, settings)
    optimizer = SlippageOptimizer(settings, estimator)
    signal = signal_for(topic, book, 200, float(book.probabilities()[199]) + 0.15)

    result = await optimizer.optimize(
        signal,
        book=book,
        bankroll=BANKROLL,
        max_cost=Millisats(15_000_000),
        full_server_ladder=True,
    )
    plan = result.plan
    assert plan is not None
    # Every configured rung was priced on the "server" ...
    assert set(settings.ladder) <= set(estimator.sizes)
    # ... and the chosen rung has the highest expected profit of all of them.
    best = max(result.ladder, key=lambda point: int(point.expected_profit_millisats))
    assert plan.contracts == best.contracts
    assert int(plan.expected_profit_millisats) == int(best.expected_profit_millisats)
    # Expected profit is concave in size: it rises to k* and falls after it.
    profits = [int(point.expected_profit_millisats) for point in result.ladder]
    peak = profits.index(max(profits))
    assert profits[: peak + 1] == sorted(profits[: peak + 1])
    assert profits[peak:] == sorted(profits[peak:], reverse=True)
    assert float(plan.net_edge_points) >= settings.min_net_edge_points
    assert int(plan.expected_profit_millisats) >= int(settings.min_expected_gain_millisats)


async def test_local_screen_confirms_only_a_few_sizes_on_the_server(
    make_settings: Callable[..., Settings], topic: TopicMeta, book: LmsrBook
) -> None:
    settings = make_settings(min_expected_gain_sats=20.0, server_confirm_points=3)
    estimator = CountingEstimator(book, settings)
    optimizer = SlippageOptimizer(settings, estimator)
    signal = signal_for(topic, book, 200, float(book.probabilities()[199]) + 0.15)
    result = await optimizer.optimize(
        signal, book=book, bankroll=BANKROLL, max_cost=Millisats(15_000_000)
    )
    assert result.plan is not None
    assert 1 <= len(estimator.sizes) <= 3
    assert len(result.local_ladder) > len(estimator.sizes)


async def test_hopeless_candidates_cost_zero_api_calls(
    make_settings: Callable[..., Settings], topic: TopicMeta, book: LmsrBook
) -> None:
    settings = make_settings(min_expected_gain_sats=250.0)  # a floor no small edge clears
    estimator = CountingEstimator(book, settings)
    optimizer = SlippageOptimizer(settings, estimator)
    small_edge = signal_for(topic, book, 200, float(book.probabilities()[199]) + 0.04)
    result = await optimizer.optimize(
        small_edge, book=book, bankroll=BANKROLL, max_cost=Millisats(15_000_000)
    )
    assert result.plan is None
    assert result.rejection is not None
    assert result.rejection.startswith("local screen")
    assert estimator.sizes == []

    no_edge = signal_for(topic, book, 200, float(book.probabilities()[199]) + 0.01)
    rejected = await optimizer.optimize(
        no_edge, book=book, bankroll=BANKROLL, max_cost=Millisats(15_000_000)
    )
    assert rejected.rejection == "raw edge below threshold"


async def test_fee_floor_blocks_micro_trades(
    make_settings: Callable[..., Settings], topic: TopicMeta, book: LmsrBook
) -> None:
    """Net edge must clear 2.5 points after the 2% commission and the 2% settlement fee."""
    settings = make_settings(min_expected_gain_sats=0.0, min_raw_edge_points=0.0)
    optimizer = SlippageOptimizer(settings, CountingEstimator(book, settings))
    thin = signal_for(topic, book, 200, float(book.probabilities()[199]) + 0.02)
    result = await optimizer.optimize(
        thin, book=book, bankroll=BANKROLL, max_cost=Millisats(15_000_000), full_server_ladder=True
    )
    assert result.plan is None
    assert all(float(p.net_edge_points) < settings.min_net_edge_points for p in result.ladder)


async def test_budget_caps_the_size(
    make_settings: Callable[..., Settings], topic: TopicMeta, book: LmsrBook
) -> None:
    settings = make_settings(min_expected_gain_sats=5.0)
    optimizer = SlippageOptimizer(settings, CountingEstimator(book, settings))
    signal = signal_for(topic, book, 200, float(book.probabilities()[199]) + 0.20)
    roomy = await optimizer.optimize(
        signal,
        book=book,
        bankroll=BANKROLL,
        max_cost=Millisats(15_000_000),
        full_server_ladder=True,
    )
    tight = await optimizer.optimize(
        signal, book=book, bankroll=BANKROLL, max_cost=Millisats(150_000), full_server_ladder=True
    )
    assert roomy.plan is not None
    assert tight.plan is not None
    assert tight.plan.contracts < roomy.plan.contracts
    assert int(tight.plan.all_in_millisats) <= 150_000
    # Model disagreement shrinks the Kelly fraction proportionally.
    cautious = await optimizer.optimize(
        signal,
        book=book,
        bankroll=BANKROLL,
        max_cost=Millisats(15_000_000),
        kelly_multiplier=0.25,
        full_server_ladder=True,
    )
    assert cautious.plan is not None
    assert cautious.plan.kelly_fraction == pytest.approx(roomy.plan.kelly_fraction * 0.25)


# --- micro-bankroll caps -----------------------------------------------------


def test_bucket_and_topic_caps_on_a_0_001_btc_bankroll(settings: Settings, ledger: Ledger) -> None:
    risk = RiskManager(settings, ledger, "dry_run", clock=lambda: NOW)
    cash = BANKROLL
    # Empty book: 15% of 100,000 sats = 15,000 sats per bucket.
    assert risk.entry_budget(1, 200, BANKROLL, cash, {}) == 15_000_000

    def held(option_id: int, sats: int) -> Position:
        return Position(1, option_id, "x", 10.0, Millisats(sats * 1000), 0.3)

    positions = {(1, 200): held(200, 9_000)}
    assert risk.entry_budget(1, 200, BANKROLL, cash, positions) == 6_000_000
    # A second bucket of the same topic still has its own 15% ...
    assert risk.entry_budget(1, 201, BANKROLL, cash, positions) == 15_000_000
    # ... until the 35% topic cap binds across correlated buckets.
    positions = {(1, 200): held(200, 15_000), (1, 201): held(201, 15_000)}
    assert risk.entry_budget(1, 202, BANKROLL, cash, positions) == 5_000_000
    positions[(1, 202)] = held(202, 5_000)
    assert risk.entry_budget(1, 203, BANKROLL, cash, positions) == 0
    # Another topic is unaffected, but cash is always a hard limit.
    assert risk.entry_budget(2, 200, BANKROLL, cash, positions) == 15_000_000
    assert risk.entry_budget(2, 200, BANKROLL, Millisats(2_000_000), positions) == 2_000_000


# --- exits -------------------------------------------------------------------


def test_exit_rules(settings: Settings) -> None:
    position = Position(1, 200, "64800-65000", 10.0, Millisats(200_000), 0.30)  # entry 20.0
    hold = evaluate_exit(position, 0.30, PriceUnits(24.0), Millisats(220_000), settings)
    assert not hold.should_exit
    assert hold.reason == "hold"
    # Market within 1.0 point of the model and the sale is profitable -> take profit.
    take = evaluate_exit(position, 0.30, PriceUnits(29.2), Millisats(260_000), settings)
    assert take.should_exit
    assert take.reason == "take_profit"
    # Converged but selling would lose money -> keep holding.
    under = evaluate_exit(position, 0.30, PriceUnits(29.2), Millisats(190_000), settings)
    assert not under.should_exit
    # Model fell more than 4.0 points below the 20.0 entry -> thesis invalidated.
    stop = evaluate_exit(position, 0.155, PriceUnits(15.0), Millisats(140_000), settings)
    assert stop.should_exit
    assert stop.reason == "invalidation"
    edge_case = evaluate_exit(position, 0.165, PriceUnits(10.0), Millisats(90_000), settings)
    assert not edge_case.should_exit


# --- configuration safety ----------------------------------------------------


def test_live_trading_needs_both_opt_ins(make_settings: Callable[..., Settings]) -> None:
    default = make_settings()
    assert default.glimpse_live_trading is False
    assert default.resolve_mode(cli_live_flag=False) == "dry_run"
    with pytest.raises(LiveTradingNotArmedError):
        default.resolve_mode(cli_live_flag=True)

    env_only = make_settings(glimpse_live_trading=True, glimpse_api_key="glp_test")
    assert env_only.resolve_mode(cli_live_flag=False) == "dry_run"
    assert env_only.resolve_mode(cli_live_flag=True) == "live"

    no_key = make_settings(glimpse_live_trading=True)
    with pytest.raises(LiveTradingNotArmedError):
        no_key.resolve_mode(cli_live_flag=True)


def test_default_bankroll_and_limits(settings: Settings) -> None:
    assert settings.bankroll_millisats == 100_000_000
    assert settings.min_expected_gain_millisats == 25_000
    assert settings.min_net_edge_points == 2.5
    assert settings.ladder == (1.0, 2.0, 5.0, 10.0, 20.0)
    assert (settings.kelly_fraction, settings.max_bucket_fraction) == (0.35, 0.15)
    assert (settings.max_topic_fraction, settings.daily_drawdown_limit) == (0.35, 0.06)
    assert settings.glimpse_requests_per_minute <= 60


def test_asset_classification() -> None:
    assert classify_asset("Hourly Bitcoin Prediction Markets") is AssetType.BTC
    assert classify_asset("Daily Ethereum Market") is AssetType.ETH
    assert classify_asset("Daily Solana Market") is AssetType.SOL
    assert classify_asset("Daily PAX Gold Market") is AssetType.GOLD
    assert (
        classify_asset("Hourly Gold Prediction Markets", "Predict the price of PAXG")
        is AssetType.GOLD
    )
    # Cross-quoted markets settle on a ratio, not a USD price: not tradable.
    assert classify_asset("Hourly Tether Gold in sats per troy ounce") is None
    assert classify_asset("Hourly WTI/BTC Prediction Markets") is None
    assert classify_asset("Will it rain tomorrow?") is None
    assert classify_asset(None, "") is None
