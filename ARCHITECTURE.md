# Architecture

This document explains how `glimpse-quant-bot` is put together, where each design decision came from, and what was measured against the live Glimpse API while building it.

Contents: [1 System overview](#1-system-overview) · [2 Design lineage: US, Europe, China](#2-design-lineage) · [3 LS-LMSR mathematics](#3-ls-lmsr-mathematics) · [4 Forecasting](#4-forecasting) · [5 Sizing and exits](#5-sizing-and-exits) · [6 Risk and state reconciliation](#6-risk-and-state-reconciliation) · [7 Ledger](#7-ledger) · [8 Rate-limit budget](#8-rate-limit-budget) · [9 Latency](#9-latency-and-why-nanoseconds-do-not-matter-here) · [10 Backtesting](#10-backtesting) · [11 Facts measured on the live API](#11-facts-measured-on-the-live-api) · [12 Deviations from the brief](#12-deviations-from-the-original-brief) · [13 References](#13-references)

## 1. System overview

Data flows one way. No module imports from a module to its right.

```text
             external_feeds.py        forecaster.py            lmsr_optimizer.py       risk_manager.py        execution.py
            ┌────────────────┐      ┌─────────────────┐      ┌──────────────────┐    ┌───────────────┐     ┌────────────────┐
 Deribit ──▶│                │      │ options_density │      │ local LS-LMSR    │    │ trading lock  │     │ enter_multi_   │
 Binance ──▶│ MarketDataHub  │─────▶│ HAR + skewed-t  │─────▶│ screen (lmsr.py) │───▶│ drawdown      │────▶│ topic_multi_leg│──▶ Glimpse
 Coinbase ─▶│ TTL cache,     │      │ GJR-GARCH / TFM │      │ server estimate  │    │ breaker, caps,│     │ exit_          │
 Kraken ───▶│ failover       │      │ linear pool     │      │ ladder, Kelly    │    │ flow control  │     │ consolidated   │
            └────────────────┘      └─────────────────┘      └──────────────────┘    └───────┬───────┘     └───────┬────────┘
                                             ▲                        ▲                      │ reconcile           │
 Glimpse WebSocket ── on_quote ──▶ strategy.py (LmsrBook per topic) ──┘                      ▼                     ▼
 Glimpse REST (45 s) ─ on_tick ──▶ universe refresh, exits, resolution            ledger.py — append-only SQLite + NDJSON
```

| Stage | Module | Responsibility | Never does |
|---|---|---|---|
| MarketDataHub | `external_feeds.py` | Spot, candles, order books, funding, option chains; caching; venue failover | Know anything about Glimpse |
| AlphaEnsemble | `options_density.py`, `forecaster.py` | Turn market data into bucket probabilities | Look at Glimpse prices |
| SlippageOptimizer | `lmsr.py`, `lmsr_optimizer.py` | Turn a probability gap into a sized, priced plan | Decide whether trading is allowed |
| RiskGatekeeper | `risk_manager.py` | Approve or reject plans; own the lock, the breaker and reconciliation | Call a trade endpoint |
| ExecutionRouter | `execution.py` | The only caller of trade-mutating SDK methods | Retry anything |
| Orchestration | `strategy.py` | `glimpse_markets.Strategy` hooks; holds the books | Contain pricing or risk logic |

The forecaster is a pure function of its inputs (`build_forecast`), and the optimiser, risk manager, exit rules and ledger take an injected clock. That is what lets `backtester.py` drive the *same objects* as live trading.

## 2. Design lineage

Each subsection lists what was studied, the specific pattern adopted, and what was left out and why.

### 2.1 United States — event engines and LMSR market making

**NautilusTrader** ([docs](https://nautilustrader.io/docs/latest/concepts/overview/), [architecture](https://docs.nautilustrader.io/concepts/architecture.html)). A Rust core with a deterministic, event-driven kernel that is shared by backtest, sandbox and live contexts; the project describes research-to-live parity as structural because the engine, event ordering and time handling are common.

- *Adopted:* parity by construction. `Backtester.run` instantiates the real `SlippageOptimizer`, `RiskManager`, `Ledger` and `evaluate_exit` and calls the real `build_forecast`; only the price source and the fill pricer are swapped. Time is an injected `clock` callable everywhere, never `time.time()` buried in logic.
- *Adopted:* one-way message flow between components rather than components calling back into each other.
- *Left out:* a message bus, nanosecond timestamps and a compiled core. One process, one WebSocket and a 60-requests-per-minute venue do not need them (§9).

**Hummingbot** ([Avellaneda-Stoikov guide](https://www.hummingbot.org/academy-content/guide-to-the-avellaneda--stoikov-strategy/), [inventory skew](https://hummingbot.org/strategies/v1-strategies/strategy-configs/inventory-skew/)). Its market-making strategy implements the Avellaneda-Stoikov reservation price `r = s − q·γ·σ²·(T − t)`: the more inventory `q` you hold and the longer until the session ends, the further you shade your quote against your own position.

- *Adopted:* the idea that held inventory must reduce appetite for more. `RiskManager.entry_budget` shrinks the budget for a bucket by what is already held in that bucket and in its topic (correlated buckets of one expiry), and the exit rule tightens as the market converges on the model.
- *Left out:* two-sided quoting. On Glimpse the venue is the market maker and entries are buy-only, so there is no bid/ask to skew.

**LMSR market makers** (Hanson 2003; Othman, Sandholm, Pennock and Reeves 2010; Glimpse whitepaper). The cost-function literature gives closed forms for price, cost and worst-case loss.

- *Adopted:* an exact local replica of the venue's cost function (§3), used to price any number of candidate sizes offline.
- *Adopted:* from the equilibrium result of Chen et al. (2010) quoted in the whitepaper — trade toward your true belief, no further than where the marginal price reaches it. `max_contracts_below_price` solves for exactly that size by bisection and adds it to the ladder.

**Prediction-market arbitrage evidence.** Saguillo et al. (2025) document persistent intra-market and combinatorial mispricing on Polymarket, with roughly $40M extracted in a year. That is the empirical case that mispricings in such markets are real and repeatable, and also that they are competed for.

### 2.2 Europe — options-implied densities and venue abstraction

**Deribit and the risk-neutral-density literature.** Deribit is the dominant crypto options venue and the data source of most academic work on Bitcoin options (Almeida et al., [arXiv:2410.15195](https://arxiv.org/html/2410.15195v2); Winkel and Härdle, [*Risks* 2023](https://mdpi.com/2227-9091/11/5/85)). The standard pipeline, also described in practitioner write-ups such as [Amberdata's](https://blog.amberdata.io/ad-derivatives-fresh-feature-pdf-cdf-svi-calibrated), is: smooth the smile, regenerate call prices, apply Breeden-Litzenberger. Naive second differences of raw quotes are numerically unstable.

- *Adopted:* exactly that pipeline in `options_density.py` (§4.1), fed by Deribit's public `get_book_summary_by_currency` (mark IV and the per-expiry forward).
- *Adapted:* the smile is a weighted quadratic in standardised moneyness rather than SVI (Gatheral and Jacquier). SVI's five parameters are poorly identified on Deribit's shortest expiries, which may have nine usable strikes; a three-parameter fit is stable there. SVI's key property — wings linear in total variance — is kept by construction through the extrapolation rule.

**ccxt** — one interface over many exchanges.

- *Adopted:* the abstraction, not the dependency. `_Venue` defines `ticker`, `candles`, `orderbook`; `BinanceSpot`, `CoinbaseSpot` and `KrakenSpot` implement it; `MarketDataHub` takes the median spot across healthy venues and falls through a priority list for everything else, with a per-venue circuit breaker and stale-on-error caching.
- *Left out:* ccxt itself. Four endpoints per venue did not justify a dependency of that size.

**vectorbt** — vectorised research.

- *Adopted:* array-at-a-time numerics. Densities, bucket probabilities, the HAR regression and the GARCH simulation operate on whole NumPy arrays; a full 500-bucket forecast takes a few milliseconds once the slow fit is cached.

### 2.3 China — modular pipelines and pre-trade risk

**vn.py / VeighNa** ([README](https://github.com/vnpy/vnpy)). An event engine at the core, gateways that hide venue differences, strategy apps on top, and a separate risk-manager module that enforces order-flow limits, order size limits and active-order counts *in front of* the gateway.

- *Adopted:* the gatekeeper position and its rule types. Nothing reaches `ExecutionRouter` without `RiskManager.check_entry`, which enforces orders-per-minute flow control, a per-bucket re-entry cooldown, a maximum order size, a maximum number of open positions and a time-to-expiry window, and reports *every* violated rule rather than the first.
- *Adopted:* the gateway idea — `execution.py` is the single module that knows how the venue is called.

**Microsoft Qlib** ([repo](https://github.com/microsoft/qlib)). A research pipeline with hard interfaces between data, factor computation (for example the Alpha158 handler), model, strategy and executor.

- *Adopted:* the stage boundaries and their order: `MarketDataHub → AlphaEnsemble → SlippageOptimizer → RiskGatekeeper → ExecutionRouter`. Each stage consumes a typed object from the previous one (`FastInputs`/`SlowFit` → `EnsembleForecast` → `BucketSignal` → `TradePlan` → `LedgerRecord`).
- *Adopted in miniature:* a factor layer. Garman-Klass variance, the three HAR horizons (1 h, 24 h, 168 h), sample kurtosis, perp funding and L2 depth imbalance are computed as named features before the density is built.
- *Left out:* a learned model over hundreds of factors. There is no labelled Glimpse history to train on yet; `record` exists to create one.

**Also consulted:** freqtrade's protections (`MaxDrawdown`, `StoplossGuard`, cooldowns) and its dry-run-first convention informed the drawdown kill-switch and the default mode. (freqtrade is not a Chinese project; it is listed here because the brief grouped it with risk-manager designs.)

## 3. LS-LMSR mathematics

### 3.1 Cost and price

A topic has `n` buckets and state `q = (q₁ … qₙ)`, the contracts outstanding on each.

```text
C(q) = 100 · b(q) · ln Σⱼ exp(qⱼ / b(q))            (cost function, in price points = sats)
b(q) = α · Σⱼ qⱼ                                    (depth grows with volume)
α    = v / (n ln n),   v = 2                        (so the opening state costs 100·q₀·(v+1))

wᵢ   = exp(qᵢ/b) / Σⱼ exp(qⱼ/b)                     (the market's probability forecast)
pᵢ   = ∂C/∂qᵢ = 100 · [ wᵢ + α · H(w) ],   H(w) = −Σⱼ wⱼ ln wⱼ
```

Derivation of the price: with `S = Σ exp(qⱼ/b)` and `zⱼ = qⱼ/b`, `∂C/∂qᵢ = 100·[α ln S + wᵢ − α Σⱼ wⱼ zⱼ]`, and `ln S − Σ wⱼ zⱼ = H(w)`. The second term is a spread common to all buckets; it is widest for a uniform forecast and shrinks as the forecast concentrates.

A trade from `q` to `q′` costs `C(q′) − C(q)` regardless of how it is split. Fees observed on the live endpoint:

```text
buy :  commission = max(1 sat, 2% · cost)          debit  = cost + commission
sell:  commission = 2% · proceeds                  credit = proceeds − commission
settlement: a winning contract pays 100 · (1 − 2%) = 98 points
```

### 3.2 Verification

`lmsr.py` was checked two ways.

Against the whitepaper's worked example (500 ranges, `v = 2`, 500 contracts each) — asserted in `tests/test_lmsr_optimizer.py`:

| Quantity | Whitepaper | Replica |
|---|---:|---:|
| `b` | 160.9 | 160.91 |
| Operator's maximum loss `C(q₀)` | ₿150,000 | 150,000.0 |
| Sum of prices | ₿300.0 | 300.00 |
| Opening price | ₿0.60 | 0.600 |
| 900 contracts on one range | ₿7,169 | 7,169.4 |
| Price after that trade | ₿34.84 | 34.839 |
| 200 contracts on each of 3 ranges | ₿475.9 | 475.90 |
| Alice: 200 × 7 ranges, with fee | ₿1,126 | 1,126.0 |

Against the live `/trades/estimate` endpoint (topic 5398, `Σq = 58,432.57`, `b = 37.61`):

| Trade | Local replica (sats) | Server (sats) |
|---|---:|---:|
| Buy 1 | 18.034 | 18.034 |
| Buy 5 | 94.168 | 94.168 |
| Buy 20 | 443.035 | 443.035 |
| Buy 60 | 1,991.422 | 1,991.422 |
| Sell 1 | 17.647 | 17.647 |
| Sell 5 | 84.503 | 84.503 |
| Buy 3 + 3 on two buckets | 103.202 | 103.202 |
| Buy 2.5 (fractional) | 45.824 | 45.824 |

### 3.3 What depth implies for capacity

For a bucket at market probability `m` that the model puts at `p`, buying until the marginal price reaches `p` earns approximately

```text
profit ≈ 100 · b · [ p ln(p/m) + (1−p) ln((1−p)/(1−m)) ]  sats  =  100 · b · KL(p ‖ m)
```

With `b = 37`, `m = 0.15`, `p = 0.30` that is about 266 sats; for `m = 0.20`, `p = 0.26` it is about 39 sats. This is why the minimum-expected-gain default is 25 sats and not 250 (§12).

## 4. Forecasting

### 4.1 Options risk-neutral density (`options_density.py`)

1. **Per-expiry smile.** Keep out-of-the-money marks with `|z| ≤ 4`, where `z = ln(K/F) / (σ_atm √T)`. Fit `σ(z) = a + b·z + c·z²` by weighted least squares (Gaussian weights centred at the money); refit as a line if `c < 0`.
2. **Wings.** Beyond the quoted strikes, total variance `w = σ²T` continues *linearly in log-moneyness* with the slope the fit has at the boundary, so `σ` and `dσ/dK` are continuous and `∂²C/∂K²` has no kink. The slope is capped at `|dw/dk| ≤ 1.9`, inside Roger Lee's bound of 2. (An earlier version clipped the smile flat; the jump in `dσ/dK` put a negative spike in the density and made the raw CDF dip by 1.5%.)
3. **Term structure.** Between expiries, interpolate total variance linearly at fixed `z`. Below the first expiry — hourly Glimpse markets usually sit below Deribit's nearest daily expiry — reuse the first slice's shape with flat forward volatility. Expiries less than 30 minutes away are dropped as too noisy.
4. **Density.** Evaluate Black-76 prices on an 8,001-point log-strike grid spanning ±12 at-the-money standard deviations, doubling the span while more than 10⁻¹⁰ of the mass remains outside. Take the three-point second derivative, floor it at zero, and divide by its `np.trapezoid` integral. The CDF is the running integral — non-decreasing from 0 to 1 by construction.
5. **Buckets.** `P(a < S_T ≤ b) = F(b) − F(a)`, normalised over the topic's buckets.

Linear-variance wings are fatter than flat ones. That is deliberate and standard, but it does put slightly more mass in far buckets than the previous version did.

### 4.2 HAR-RV + microstructure density

- **Variance.** Garman-Klass variance per hourly bar; log-HAR regression on lags of 1 h, the 24 h mean and the 168 h mean; iterated forward bar by bar to the market's expiry; fractional final bar. The back-transform uses a ratio calibration `Σ RV / Σ exp(x̂)`, because Duan's smearing overshoots on left-skewed log-variance residuals (it forecast 0.74%/h for BTC against 0.42–0.58% realised; the ratio version forecast 0.68%).
- **Shape.** A Fernández-Steel two-piece skewed Student-t, standardised to unit variance. Degrees of freedom come from sample kurtosis, relaxed toward Gaussian as `1/√horizon`. Used instead of a Cornish-Fisher expansion because its CDF is monotone for every parameter value.
- **Microstructure tilt.** Positive perpetual funding (crowded longs) skews the density left via `γ = exp(−0.08·tanh(f / 0.03%))`. L2 depth imbalance within 15 bps of mid adds a drift of at most `0.15·σ`, decaying as `√(5 min / horizon)` — imbalance is informative over minutes, not hours (Cont, Kukanov and Stoikov 2014; Bieganowski 2026).

### 4.3 GJR-GARCH / TimesFM

GJR-GARCH(1,1) by Gaussian quasi-maximum-likelihood (L-BFGS-B, stationarity enforced), then filtered historical simulation: 3,000 paths bootstrapped from standardised residuals, re-centred to a martingale. If `glimpse-markets[forecasting]` is installed, the SDK's `TimesFMForecaster` (TimesFM 2.5, Apache-2.0) supplies deciles instead.

### 4.4 Pooling

A linear opinion pool — a weighted mixture of the component CDFs — with weights 50/30/20 where options exist and 60/40 otherwise, renormalised if a component is missing. The pooled quantile function (399 levels) is mapped onto buckets with the SDK's `bucket_probabilities()`, then normalised to sum to one. The weighted Jensen-Shannon divergence between components, normalised to `[0, 1]`, is reported as `disagreement`; the Kelly multiplier is `clip(1 − 2·JSD, 0.25, 1)`.

## 5. Sizing and exits

For each bucket with raw edge `100·p − price ≥ 3` points:

1. **Budget.** `min(risk caps, 0.35 · κ · f* · equity)` with `f* = (p·π − c)/(π − c)`, `π = 98`.
2. **Candidate sizes.** The configured ladder `[1, 2, 5, 10, 20]` plus 50%, 75% and 100% of the analytic optimum (the size at which the marginal all-in price equals fair value).
3. **Local screen.** Price every candidate on the replica. If none clears the thresholds, stop — no API request was made.
4. **Server confirmation.** Price the best three candidates with `/trades/estimate` (the full ladder in `demo-trades`). The decision uses server numbers only.
5. **Choose** the feasible size with the highest expected profit in millisats.

Expected profit is concave in size, so the ladder brackets a single maximum; a test asserts this shape.

Exit rules (`evaluate_exit`), checked every tick against the local book:

| Rule | Condition | Rationale |
|---|---|---|
| Take profit | `P_market ≥ P_model − 1.0` and net sale proceeds exceed cost | The remaining edge no longer pays for the variance of a binary settlement; recycle the bankroll |
| Invalidation | `100·P_model < entry price − 4.0` | The thesis is gone |
| Otherwise | hold | |

## 6. Risk and state reconciliation

### 6.1 Gatekeeper rules

| Rule | Default | Blocks exits? |
|---|---|---|
| Trading lock (exchange state uncertain) | – | Yes |
| Daily drawdown kill-switch | −6% from the UTC day's equity peak | No |
| Per-bucket exposure | 15% of equity | No |
| Per-topic exposure | 35% of equity | No |
| Orders per minute | 6 | No |
| Re-entry cooldown per bucket | 120 s | No |
| Open positions | 12 | No |
| Time to expiry | 240 s to 30 h | No |

Equity is cash plus the *liquidation* value of positions (what selling into the book would net), not mark-to-model.

### 6.2 Ambiguous trade state

Glimpse has no idempotency key. When a trade-mutating request fails at the network layer the SDK raises `GlimpseAmbiguousTradeStateError`: the order may or may not have executed.

```text
ExecutionRouter.enter(plan)
        │
        ├── SDK returns ──────────────────────────────▶ ledger.record_entry ─▶ PositionTracker.invalidate()
        │
        └── GlimpseAmbiguousTradeStateError
                 │
                 ▼
          RiskManager.lock()                 ◀── no entry and no exit can pass while locked
          ledger: "…; order NOT retried"
                 │
                 ▼
          client.portfolio_active() ────── fails ──▶ stay LOCKED; on_tick retries the reconcile
                 │ ok
                 ▼
          diff exchange positions against ledger.positions()
                 │
          ┌──────┴───────────────────────────────┐
          ▼                                      ▼
   exchange holds the new contracts        exchange unchanged
   → adjustment row with the               → no adjustment
     expected cost (landed = true)           (landed = false)
          └──────────────┬───────────────────────┘
                         ▼
          ledger: reconcile record listing every discrepancy
          RiskManager.unlock()
                         ▼
          trading resumes; the next signal is evaluated against reconciled positions
```

Properties, each covered by a test in `tests/test_risk_reconciler.py`:

- The order is submitted **exactly once** whether or not it landed.
- A landed order is adopted with its expected cost, so caps and P&L stay correct.
- While the portfolio is unreachable nothing is sent to the exchange.
- The same routine runs at start-up in live mode and whenever the cached `PositionTracker` disagrees with the ledger.
- Markets that have ended but not settled are excluded from the diff: they leave the *active* portfolio before their payout is booked.

## 7. Ledger

One table, `records`, append-only: SQLite triggers abort any `UPDATE` or `DELETE`. Every signal, estimate ladder, fill, settlement, adjustment, reconciliation, risk event and equity snapshot is a row, mirrored to an NDJSON audit file. Positions, cash and P&L are derived by replaying rows, which is what makes the ledger a trustworthy reference during reconciliation. Money columns are integer millisats; price columns are 0–100 points.

## 8. Rate-limit budget

Glimpse allows 60 requests per 60 seconds per key; the client is throttled to 50.

| Source | Requests per minute |
|---|---:|
| Universe refresh: one page per tradable batch every 45 s (six batches) | ~8 |
| Resolution check: one `market_quotes` per expired position | 0–3 |
| Confirming estimates: up to 3 per candidate that passes the local screen | 0–20 |
| Fills | 0–6 |
| Wallet poll (live mode) | ~1 |

The WebSocket carries full share vectors, so books update without REST. `PositionTracker` caches the portfolio and is invalidated after every trade attempt.

## 9. Latency, and why nanoseconds do not matter here

The research on speed in markets is about *continuous limit order books*. Budish, Cramton and Shim (2015) show that correlations between related instruments break down at millisecond horizons, creating mechanical arbitrage that goes to the fastest participant. Aquilina, Budish and O'Neill (2022) measure these races on the London Stock Exchange: about one per minute per FTSE-100 symbol, with a modal duration of 5–10 *microseconds*, and estimate the stakes at roughly $5 billion a year globally. Vendor benchmarks (STAC-T0) put FPGA tick-to-trade times in the tens of nanoseconds; those are hardware results, not peer-reviewed research, and no academic study of nanosecond-scale effects in crypto markets was found. For crypto specifically, a University of Bamberg study of Binance triangular arbitrage found roughly 80 ms of delay from a European server and that latency eliminated most of the apparent opportunities.

None of that mechanism exists on Glimpse:

- There is no queue. The market maker's cost function is path-independent, so arriving first buys no priority, only a slightly better point on the same curve.
- Quotes change only when someone trades. The real race is *stale-quote sniping*: spot moves, the Glimpse distribution does not, and whoever notices first buys the newly likely buckets. That race is run over REST at about 100 ms per request under a 60-per-minute limit.

So the engineering effort went where it pays: reacting on the WebSocket event rather than a poll, keeping the forecast path to a few milliseconds through cached slow fits and vectorised math, and pricing locally so a decision needs one to three confirming requests instead of a full ladder.

## 10. Backtesting

`Backtester` replays a recording in fixed steps. At each step it settles expired positions, rebuilds each book from the recording plus the bot's own open positions, refits models on candles that had closed before that time, runs exits then entries through the production components, and marks equity to liquidation value. It reports ROI, annualised Sharpe and Sortino on per-step returns, maximum drawdown and win rate.

Recordings follow the SDK's `MarketRecorder` NDJSON format and are loaded with `MarketRecorder.load`. Because that format holds quotes rather than share vectors, the book is reconstructed from `w ∝ price` under an assumed depth: `qᵢ = b·ln wᵢ + c` with `c` fixed by `Σq = b/α`.

The synthetic world (`synthetic_dataset`) exists to test the machinery against a known truth, with a correctly priced control. Its results are in the README; they are evidence that the pipeline is wired correctly, not a forecast of live returns.

## 11. Facts measured on the live API

Recorded during development (November 2026); re-check before relying on them.

- Eight batches were listed. Six are USD-quoted and tradable: hourly and daily BTC, daily ETH, daily SOL, hourly and daily gold (PAXG). Two are ratio-quoted and excluded: "Hourly Tether Gold in sats per troy ounce" and "Hourly WTI/BTC".
- Topics have 500 buckets (650 for SOL), named `"LO-HI"`. Hourly markets are seeded with 80 contracts per bucket, daily ones with 40.
- `alpha` on the WebSocket is `0.00064364476 = 2 / (500 ln 500)`, and it is the same constant on the 650-bucket Solana markets: their `subsidy_amount` of 14,839.06 sats equals `C(q₀)` under that alpha, whereas `2 / (650 ln 650)` would give 12,000. `lmsr.GLIMPSE_ALPHA` encodes this; deriving alpha from the bucket count mispriced SOL by up to 15%.
- Quoted `yes_price` is rounded to an integer; the exact marginal price follows from the share vector.
- `batch_active_markets_page` ignores the SDK's `page` / `page_size` arguments (the server expects `limit` / `offset`) and returns the 24 nearest-expiry markets with every bucket's share count.
- The feed sends an update only when a trade occurs and has no resolution event; resolution is detected by polling `quote_mode`, and the winning option comes from `batch_resolved_markets_page`.
- Deribit returned an empty chain for SOL, confirming the fallback weighting.
- Observed depth `b`: about 35–40 contracts on hourly BTC and gold, 13–18 on daily markets.

## 12. Deviations from the original brief

| Brief | Implementation | Reason |
|---|---|---|
| Minimum expected gain 250 sats | Default **25 sats** | At `b = 13–40` a trade rarely offers more than ~150 sats; 250 produced zero fills in replay. 25 sats is still 25× the minimum commission. Configurable. |
| Ladder `[1, 2, 5, 10, 20]` | That ladder plus sizes around the analytic optimum | The optimum is often between or beyond the fixed rungs. |
| `estimate_trade` for every rung | Local screen first, server confirmation of the best three (full server ladder in `demo-trades`) | The replica is exact; this saves most of the rate limit. |
| Cornish-Fisher / Student-t | Fernández-Steel skewed Student-t | Monotone CDF for all parameters. |
| Twelve source files | Three more: `lmsr.py`, `execution.py`, `demo.py` | Separation of the pricer, the single order-sending module and the report generator. |
| Hold-to-`quote_mode` polling via `market_quotes` | Universe pages every 45 s, `market_quotes` only for expired positions | One page returns 24 full books. |

## 13. References

Market design and prediction markets

- R. Hanson, "Combinatorial Information Market Design", *Information Systems Frontiers* 5 (2003).
- A. Othman, T. Sandholm, D. Pennock, D. Reeves, "A Practical Liquidity-Sensitive Automated Market Maker", EC 2010.
- Y. Chen et al., "Gaming Prediction Markets: Equilibrium Strategies with a Market Maker", *Algorithmica* 58 (2010).
- Y. Chen, J. Wortman Vaughan, "A New Understanding of Prediction Markets via No-Regret Learning", EC 2010.
- A. Beygelzimer, J. Langford, D. Pennock, "Learning Performance of Prediction Markets with Kelly Bettors", AAMAS 2012. [arXiv:1201.6655](https://arxiv.org/abs/1201.6655) — also the theoretical case for fractional Kelly.
- O. Saguillo, V. Ghafouri, L. Kiffer, G. Suarez-Tangil, "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets", AFT 2025. [arXiv:2508.03474](https://econpapers.repec.org/paper/arxpapers/2508.03474.htm)
- J. L. Kelly, "A New Interpretation of Information Rate", *Bell System Technical Journal* (1956).

Options-implied densities

- D. Breeden, R. Litzenberger, "Prices of State-Contingent Claims Implicit in Option Prices", *Journal of Business* (1978).
- R. Lee, "The Moment Formula for Implied Volatility at Extreme Strikes", *Mathematical Finance* (2004).
- J. Gatheral, A. Jacquier, "Arbitrage-free SVI volatility surfaces", *Quantitative Finance* (2014).
- C. Almeida, M. Grith, R. Miftachov, Z. Wang, "Risk Premia in the Bitcoin Market". [arXiv:2410.15195](https://arxiv.org/html/2410.15195v2)
- J. Winkel, W. Härdle, "Pricing Kernels and Risk Premia implied in Bitcoin Options", *Risks* 11 (2023). [mdpi.com](https://mdpi.com/2227-9091/11/5/85)
- "Hedging Cryptocurrency Options". [arXiv:2112.06807](https://arxiv.org/pdf/2112.06807)

Volatility

- F. Corsi, "A Simple Approximate Long-Memory Model of Realized Volatility", *Journal of Financial Econometrics* (2009).
- M. Garman, M. Klass, "On the Estimation of Security Price Volatilities from Historical Data", *Journal of Business* (1980).
- L. Glosten, R. Jagannathan, D. Runkle, "On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks", *Journal of Finance* (1993).
- C. Fernández, M. Steel, "On Bayesian Modeling of Fat Tails and Skewness", *JASA* (1998).
- L. Catania, M. Sandholdt, "Bitcoin at High Frequency", *JRFM* 12 (2019). [mdpi.com](https://mdpi.com/1911-8074/12/1/36)
- A. Das, W. Kong, R. Sen, Y. Zhou, "A decoder-only foundation model for time-series forecasting", ICML 2024. [Google Research](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)

Crypto microstructure and latency

- R. Cont, A. Kukanov, S. Stoikov, "The Price Impact of Order Book Events", *Journal of Financial Econometrics* 12 (2014). [arXiv:1011.6402](https://arxiv.org/abs/1011.6402)
- I. Makarov, A. Schoar, "Trading and Arbitrage in Cryptocurrency Markets", *Journal of Financial Economics* 135 (2020).
- S. He, A. Manela, O. Ross, V. von Wachter, "Fundamentals of Perpetual Futures". [arXiv:2212.06888](https://arxiv.org/pdf/2212.06888)
- Bieganowski, "Explainable Patterns in Cryptocurrency Microstructure" (2026). [arXiv:2602.00776](https://arxiv.org/html/2602.00776)
- N. Hautsch et al., "Building Trust Takes Time: Limits to Arbitrage for Blockchain-Based Assets". [arXiv:1812.00595](https://arxiv.org/abs/1812.00595)
- E. Budish, P. Cramton, J. Shim, "The High-Frequency Trading Arms Race: Frequent Batch Auctions as a Market Design Response", *QJE* 130 (2015).
- M. Aquilina, E. Budish, P. O'Neill, "Quantifying the High-Frequency Trading 'Arms Race'", *QJE* 137 (2022). [BIS WP 955](https://www.bis.org/publ/work955.pdf)
- M. Avellaneda, S. Stoikov, "High-frequency trading in a limit order book", *Quantitative Finance* (2008).

Open-source projects studied: [NautilusTrader](https://nautilustrader.io/docs/latest/concepts/overview/), [Hummingbot](https://hummingbot.org/strategies/avellaneda-market-making), [vn.py](https://github.com/vnpy/vnpy), [Qlib](https://github.com/microsoft/qlib), ccxt, vectorbt, freqtrade.
