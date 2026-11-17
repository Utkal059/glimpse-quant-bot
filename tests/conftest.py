"""Shared fixtures. No test touches the network or a real ``.env`` file."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from glimpse_quant_bot.config import Settings
from glimpse_quant_bot.ledger import Ledger
from glimpse_quant_bot.lmsr import LmsrBook
from glimpse_quant_bot.types import AssetType, Bucket, TopicMeta

NOW = 1_800_000_000.0


@pytest.fixture
def make_settings() -> Callable[..., Settings]:
    def factory(**overrides: object) -> Settings:
        return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]

    return factory


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


@pytest.fixture
def ledger() -> Iterator[Ledger]:
    book = Ledger(":memory:", clock=lambda: NOW)
    yield book
    book.close()


@pytest.fixture
def topic() -> TopicMeta:
    """A 500-bucket BTC market expiring in two hours, $200 buckets from $25,000."""
    buckets = tuple(
        Bucket(
            i + 1, f"{25_000 + 200 * i}-{25_200 + 200 * i}", 25_000.0 + 200 * i, 25_200.0 + 200 * i
        )
        for i in range(500)
    )
    return TopicMeta(5398, "batch", "Hourly Bitcoin", AssetType.BTC, int(NOW) + 7200, buckets)


@pytest.fixture
def book(topic: TopicMeta) -> LmsrBook:
    """A live-like book: 80 subsidised shares everywhere, trading around bucket 200."""
    shares = [80.0] * 500
    for offset, extra in ((-2, 520.0), (-1, 560.0), (0, 575.0), (1, 568.0), (2, 540.0)):
        shares[199 + offset] += extra
    return LmsrBook.from_shares(topic.topic_id, [b.option_id for b in topic.buckets], shares)
