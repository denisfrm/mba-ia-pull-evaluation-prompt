"""Testes das rubricas usadas pelo LLM-as-Judge."""

import pytest

from src.metrics import rating_to_score


@pytest.mark.parametrize(
    ('rating', 'expected'),
    [
        ('excellent', 1.0),
        ('ACCEPTABLE', 0.9),
        (' insufficient ', 0.7),
        ('incorrect', 0.0),
        ('valor-inválido', 0.0),
    ],
)
def test_rating_to_score(rating: str, expected: float) -> None:
    """A conversão ordinal deve ser determinística."""
    assert rating_to_score(rating) == expected
