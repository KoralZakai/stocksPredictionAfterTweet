import pytest
from pydantic import ValidationError

from config.settings import SETTINGS, Settings


def test_preregistered_values() -> None:
    assert len(SETTINGS.etfs) == 10
    assert SETTINGS.entry_anchor == "open"
    assert SETTINGS.horizon_days == (1, 2, 3)
    assert SETTINGS.k == 0.5
    assert SETTINGS.embargo_sessions >= 3  # §3.6


def test_frozen() -> None:
    with pytest.raises(ValidationError):
        SETTINGS.k = 0.9


def test_invariant_validators() -> None:
    with pytest.raises(ValidationError):
        Settings(embargo_sessions=1)
    with pytest.raises(ValidationError):
        Settings(k=0.0)
