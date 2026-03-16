"""Shared fixtures for credence-engine tests."""

import pytest
from credence.julia_bridge import CredenceBridge


@pytest.fixture(scope="session")
def bridge():
    """Session-scoped CredenceBridge (Julia loads once for all tests)."""
    return CredenceBridge()
