import pytest
from spectrum.leakage import clear_fingerprints


@pytest.fixture(autouse=True)
def _fresh_future_fingerprints():
    """Each test sees only the future it produced itself."""
    clear_fingerprints()
    yield
    clear_fingerprints()
