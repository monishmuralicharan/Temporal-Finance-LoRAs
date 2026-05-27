import pytest

from temporal_finance.evaluation import compute_simple_return


def test_compute_simple_return_uses_end_over_start():
    assert compute_simple_return(100.0, 105.0) == pytest.approx(0.05)


def test_compute_simple_return_rejects_zero_start():
    with pytest.raises(ValueError):
        compute_simple_return(0.0, 105.0)

