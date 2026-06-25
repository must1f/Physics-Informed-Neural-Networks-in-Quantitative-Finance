import pandas as pd
import pytest
from unittest.mock import patch
from src.data.fetcher import DataFetcher


@pytest.fixture
def fetcher(tmp_path):
    return DataFetcher(cache_dir=tmp_path)


def _mock_vix_download(*args, **kwargs):
    """Minimal yfinance response for ^VIX — no Volume column."""
    idx = pd.date_range("2020-01-02", periods=300, freq="B")
    return pd.DataFrame({"Close": 15.0 + idx.day_of_year * 0.01}, index=idx)


def test_fetch_close_returns_series(fetcher):
    with patch("src.data.fetcher.yf.download", side_effect=_mock_vix_download):
        result = fetcher.fetch_close("^VIX", "2020-01-01", "2021-06-30")
    assert isinstance(result, pd.Series)
    assert result.name == "Close"
    assert len(result) == 300


def test_fetch_close_uses_cache(fetcher):
    with patch("src.data.fetcher.yf.download", side_effect=_mock_vix_download) as mock_dl:
        fetcher.fetch_close("^VIX", "2020-01-01", "2021-06-30")
        fetcher.fetch_close("^VIX", "2020-01-01", "2021-06-30")
    mock_dl.assert_called_once()  # second call hits cache


def test_fetch_close_raises_on_empty(fetcher):
    with patch("src.data.fetcher.yf.download", return_value=pd.DataFrame()):
        with pytest.raises(ValueError, match="No data"):
            fetcher.fetch_close("^VIX", "2020-01-01", "2021-06-30")
