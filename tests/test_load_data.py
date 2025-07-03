from unittest.mock import patch
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from src.config import config
from src.steps.load_data import download_data, split_data
import os


def test_download_data():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = os.path.join(temp_dir, "test_folder")

        with patch("src.steps.load_data.urllib.request.urlretrieve") as mock_retrieve:
            download_data(test_dir, "test.txt", "http://example.com/test.txt")
            assert Path(test_dir).exists()
            mock_retrieve.assert_called_once_with(
                "http://example.com/test.txt",
                Path(test_dir) / "test.txt"
            )

def test_split_data():
    X = pd.Series(np.arange(100))
    y = pd.Series(np.random.choice(['A', 'B', 'C'], size=100))
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X=X,
        y=y,
        splits=(0.6, 0.2, 0.2),
        random_state=1,
    )

    assert isinstance(X_train, pd.Series)
    assert isinstance(X_val, pd.Series)
    assert isinstance(X_test, pd.Series)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_val, pd.Series)
    assert isinstance(y_test, pd.Series)

    assert len(X_train) == 60
    assert len(X_val) == 20
    assert len(X_test) == 20


# in integration/test_download_data_integration.py
# integration test
# def test_download_data_integration():
#     with tempfile.TemporaryDirectory() as temp_dir:
#         # Use a reliable small file for testing
#         url = "https://httpbin.org/robots.txt"  # or any small, stable test file
        
#         download_data(temp_dir, "robots.txt", url)
        
#         downloaded_file = Path(temp_dir) / "robots.txt"
#         assert downloaded_file.exists()
#         assert downloaded_file.stat().st_size > 0