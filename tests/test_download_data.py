from unittest.mock import patch
import tempfile
from pathlib import Path
from src.download_data import download_data
import os


def test_download_data():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = os.path.join(temp_dir, "test_folder")

        with patch("src.download_data.urllib.request.urlretrieve") as mock_retrieve:
            download_data(test_dir, "test.txt", "http://example.com/test.txt")
            assert Path(test_dir).exists()
            mock_retrieve.assert_called_once_with(
                "http://example.com/test.txt",
                Path(test_dir) / "test.txt"
            )

            
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