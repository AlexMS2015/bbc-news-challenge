import os
import urllib.request
from pathlib import Path
from loguru import logger
from .config import config


@logger.catch
def download_data(dir:str, filename:str, url:str) -> None:
    '''
    Downloads data from a given URL to a specified directory with a specified filename.
    '''
    logger.info(f"Downloading data from {url} to {dir}/{filename}")
    path = Path(dir)
    path.mkdir(exist_ok=True)
    path = path / filename
    try:
        urllib.request.urlretrieve(url, path)
        logger.info(f"Data downloaded successfully to {path}")
    except Exception as e:
        logger.critical(f"Failed to download data from {url} to {path}")
        raise


if __name__ == "__main__":
    download_data(
        dir=config.data['dir'],
        filename=config.data['filename'],
        url=config.data['url']
    )