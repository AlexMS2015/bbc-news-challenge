import urllib.request
from pathlib import Path
from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split


@logger.catch
def download_data(dir: str, filename: str, url: str) -> None:
    """
    Download a file from a URL into the given directory.
    """
    logger.info(f"Downloading data from {url} to {dir}/{filename}")
    path = Path(dir)
    path.mkdir(exist_ok=True)
    path = path / filename
    try:
        urllib.request.urlretrieve(url, path)
        logger.info(f"Data downloaded successfully to {path}")
    except Exception:
        logger.critical(f"Failed to download data from {url} to {path}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with NaNs in 'text' and remove duplicates.
    """
    logger.info("Cleaning data by dropping rows with NaN values in 'text' column")
    initial_shape = df.shape
    df = df.dropna(subset=["text"]).drop_duplicates()
    final_shape = df.shape
    logger.info(f"Data cleaned: {initial_shape} -> {final_shape}")
    return df


def reset_pair(X: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Reset index for X and y.
    """
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return X, y


def split_data(
    X: pd.Series, 
    y: pd.Series, 
    random_state: int, 
    test_size: float
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Stratified train/test split.
    """
    logger.info("Splitting data into train, validation, and test sets")
    assert test_size < 1, "Test size must be less than 1"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    X_train, y_train = reset_pair(X_train, y_train)
    X_test, y_test = reset_pair(X_test, y_test)

    logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test")

    return X_train, y_train, X_test, y_test
