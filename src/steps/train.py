from loguru import logger
from sklearn.linear_model import LogisticRegression
from typing import Any
import numpy as np


def train_logistic_regression(
    X: Any,
    y: Any,
    random_state: int,
    **logistic_params
) -> LogisticRegression:
    """
    Train a Logistic Regression model on the provided data.

    Args:
        X: Training features.
        y: Training labels.
        random_state: Random seed for reproducibility.
        **logistic_params: Additional parameters for LogisticRegression.

    Returns:
        Trained Logistic Regression model.
    """
    logger.info("Training Logistic Regression model")

    model = LogisticRegression(random_state=random_state, **logistic_params)
    model.fit(X, y)

    return model


def predict(
    model: LogisticRegression,
    X: Any
) -> tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with a trained Logistic Regression model.

    Args:
        model: Trained Logistic Regression model.
        X: Features to predict on.

    Returns:
        Tuple of (predicted labels, predicted probabilities).
    """
    logger.info("Making predictions with Logistic Regression model")

    pred = model.predict(X)
    pred_prob = model.predict_proba(X)

    return pred, pred_prob
