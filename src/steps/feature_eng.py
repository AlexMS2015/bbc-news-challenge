from loguru import logger
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd


def embed_text(
    text_train: pd.Series, text_test: pd.Series, model_name: str, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies sentence transformer embeddings to the training and test sets.

    Args:
        text_train: Pandas Series of training text samples.
        text_test: Pandas Series of test text samples.
        model_name: Name of the sentence transformer model.
        device: Device to load the model on ('cpu' or 'cuda').

    Returns:
        Tuple containing NumPy arrays of training and test embeddings.
    """
    logger.info("Applying sentence transformer embeddings")

    encoder = SentenceTransformer(model_name, device=device)
    embeddings_train = encoder.encode(text_train)
    embeddings_test = encoder.encode(text_test)

    logger.info("Sentence transformer embeddings applied successfully")

    return embeddings_train, embeddings_test
