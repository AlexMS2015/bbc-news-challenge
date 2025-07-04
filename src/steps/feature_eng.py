from loguru import logger
from sentence_transformers import SentenceTransformer


def embed_text(text_train, text_test, model_name: str, device: str) -> tuple:
    """
    Applies sentence transformer embeddings to the training and test sets.
    """
    logger.info("Applying sentence transformer embeddings")

    encoder = SentenceTransformer(model_name, device=device)
    embeddings_train = encoder.encode(text_train)
    embeddings_test = encoder.encode(text_test)

    logger.info("Sentence transformer embeddings applied successfully")

    return encoder, embeddings_train, embeddings_test
