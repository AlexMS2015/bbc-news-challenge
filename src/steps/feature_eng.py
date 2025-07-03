from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer


def apply_tfidf(text_train, text_val, text_test, **tfidf_params) -> tuple:
    """
    Applies TF-IDF vectorization to the training, validation, and test sets.
    """
    logger.info("Applying TF-IDF vectorization")

    logger.debug(f"TF-IDF parameters: {tfidf_params}")
    vectorizer = TfidfVectorizer(**tfidf_params)
    tfidf_train = vectorizer.fit_transform(text_train)
    tfidf_val = vectorizer.transform(text_val)
    tfidf_test = vectorizer.transform(text_test)
    feature_names = vectorizer.get_feature_names_out()
    
    logger.info("TF-IDF vectorization applied successfully")
    
    return vectorizer, feature_names, tfidf_train, tfidf_val, tfidf_test