from loguru import logger
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X, y, random_state, **logistic_params):
    """
    Trains a Logistic Regression model on the provided training data.
    
    Parameters:
    - X_train_tfidf: The TF-IDF transformed training features.
    - y_train: The labels for the training data.
    
    Returns:
    - model: The trained Logistic Regression model.
    """
    logger.info("Training Logistic Regression model")
    
    # Initialize and fit the model

    model = LogisticRegression(random_state=random_state, **logistic_params)
    model.fit(X, y)

    return model


def predict(model, X):
    """
    Makes predictions using the trained Logistic Regression model.
    
    Parameters:
    - model: The trained Logistic Regression model.
    - X: The features to predict on.
    
    Returns:
    - predictions: The predicted labels.
    """
    logger.info("Making predictions with Logistic Regression model")
    
    pred = model.predict(X)
    pred_prob = model.predict_proba(X)
    
    return pred, pred_prob