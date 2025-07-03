from .config import config
from loguru import logger
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from .steps.load_data import (
    download_data,
    clean_data,
    split_data,
)
from .steps.feature_eng import apply_tfidf
from .steps.train import (
    train_logistic_regression,
    predict,
)
from .steps.evaluate import (
    evaluate_classifier,
    get_top_bottom_features_plot,
    get_local_feature_contributions,
    get_rank_and_probs,
    build_error_df
)


def run_pipeline():
    """
    Main function to run the entire pipeline.
    """
    logger.info("Starting the data pipeline")

    # Download, load and clean data
    download_data(
        dir=config.folders['data'],
        filename=config.data['filename'],
        url=config.data['url']
    )
    df = pd.read_csv(config.data_path / config.data['filename'])
    df = clean_data(df)
    X, y = df.text, df.category

    # Split data into train, validation, and test sets
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X=X,
        y=y,
        splits=config.data['train_val_test_split'],
        random_state=config.random_state
    )
    logger.debug(X_train.head())

    # Apply TF-IDF vectorization
    vectorizer, feature_names, X_train_tfidf, X_val_tfidf, X_test_tfidf = apply_tfidf(
        X_train,
        X_val,
        X_test,
        **config.feature_eng['tfidf']
    )
    joblib.dump(vectorizer, config.models_path / 'vectorizer.pkl')
    np.save(config.data_path / 'X_train_tfidf.npy', X_train_tfidf.toarray())
    np.save(config.data_path / 'X_val_tfidf.npy', X_val_tfidf.toarray())
    np.save(config.data_path / 'X_test_tfidf.npy', X_test_tfidf.toarray())

    # Train Logistic Regression model
    model = train_logistic_regression(
        X_train_tfidf,
        y_train,
        random_state=config.random_state,
        **config.model['logistic_regression']
    )
    joblib.dump(model, config.models_path / 'logistic_model.pkl')

    # Make predictions
    y_train_pred, _ = predict(model, X_train_tfidf)
    y_val_pred, y_val_pred_prob = predict(model, X_val_tfidf)
    np.save(config.data_path / 'y_train_pred.npy', y_train_pred)
    np.save(config.data_path / 'y_val_pred.npy', y_val_pred)
    np.save(config.data_path / 'y_val_pred_prob.npy', y_val_pred_prob)
    
    # Evaluate the classifier
    logger.info("Evaluating the classifier on training and validation sets")
    report, cm = evaluate_classifier(y_train, y_train_pred, model.classes_)
    cm.figure_.savefig(
        config.eval_path / 'confusion_matrix_train.png',
        dpi=300,
        bbox_inches='tight'
    )
    pd.DataFrame(report).round(4).to_csv(config.eval_path / 'classification_report_train.csv')
    report, cm = evaluate_classifier(y_val, y_val_pred, model.classes_)
    cm.figure_.savefig(
        config.eval_path / 'confusion_matrix_val.png',
        dpi=300,
        bbox_inches='tight'
    )
    pd.DataFrame(report).round(4).to_csv(config.eval_path / 'classification_report_val.csv')

    
    features_plot = get_top_bottom_features_plot(
        model.classes_,
        model.coef_,
        feature_names
    )
    features_plot.savefig(config.eval_path / 'top_features.png', bbox_inches='tight')

    # Get local feature contributions
    logger.info("Performing error analysis")
    errors = y_val != y_val_pred
    classes = model.classes_
    coef = model.coef_
    top_features, top_contribs = get_local_feature_contributions(
        y_val_pred[errors],
        classes,
        X_val_tfidf.toarray()[errors],
        coef,
        feature_names,
        top_n=5,
    )
    rank_true, prob_true, prob_pred = get_rank_and_probs(
        y_val_pred_prob[errors],
        y_val[errors],
        y_val_pred[errors],
        classes
    )
    error_df = build_error_df(
        X_val[errors],
        y_val[errors],
        y_val_pred[errors],
        rank_true,
        prob_true,
        prob_pred,
        top_features,
        top_contribs
    )
    error_df.to_csv(config.eval_path / 'error_analysis_val.csv')

    # TEST SET PERFORMANCE
    # RETRAIN THE MODEL ON ALL THE DATA

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    run_pipeline()