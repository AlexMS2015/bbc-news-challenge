from .config import config
from loguru import logger
import numpy as np
import pandas as pd
import joblib
from .steps.load_data import (
    download_data,
    clean_data,
    split_data,
)
from .steps.feature_eng import embed_text
from .steps.train import (
    train_logistic_regression,
    predict,
)
from .steps.evaluate import (
    evaluate_classifier,
    correct_vs_incorrect,
    plot_embeddings_with_errors,
    get_rank_and_probs,
    build_error_df,
)


def run_pipeline():
    """
    Main function to run the entire pipeline.
    """
    logger.info("Starting the data pipeline")

    # Download, load and clean data
    download_data(
        dir=config.folders["data"],
        filename=config.data["filename"],
        url=config.data["url"],
    )
    df = pd.read_csv(config.data_path / config.data["filename"])
    df = clean_data(df)
    X, y = df.text, df.category

    # Split data into train, and test sets
    X_train, y_train, X_test, y_test = split_data(
        X=X, y=y, test_size=config.data["test_size"], random_state=config.random_state
    )
    logger.debug(X_train.head())

    # Embed text using Sentence Transformers
    X_train_embed, X_test_embed = embed_text(
        X_train,
        X_test,
        model_name=config.feature_eng["sentence_transformers"]["model_name"],
        device=config.feature_eng["sentence_transformers"]["device"],
    )
    np.save(config.data_path / "X_train_embed.npy", X_train_embed)
    np.save(config.data_path / "X_test_embed.npy", X_test_embed)

    # Train Logistic Regression model
    model = train_logistic_regression(
        X_train_embed,
        y_train,
        random_state=config.random_state,
        **config.model["logistic_regression"],
    )
    joblib.dump(model, config.models_path / "logistic_model.pkl")

    # Make predictions
    y_train_pred, _ = predict(model, X_train_embed)
    y_test_pred, y_test_pred_prob = predict(model, X_test_embed)
    np.save(config.data_path / "y_train_pred.npy", y_train_pred)
    np.save(config.data_path / "y_test_pred.npy", y_test_pred)
    np.save(config.data_path / "y_test_pred_prob.npy", y_test_pred_prob)

    # Evaluate the classifier
    logger.info("Evaluating the classifier on training and test sets")
    report, cm = evaluate_classifier(y_train, y_train_pred, model.classes_)
    cm.figure_.savefig(
        config.eval_path / "confusion_matrix_train.png", dpi=300, bbox_inches="tight"
    )
    pd.DataFrame(report).round(4).to_csv(
        config.eval_path / "classification_report_train.csv"
    )
    report, cm = evaluate_classifier(y_test, y_test_pred, model.classes_)
    cm.figure_.savefig(
        config.eval_path / "confusion_matrix_test.png", dpi=300, bbox_inches="tight"
    )
    pd.DataFrame(report).round(4).to_csv(
        config.eval_path / "classification_report_test.csv"
    )

    # Error analysis
    logger.info("Performing error analysis")
    correct_vs_incorrect_fig = correct_vs_incorrect(
        y_test_pred_prob, y_test_pred, y_test
    )
    correct_vs_incorrect_fig.savefig(
        config.eval_path / "correct_vs_incorrect.png", dpi=300, bbox_inches="tight"
    )

    umap_fig = plot_embeddings_with_errors(X_test_embed, y_test, y_test_pred)
    umap_fig.savefig(
        config.eval_path / "embeddings_with_errors.png", dpi=300, bbox_inches="tight"
    )
    errors = y_test != y_test_pred
    classes = model.classes_
    rank_true, prob_true, prob_pred = get_rank_and_probs(
        y_test_pred_prob[errors], y_test[errors], y_test_pred[errors], classes
    )
    error_df = build_error_df(
        X_test[errors],
        y_test[errors],
        y_test_pred[errors],
        rank_true,
        prob_true,
        prob_pred,
    )
    error_df.to_csv(config.eval_path / "error_analysis_test.csv")

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    run_pipeline()
