import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


def evaluate_classifier(y: pd.Series, y_pred: pd.Series, classes: list) -> tuple:
    """
    Generate classification report and confusion matrix.
    """
    report = classification_report(y, y_pred, labels=classes, output_dict=True)
    cmd = ConfusionMatrixDisplay.from_predictions(
        y, y_pred, labels=classes, normalize="true", xticks_rotation="vertical"
    )
    return report, cmd


def correct_vs_incorrect(pred_prob: pd.Series, pred: pd.Series, actual: np.ndarray) -> Figure:
    """
    Plot confidence distribution for correct vs incorrect predictions.
    """
    errors = pred != actual
    max_probs = np.max(pred_prob, axis=1)

    fig, ax = plt.subplots()
    sns.kdeplot(max_probs[~errors], label='Correct', fill=True, ax=ax)
    sns.kdeplot(max_probs[errors], label='Incorrect', fill=True, ax=ax)

    ax.set_xlabel("Max predicted probability (confidence)")
    ax.set_title("Model confidence: Correct vs. Incorrect predictions")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return fig


# def get_class_indices(y, classes):
#     class_to_index = {label: i for i, label in enumerate(classes)}
#     return [class_to_index[label] for label in y]


# def get_local_feature_contributions(
#     y_pred, classes, X_tfidf, coef, feature_names, top_n
# ):
#     pred_class_idx = get_class_indices(y_pred, classes)

#     tfidf = X_tfidf
#     weights = coef[pred_class_idx]
#     contribs = tfidf * weights

#     idx_topn = np.argsort(np.abs(contribs), axis=1)[:, -top_n:]
#     top_features = feature_names[idx_topn]
#     top_contribs = np.take_along_axis(contribs, idx_topn, axis=1)

#     return top_features, top_contribs


def get_rank_and_probs(
    y_pred_prob: pd.Series, y_true: pd.Series, y_pred: pd.Series, classes: list
) -> tuple[list[int], list[float], list[float]]:
    """
    Get rank and predicted probabilities for true and predicted classes.
    """
    prob_df = pd.DataFrame(y_pred_prob, columns=classes)
    rank_df = prob_df.rank(ascending=False, axis=1).astype("int")
    rank_true = [rank_df.loc[i, true] for i, true in enumerate(y_true)]
    prob_true = [prob_df.loc[i, true] for i, true in enumerate(y_true)]
    prob_pred = [prob_df.loc[i, pred] for i, pred in enumerate(y_pred)]
    return rank_true, prob_true, prob_pred


def build_error_df(
    X_val: pd.Series,
    y_val: pd.Series,
    y_val_pred: pd.Series,
    rank_true: list,
    prob_true: list,
    prob_pred: list,
) -> pd.DataFrame:
    """
    Create a DataFrame summarizing prediction errors and confidence.
    """
    error_df = (
        pd.DataFrame({"text": X_val, "true": y_val, "predicted": y_val_pred})
        .assign(
            rank_true=rank_true,
            prob_true=prob_true,
            prob_pred=prob_pred,
            # top_features=top_features.tolist(),
            # top_features_contrib=top_features_contrib.round(2).tolist(),
        )
        .reset_index(drop=True)
    )

    error_df = error_df.sort_values(["true", "predicted"]).round(3)
    return error_df
