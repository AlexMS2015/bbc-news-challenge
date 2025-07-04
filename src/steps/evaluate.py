import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import umap
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder


def evaluate_classifier(y: pd.Series, y_pred: pd.Series, classes: list) -> tuple:
    """
    Generate classification report and confusion matrix.
    """
    report = classification_report(y, y_pred, labels=classes, output_dict=True)
    cmd = ConfusionMatrixDisplay.from_predictions(
        y, y_pred, labels=classes, normalize="true", xticks_rotation="vertical"
    )
    return report, cmd


def correct_vs_incorrect(
    pred_prob: pd.Series, pred: pd.Series, actual: np.ndarray
) -> Figure:
    """
    Plot confidence distribution for correct vs incorrect predictions.
    """
    errors = pred != actual
    max_probs = np.max(pred_prob, axis=1)

    fig, ax = plt.subplots()
    sns.kdeplot(max_probs[~errors], label="Correct", fill=True, ax=ax)
    sns.kdeplot(max_probs[errors], label="Incorrect", fill=True, ax=ax)

    ax.set_xlabel("Max predicted probability (confidence)")
    ax.set_title("Model confidence: Correct vs. Incorrect predictions")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return fig


def plot_embeddings_with_errors(embeddings, true_labels, pred_labels):
    """
    Plots 2D UMAP embeddings with points colored by true class and highlights
    misclassified points.
    """
    le = LabelEncoder()
    y_true_enc = le.fit_transform(true_labels)
    y_pred_enc = le.transform(pred_labels)

    misclassified = y_true_enc != y_pred_enc
    reducer = umap.UMAP(random_state=42)
    reduced = reducer.fit_transform(embeddings)

    colors = plt.cm.Set1(np.linspace(0, 1, len(le.classes_)))
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, label in enumerate(le.classes_):
        is_class = y_true_enc == i
        ax.scatter(
            *reduced[is_class & ~misclassified].T,
            c=[colors[i]],
            s=30,
            alpha=0.6,
            label=f"{label}",
        )
        ax.scatter(
            *reduced[is_class & misclassified].T,
            c=[colors[i]],
            s=80,
            alpha=1.0,
            edgecolors="black",
            linewidth=1.5,
            label=f"{label} incorrect",
        )

    ax.set_title("UMAP: Validation Embeddings with Misclassifications")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return fig


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
        .assign(rank_true=rank_true, prob_true=prob_true, prob_pred=prob_pred)
        .reset_index(drop=True)
    )

    error_df = error_df.sort_values(["true", "predicted"]).round(3)
    return error_df
