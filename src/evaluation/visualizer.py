import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_confusion_matrix(y_true, y_pred):
    os.makedirs("outputs/graphs", exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.savefig("outputs/graphs/confusion_matrix.png")
    plt.close()


def plot_roc_curve(y_true, scores):
    os.makedirs("outputs/graphs", exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("outputs/graphs/roc_curve.png")
    plt.close()

    return roc_auc


def plot_anomaly_scores(
    scores,
    threshold=None
):

    plt.figure(
        figsize=(12,5)
    )

    plt.plot(
        scores,
        label="Anomaly Score"
    )

    if threshold is not None:

        plt.axhline(
            y=threshold,
            linestyle="--",
            label="Threshold"
        )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Frame-level Anomaly Score"
    )

    plt.legend()

    plt.savefig(
        "outputs/graphs/anomaly_scores.png"
    )

    plt.close()