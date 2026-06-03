from .metrics import calculate_metrics
from .visualizer import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_anomaly_scores
)

def evaluate_model(
    y_true,
    y_pred,
    scores
):

    results = calculate_metrics(
        y_true,
        y_pred
    )

    print(results)

    plot_confusion_matrix(
        y_true,
        y_pred
    )

    plot_roc_curve(
        y_true,
        scores
    )

    plot_anomaly_scores(
        scores
    )

    return results