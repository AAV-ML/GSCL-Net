"""Publication-oriented prediction plots."""
import matplotlib.pyplot as plt

from training import safe_pearson


def plot_ensemble_predictions(targets, gnn_predictions, lstm_predictions, ensemble_predictions, output_path):
    values = [targets, gnn_predictions, lstm_predictions, ensemble_predictions]
    length = min(map(len, values))
    if length < 2:
        return
    targets, gnn_predictions, lstm_predictions, ensemble_predictions = [value[:length] for value in values]
    panels = [
        ('GNN (AIOM+ERM)', gnn_predictions),
        ('BiLSTM (ERM)', lstm_predictions),
        ('Ensemble average', ensemble_predictions),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, (name, predictions) in zip(axes, panels):
        axis.scatter(targets, predictions, alpha=0.1, s=10)
        limits = [min(targets.min(), predictions.min()), max(targets.max(), predictions.max())]
        axis.plot(limits, limits, 'r--')
        axis.set_title(f'{name}\nr={safe_pearson(targets, predictions):.4f}')
        axis.set_xlabel('True value')
        axis.set_ylabel('Predicted value')
        axis.grid(True)
        axis.axis('equal')
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
