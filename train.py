"""GSCL-Net training entry point."""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Subset, TensorDataset
from torch_geometric.loader import DataLoader as GNNDataLoader

from config import (
    BATCH_SIZE, DATA_SPLIT_SEED, DEVICE, EPOCHS, GNN_LR, GNN_WEIGHT_DECAY,
    LSTM_LR, LSTM_WEIGHT_DECAY, MAX_SEQ_LENGTH, N_TRAIN, N_VAL, set_seed,
)
from data_utils import ProteinDataBuilder
from models import BiLSTM_Refiner, GNN_Refiner
from plotting import plot_ensemble_predictions
from training import get_predictions, safe_pearson, train_base_model

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = ['sans-serif']
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['text.usetex'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'Production.csv')
    set_seed(DATA_SPLIT_SEED)

    dataframe = pd.read_csv(data_path)
    dataframe = dataframe[dataframe['AA'].str.len() == MAX_SEQ_LENGTH].copy().reset_index(drop=True)
    if len(dataframe) < N_TRAIN + N_VAL:
        raise ValueError(f'Not enough samples: need at least {N_TRAIN + N_VAL}, found {len(dataframe)}.')
    print(f'Loaded {len(dataframe)} valid samples on {DEVICE}.', flush=True)

    builder = ProteinDataBuilder(os.path.join(base_dir, 'embeddings', 'esm2_features_aligned.pt'))
    all_data = builder.build_all_data(dataframe, target_column='Production')
    all_graphs = all_data['gnn_graphs']
    all_lstm_features = all_data['lstm_features']
    all_targets = all_data['targets']

    indices = np.arange(len(all_graphs))
    test_size = len(indices) - N_TRAIN - N_VAL
    train_val_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=DATA_SPLIT_SEED, shuffle=True)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=N_VAL, random_state=DATA_SPLIT_SEED)
    print(f'Split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}', flush=True)

    train_gnn_loader = GNNDataLoader([all_graphs[index] for index in train_indices], batch_size=BATCH_SIZE, shuffle=True)
    val_gnn_loader = GNNDataLoader([all_graphs[index] for index in val_indices], batch_size=BATCH_SIZE)
    test_gnn_loader = GNNDataLoader([all_graphs[index] for index in test_indices], batch_size=BATCH_SIZE)
    full_lstm_dataset = TensorDataset(all_lstm_features, all_targets)
    train_lstm_loader = TorchDataLoader(Subset(full_lstm_dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True)
    val_lstm_loader = TorchDataLoader(Subset(full_lstm_dataset, val_indices), batch_size=BATCH_SIZE)
    test_lstm_loader = TorchDataLoader(Subset(full_lstm_dataset, test_indices), batch_size=BATCH_SIZE)

    prediction_dir = os.path.join(base_dir, '_preds')
    os.makedirs(prediction_dir, exist_ok=True)
    gnn = train_base_model(GNN_Refiner, train_gnn_loader, val_gnn_loader, GNN_LR, GNN_WEIGHT_DECAY, os.path.join(base_dir, 'retrained_gnn_refiner_best.pth'), is_gnn=True, epochs=EPOCHS)
    lstm = train_base_model(BiLSTM_Refiner, train_lstm_loader, val_lstm_loader, LSTM_LR, LSTM_WEIGHT_DECAY, os.path.join(base_dir, 'retrained_lstm_refiner_best.pth'), is_gnn=False, epochs=EPOCHS)

    gnn_predictions, test_targets = get_predictions(gnn, test_gnn_loader, is_gnn=True)
    lstm_predictions, _ = get_predictions(lstm, test_lstm_loader, is_gnn=False)
    ensemble_predictions = (gnn_predictions.astype(np.float64) + lstm_predictions.astype(np.float64)) / 2.0
    metrics = np.array([
        safe_pearson(test_targets, gnn_predictions),
        safe_pearson(test_targets, lstm_predictions),
        safe_pearson(test_targets, ensemble_predictions),
        np.mean((test_targets.astype(np.float64) - ensemble_predictions) ** 2),
    ])
    print(f'GNN r={metrics[0]:.4f}; BiLSTM r={metrics[1]:.4f}; ensemble r={metrics[2]:.4f}, MSE={metrics[3]:.4f}, R2={r2_score(test_targets, ensemble_predictions):.4f}', flush=True)
    np.save(os.path.join(prediction_dir, 'refiner_r.npy'), metrics)
    plot_ensemble_predictions(test_targets, gnn_predictions, lstm_predictions, ensemble_predictions, os.path.join(base_dir, 'ensemble_scatter_AIOM_ERM_Average.png'))


if __name__ == '__main__':
    main()
