"""Training, prediction, and metric helpers."""
import copy

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config import AIOM_PONDER_LAMBDA, BASE_LEARNER_SEED, DEVICE, PATIENCE, EPOCHS, set_seed


@torch.no_grad()
def get_predictions(model, loader, is_gnn):
    model.eval()
    predictions, targets = [], []
    for batch in tqdm(loader, desc='Predicting', leave=False):
        if is_gnn:
            batch = batch.to(DEVICE)
            batch_targets = batch.y.view(-1, 1)
            batch_predictions, _, _ = model(batch)
        else:
            features, batch_targets = batch
            batch_predictions = model(features.to(DEVICE))
            batch_targets = batch_targets.to(DEVICE)
        predictions.append(batch_predictions.view_as(batch_targets).cpu().numpy())
        targets.append(batch_targets.cpu().numpy())
    if not predictions:
        return np.array([]), np.array([])
    return np.concatenate(predictions).squeeze(), np.concatenate(targets).squeeze()


def safe_pearson(x, y):
    x, y = np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()
    if len(x) < 2 or len(x) != len(y):
        return 0.0
    x_centered, y_centered = x - x.mean(), y - y.mean()
    denominator = np.sqrt(float((x_centered ** 2).sum()) * float((y_centered ** 2).sum()))
    return 0.0 if denominator == 0 else float((x_centered * y_centered).sum() / denominator)


def train_base_model(model_class, train_loader, val_loader, lr, weight_decay, model_path,
                     patience=PATIENCE, epochs=EPOCHS, is_gnn=True):
    set_seed(BASE_LEARNER_SEED)
    model = model_class().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = torch.nn.MSELoss()
    best_loss, wait, best_state = float('inf'), 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f'{model_class.__name__} Epoch {epoch}/{epochs}', leave=False):
            optimizer.zero_grad()
            if is_gnn:
                batch = batch.to(DEVICE)
                batch_targets = batch.y.view(-1, 1)
                predictions, ponder_cost, _ = model(batch)
                loss = criterion(predictions, batch_targets) + AIOM_PONDER_LAMBDA * ponder_cost
                sample_count = batch.num_graphs
            else:
                features, batch_targets = batch
                features, batch_targets = features.to(DEVICE), batch_targets.to(DEVICE)
                predictions = model(features)
                loss = criterion(predictions, batch_targets)
                sample_count = features.size(0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * sample_count
        val_predictions, val_targets = get_predictions(model, val_loader, is_gnn)
        val_loss = mean_squared_error(val_targets, val_predictions)
        scheduler.step(val_loss)
        print(f'{model_class.__name__} epoch {epoch:03d}: train_loss={total_loss / len(train_loader.dataset):.4f}, val_loss={val_loss:.4f}, val_r2={r2_score(val_targets, val_predictions):.4f}', flush=True)
        if val_loss < best_loss:
            best_loss, wait, best_state = val_loss, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait > patience:
                break
    if best_state is None:
        raise RuntimeError('No valid checkpoint was produced.')
    torch.save(best_state, model_path)
    model.load_state_dict(best_state)
    return model
