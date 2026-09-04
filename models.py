"""GSCL-Net model definitions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

from config import (
    AIOM_GATE_TEMPERATURE, AIOM_NUM_LAYERS, ESM_DIM, GLOBAL_MANUAL_DIM,
    GLOBAL_PROTPARAM_DIM, ONEHOT_DIM, PCP_DIM, POS_ENCODING_DIM, REFINE_DIM,
    USE_AIOM,
)


class EmbeddingRefiner(nn.Module):
    def __init__(self, in_dim=ESM_DIM, out_dim=REFINE_DIM, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.gate = nn.Linear(in_dim, out_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, embeddings):
        return self.drop(self.proj(embeddings) * torch.sigmoid(self.gate(embeddings)))


class GNN_Refiner(nn.Module):
    def __init__(self, refine_dim=REFINE_DIM, hidden_dim=512, num_layers=AIOM_NUM_LAYERS):
        super().__init__()
        self.num_layers = num_layers
        conv_output_dim = hidden_dim // 4
        node_feature_dim = refine_dim + ONEHOT_DIM + POS_ENCODING_DIM + PCP_DIM
        self.refiner = EmbeddingRefiner(ESM_DIM, refine_dim)
        self.convs = nn.ModuleList([
            GATv2Conv(node_feature_dim, conv_output_dim, heads=4),
            GATv2Conv(hidden_dim, conv_output_dim, heads=4),
            GATv2Conv(hidden_dim, conv_output_dim),
        ])
        self.dropouts = [0.4, 0.3, 0.0]
        self.readout_projs = nn.ModuleList([
            nn.Linear(hidden_dim, conv_output_dim),
            nn.Linear(hidden_dim, conv_output_dim),
            nn.Identity(),
        ])
        self.stop_gates = nn.ModuleList([
            nn.Sequential(nn.Linear(conv_output_dim, 64), nn.ReLU(), nn.Linear(64, 1))
            for _ in range(num_layers - 1)
        ])
        final_dim = conv_output_dim + refine_dim + ONEHOT_DIM + PCP_DIM + GLOBAL_MANUAL_DIM + GLOBAL_PROTPARAM_DIM
        self.mlp = nn.Sequential(
            nn.Linear(final_dim, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 1)
        )

    def _soft_halting(self, stop_logits):
        remaining = torch.ones(stop_logits[0].size(0), device=stop_logits[0].device)
        weights, ponder_cost = [], torch.zeros_like(remaining)
        for stop_logit in stop_logits:
            probability = torch.sigmoid(stop_logit / AIOM_GATE_TEMPERATURE)
            weights.append(remaining * probability)
            remaining = remaining * (1.0 - probability)
            ponder_cost = ponder_cost + remaining
        weights.append(remaining)
        return torch.stack(weights, dim=1), ponder_cost.mean()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        master_indices = data.ptr[1:] - 1
        aa_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
        aa_mask[master_indices] = False
        raw_aa = x[aa_mask]
        aa_batch = batch[aa_mask]
        refined = self.refiner(x[:, :ESM_DIM])
        x = torch.cat([refined, x[:, ESM_DIM:]], dim=1)
        embed_pool = global_mean_pool(torch.cat([refined[aa_mask], raw_aa[:, ESM_DIM:ESM_DIM + ONEHOT_DIM]], dim=1), aa_batch)
        pcp_pool = global_mean_pool(raw_aa[:, ESM_DIM + ONEHOT_DIM + POS_ENCODING_DIM:], aa_batch)

        master_representations, stop_logits = [], []
        hidden = x
        for layer_index, convolution in enumerate(self.convs):
            hidden = F.relu(convolution(hidden, edge_index))
            if self.dropouts[layer_index] > 0:
                hidden = F.dropout(hidden, p=self.dropouts[layer_index], training=self.training)
            master_representation = self.readout_projs[layer_index](hidden[master_indices])
            master_representations.append(master_representation)
            if layer_index < self.num_layers - 1:
                gate_input = master_representation if layer_index == 0 else master_representation - master_representations[layer_index - 1]
                stop_logits.append(self.stop_gates[layer_index](gate_input).squeeze(-1))

        if USE_AIOM:
            weights, ponder_cost = self._soft_halting(stop_logits)
            stacked = torch.stack(master_representations, dim=1)
            master_output = (weights.unsqueeze(-1) * stacked).sum(dim=1)
            expected_depth = (weights * torch.arange(1, self.num_layers + 1, device=x.device).float()).sum(dim=1).mean()
        else:
            master_output = master_representations[-1]
            ponder_cost = torch.zeros((), device=x.device)
            expected_depth = torch.tensor(float(self.num_layers), device=x.device)
        final_input = torch.cat([master_output, embed_pool, pcp_pool, data.z_global_manual, data.z_protparam], dim=1)
        return self.mlp(final_input), ponder_cost, expected_depth


class BiLSTM_Refiner(nn.Module):
    def __init__(self, refine_dim=REFINE_DIM, hidden_dim=512, fc_dim=512, dropout=0.3):
        super().__init__()
        self.refiner = EmbeddingRefiner(ESM_DIM, refine_dim)
        self.lstm = nn.LSTM(refine_dim + ONEHOT_DIM + PCP_DIM, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, fc_dim), nn.BatchNorm1d(fc_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(fc_dim, 1)
        )

    def forward(self, features):
        refined = self.refiner(features[:, :, :ESM_DIM])
        sequence_features = torch.cat([refined, features[:, :, ESM_DIM:]], dim=-1)
        lstm_output, _ = self.lstm(sequence_features)
        attention = torch.softmax(self.attn(lstm_output), dim=1)
        return self.fc(torch.sum(attention * lstm_output, dim=1))
