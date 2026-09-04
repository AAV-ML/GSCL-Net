"""Sequence features and PyG/Torch dataset construction."""
import math
import os
import sys
from typing import List

import numpy as np
import torch
from Bio.SeqUtils import ProtParamData
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from torch_geometric.data import Data
from tqdm import tqdm

from config import (
    AA_PROPERTIES_EXTENDED, AMINO_ACIDS, EMBED_DIM, GLOBAL_PROTPARAM_DIM,
    MAX_SEQ_LENGTH, NODE_FEATURE_DIM, ONEHOT_DIM, PCP_DIM, ESM_DIM, USE_ESM,
)


def calculate_hydrophobic_moment(sequence: str) -> float:
    if not sequence:
        return 0.0
    x_sum = y_sum = 0.0
    for index, amino_acid in enumerate(sequence):
        hydrophobicity = AA_PROPERTIES_EXTENDED.get(amino_acid, [0.0])[0]
        angle = index * math.radians(100)
        x_sum += hydrophobicity * math.cos(angle)
        y_sum += hydrophobicity * math.sin(angle)
    return math.sqrt(x_sum ** 2 + y_sum ** 2) / len(sequence)


def calculate_aggregation_ratio(sequence: str) -> float:
    aggregation_residues = {'C', 'P', 'W', 'F', 'I', 'L', 'V'}
    soluble_residues = {'R', 'K', 'D', 'E'}
    aggregation_count = sum(amino_acid in aggregation_residues for amino_acid in sequence)
    soluble_count = sum(amino_acid in soluble_residues for amino_acid in sequence)
    return aggregation_count / (soluble_count + 1)


def onehot_encoding(sequence: str) -> np.ndarray:
    onehot = np.zeros((len(sequence), ONEHOT_DIM), dtype=np.float32)
    amino_acid_to_index = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
    for index, amino_acid in enumerate(sequence):
        if amino_acid in amino_acid_to_index:
            onehot[index, amino_acid_to_index[amino_acid]] = 1.0
    return onehot


class ProteinDataBuilder:
    def __init__(self, esm_tensor_path=None):
        if esm_tensor_path is None:
            esm_tensor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embeddings', 'esm2_features_aligned.pt')
        self.valid_protparam_chars = set(getattr(ProtParamData, 'standard_amino_acids', 'ACDEFGHIKLMNPQRSTVWY'))
        if USE_ESM:
            print(f'Loading ESM tensor: {esm_tensor_path}', flush=True)
            try:
                self.aligned_esm_tensor = torch.load(esm_tensor_path, map_location='cpu')
            except FileNotFoundError:
                print(f'ERROR: ESM feature file not found: {esm_tensor_path}', flush=True)
                sys.exit(1)
        else:
            self.aligned_esm_tensor = None

    def _get_physchem_features(self, sequence: str) -> np.ndarray:
        matrix = np.zeros((len(sequence), PCP_DIM), dtype=np.float32)
        for index, amino_acid in enumerate(sequence):
            matrix[index] = AA_PROPERTIES_EXTENDED.get(amino_acid, [0.0] * PCP_DIM)
        return matrix

    def _get_positional_encoding(self, sequence_length: int) -> np.ndarray:
        matrix = np.zeros((sequence_length, MAX_SEQ_LENGTH), dtype=np.float32)
        for index in range(min(sequence_length, MAX_SEQ_LENGTH)):
            matrix[index, index] = 1.0
        return matrix

    def _create_graph_edges(self, sequence_length: int) -> List[List[int]]:
        edge_index = [[], []]
        master_index = sequence_length
        for index in range(sequence_length):
            edge_index[0].extend([index, master_index])
            edge_index[1].extend([master_index, index])
            if index < sequence_length - 1:
                edge_index[0].extend([index, index + 1])
                edge_index[1].extend([index + 1, index])
        return edge_index

    def _calculate_protparam_features(self, sequence: str) -> List[float]:
        cleaned = ''.join(char for char in sequence.upper() if char in self.valid_protparam_chars)
        if not cleaned:
            return [0.0] * GLOBAL_PROTPARAM_DIM
        try:
            analyzed = ProteinAnalysis(cleaned)
            try:
                isoelectric_point = analyzed.isoelectric_point()
            except ValueError:
                isoelectric_point = 7.0
            values = [analyzed.instability_index(), isoelectric_point, analyzed.aromaticity(), analyzed.gravy()]
            return [value if np.isfinite(value) else 0.0 for value in values]
        except Exception:
            return [0.0] * GLOBAL_PROTPARAM_DIM

    def _build_graph(self, sequence, target, protparam_features, embed_matrix) -> Data:
        physicochemical = self._get_physchem_features(sequence)
        positional = self._get_positional_encoding(len(sequence))
        amino_acid_features = np.concatenate([embed_matrix, positional, physicochemical], axis=1)
        master_features = np.zeros((1, NODE_FEATURE_DIM), dtype=np.float32)
        node_features = np.concatenate([amino_acid_features, master_features], axis=0)
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(self._create_graph_edges(len(sequence)), dtype=torch.long),
            num_nodes=len(sequence) + 1,
            z_global_manual=torch.tensor([calculate_hydrophobic_moment(sequence), calculate_aggregation_ratio(sequence)], dtype=torch.float).view(1, -1),
            z_protparam=torch.tensor(protparam_features, dtype=torch.float).view(1, -1),
        )
        if target is not None:
            data.y = torch.tensor([target], dtype=torch.float)
        return data

    def build_all_data(self, dataframe, target_column: str):
        graphs, lstm_features, targets, valid_indices = [], [], [], []
        protparam_cache = {sequence: self._calculate_protparam_features(sequence) for sequence in tqdm(dataframe['AA'].unique(), desc='Computing ProtParam')}
        for row_number, (index, row) in enumerate(tqdm(dataframe.iterrows(), total=len(dataframe), desc='Building datasets')):
            sequence = row['AA']
            target = row.get(target_column)
            embed_parts = []
            if USE_ESM:
                embed_parts.append(self.aligned_esm_tensor[row_number].numpy())
            embed_parts.append(onehot_encoding(sequence))
            embed_matrix = np.concatenate(embed_parts, axis=1)
            graphs.append(self._build_graph(sequence, target, protparam_cache[sequence], embed_matrix))
            lstm_features.append(np.concatenate([embed_matrix, self._get_physchem_features(sequence)], axis=1))
            targets.append(target)
            valid_indices.append(index)
        return {
            'gnn_graphs': graphs,
            'lstm_features': torch.tensor(np.asarray(lstm_features), dtype=torch.float),
            'targets': torch.tensor(np.asarray(targets), dtype=torch.float).view(-1, 1),
            'valid_indices': np.asarray(valid_indices),
        }
