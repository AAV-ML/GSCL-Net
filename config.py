"""Project configuration and reproducibility helpers."""
import math
import random

import numpy as np
import torch

AA_PROPERTIES_EXTENDED = {
    'A': [1.8, 0.0, 0.0, 0.357, 0.0, 0.61, 1.42, 0.83, 0.66], 'R': [-4.5, 1.0, 1.0, 0.301, 0.0, 1.80, 0.98, 0.93, 0.95],
    'N': [-3.5, 0.0, 1.0, 0.463, 1.0, 1.60, 0.76, 0.89, 1.46], 'D': [-3.5, -1.0, 1.0, 0.476, 1.0, 1.50, 0.72, 0.90, 1.41],
    'C': [2.5, 0.0, 0.0, 0.346, 0.0, 1.30, 0.70, 1.19, 0.74], 'E': [-3.5, -1.0, 1.0, 0.398, 1.0, 1.90, 1.50, 0.75, 0.97],
    'Q': [-3.5, 0.0, 1.0, 0.418, 1.0, 1.80, 1.27, 0.80, 1.23], 'G': [-0.4, 0.0, 0.0, 0.544, 0.0, 0.40, 0.57, 0.75, 1.56],
    'H': [-3.2, 0.0, 1.0, 0.322, 1.0, 1.70, 0.89, 0.87, 0.96], 'I': [4.5, 0.0, 0.0, 0.375, 0.0, 1.60, 1.09, 1.67, 0.47],
    'L': [3.8, 0.0, 0.0, 0.369, 0.0, 1.60, 1.34, 1.22, 0.61], 'K': [-3.9, 1.0, 1.0, 0.316, 1.0, 1.70, 1.07, 0.73, 1.01],
    'M': [1.9, 0.0, 0.0, 0.367, 0.0, 1.70, 1.45, 1.05, 0.60], 'F': [2.8, 0.0, 0.0, 0.305, 0.0, 2.00, 1.12, 1.28, 0.60],
    'P': [-1.6, 0.0, 0.0, 0.477, 0.0, 1.40, 0.57, 0.55, 1.52], 'S': [-0.8, 0.0, 1.0, 0.509, 1.0, 0.90, 0.75, 0.96, 1.43],
    'T': [-0.7, 0.0, 1.0, 0.444, 1.0, 1.10, 0.83, 1.19, 0.96], 'W': [-0.9, 0.0, 0.0, 0.303, 0.0, 2.40, 1.02, 1.14, 0.75],
    'Y': [-1.3, 0.0, 1.0, 0.366, 0.0, 1.90, 1.08, 1.25, 0.69], 'V': [4.2, 0.0, 0.0, 0.386, 0.0, 1.40, 1.06, 1.70, 0.50]
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_SEQ_LENGTH = 7
AMINO_ACIDS = list(AA_PROPERTIES_EXTENDED)
ESM_DIM = 480
# ESM-2 checkpoint used to generate the bundled per-residue embeddings.
ESM_MODEL_NAME = 'esm2_t12_35M_UR50D'
ONEHOT_DIM = 20
USE_ESM = True
USE_ONEHOT = True
EMBED_DIM = (ESM_DIM if USE_ESM else 0) + (ONEHOT_DIM if USE_ONEHOT else 0)
POS_ENCODING_DIM = MAX_SEQ_LENGTH
PCP_DIM = 9
NODE_FEATURE_DIM = EMBED_DIM + POS_ENCODING_DIM + PCP_DIM
GLOBAL_MANUAL_DIM = 2
GLOBAL_PROTPARAM_DIM = 4
LSTM_INPUT_DIM = EMBED_DIM + PCP_DIM
USE_AIOM = True
AIOM_NUM_LAYERS = 3
AIOM_PONDER_LAMBDA = 0.05
AIOM_GATE_TEMPERATURE = 0.5
REFINE_DIM = 128
BATCH_SIZE = 500
GNN_LR = 0.0008
LSTM_LR = 0.001
GNN_WEIGHT_DECAY = 0.001
LSTM_WEIGHT_DECAY = 0.001
EPOCHS = 300
PATIENCE = 30
N_TRAIN = 24000
N_VAL = 5000
BASE_LEARNER_SEED = 123
DATA_SPLIT_SEED = 17
HYDROPHOBICITY_SCALE = {aa: props[0] for aa, props in AA_PROPERTIES_EXTENDED.items()}
THETA = math.radians(100)
AGG_RESIDUES = {'C', 'P', 'W', 'F', 'I', 'L', 'V'}
SOL_RESIDUES = {'R', 'K', 'D', 'E'}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
