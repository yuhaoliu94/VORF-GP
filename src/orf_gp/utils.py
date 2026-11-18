import numpy as np
from numpy.linalg import multi_dot
import random
import os

def symmetric_mult(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return multi_dot([A, B, A.T])

def triple_mult(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    return multi_dot([A, B, C])

def get_random_state(random_state: int = 520):
    return np.random.RandomState(random_state)

def l2_normalize(A: np.ndarray) -> np.ndarray:
    col_norms = np.linalg.norm(A, ord=2, axis=0, keepdims=True)

    A = A / np.maximum(col_norms, 1e-9)
    return A

def dynamic_clip(grad, sigma_multiplier=3.0, min_threshold=0.1):
    std = np.std(grad)
    limit = max(sigma_multiplier * std, min_threshold)
    return np.clip(grad, -limit, limit)


def force_symmetric(A: np.ndarray) -> np.ndarray:
    B = (A + A.T) / 2.0
    diags = B.diagonal()
    if np.any(diags < 1e-6):
        np.fill_diagonal(B, np.maximum(diags, 1e-6))

    return B


def spectral_normalize(A: np.ndarray) -> np.ndarray:
    eig_values = np.linalg.eigvals(A)
    spectral_radius = np.max(np.abs(eig_values))
    A_norm = A / spectral_radius
    return A_norm


def set_global_seeds(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    except ImportError:
        pass

    print(f"✅ Random seed fixed: {seed}")