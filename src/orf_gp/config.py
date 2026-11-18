from dataclasses import dataclass, field
from typing import Dict, Any
from datetime import datetime

@dataclass
class GPConfig:
    rff_model: str = 'Orthogonal'  # Feature family
    model_seed: int = 520  # RNG seed

    J: int = 100  # Features
    Dx: int = 0  # Inputs
    Dy: int = 0  # Outputs

    sigma_l: float = 1.0  # Prior variance
    gamma_obs: float = 1.0  # Forgetting factor
    MC: int = 50  # MC samples

    beta_kl: float = 1.0  # KL weight
    beta_ortho: float = 1.0  # Orthogonality weight
    meta_lr: dict = field(default_factory=lambda: {  # Parameter learning rates
        'mu': 1e-2,
        'b': 1e-2,
        'l': 1e-2,
        'sigma': 1e-2
    })
    init_gamma: float | None = None  # Initial RBF gamma
    init_log_l: float | None = None  # Initial log lengthscale
    init_sigma: float | None = None  # Initial frequency std
    alpha_meta: float = 1e-2  # Hyper learning rate
    fixed_step: int = -1  # Warmup steps

    log_interval: int = 500  # Report interval
    verbal: bool = True  # Verbose output
