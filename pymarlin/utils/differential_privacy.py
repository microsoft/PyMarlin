"""Differential Privacy utils"""
from typing import Optional
from dataclasses import dataclass

@dataclass
class DifferentialPrivacyArguments:
    noise_multiplier: float = 1.0   # Scaling for the noise variance
    per_sample_max_grad_norm: float = 1.0    # Clips the per sample gradients
    sample_rate: float = 0.0   # Should be set as batch_size/number_of_samples (see doc for special cases)
    delta: Optional[float] = None   # Typically set as o(1/number_of_samples), only required to calculate privacy budget (epsilon)

# Wrap any No-DP optimizer to distinguish from the DP optimizer
# This is expected to be a very rare situation
class NoDPWrap:
    def __init__(self, optimizer):
        self.optimizer = optimizer
