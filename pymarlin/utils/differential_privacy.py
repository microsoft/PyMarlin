"""Differential Privacy utils"""
from dataclasses import dataclass

@dataclass
class DifferentialPrivacyArguments:
    noise_multiplier: float = 1.0
    per_sample_max_grad_norm: float = 1.0
    sample_rate: float = 1.0
    delta: float = 1e-5

# Wrap any No-DP optimizer to distinguish from the DP optimizer
# This is expected to be a very rare situation
class NoDPWrap:
    def __init__(self, optimizer):
        self.optimizer = optimizer
