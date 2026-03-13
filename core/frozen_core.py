"""
BhaVi - Frozen Core Field
=========================
This is the ROOT of BhaVi.

It holds fundamental principles that NEVER change.
Cannot be updated by training.
Cannot be manipulated by new data.
Everything else in BhaVi builds ON TOP of this.

Think of it like Maxwell's equations -
they never change, but infinite phenomena emerge from them.

Author: BhaVi Project
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class FrozenCoreField(nn.Module):
    """
    The immutable foundation of BhaVi.

    This field encodes fundamental reasoning principles:
    1. Uncertainty is always present (aleatoric + epistemic)
    2. Multiple interpretations can coexist (superposition)
    3. Curiosity drives learning (never satisfied)
    4. Decisions require evidence (not just pattern matching)
    5. Memory has structure (roots vs branches)

    Parameters: ~1M (tiny but fundamental)
    Once trained on fundamentals: ALL WEIGHTS FROZEN FOREVER
    """

    def __init__(self, field_dim: int = 256):
        super().__init__()

        self.field_dim = field_dim

        # ── Fundamental Field Encoder ──────────────────────────────
        # Encodes ANY input into the fundamental representation space
        # This is the lens through which BhaVi sees everything
        self.field_encoder = nn.Sequential(
            nn.Linear(field_dim, field_dim),
            nn.LayerNorm(field_dim),
            nn.GELU(),
            nn.Linear(field_dim, field_dim),
            nn.LayerNorm(field_dim),
        )

        # ── Fundamental Uncertainty Estimator ─────────────────────
        # Hardcodes the principle: everything has uncertainty
        # Outputs (aleatoric_uncertainty, epistemic_uncertainty)
        self.uncertainty_principle = nn.Sequential(
            nn.Linear(field_dim, field_dim // 2),
            nn.GELU(),
            nn.Linear(field_dim // 2, 2),  # [aleatoric, epistemic]
            nn.Softplus()                   # Always positive
        )

        # ── Fundamental Structure Detector ────────────────────────
        # Detects the deep structure of any input
        # Learns what is "fundamental" vs "surface"
        self.structure_detector = nn.Sequential(
            nn.Linear(field_dim, field_dim),
            nn.GELU(),
            nn.Linear(field_dim, field_dim),
            nn.Tanh()
        )

        # ── Consistency Checker ───────────────────────────────────
        # Checks if new information is consistent with fundamentals
        # Returns consistency score [0, 1]
        # 1.0 = fully consistent with roots
        # 0.0 = contradicts fundamentals
        self.consistency_checker = nn.Sequential(
            nn.Linear(field_dim * 2, field_dim),
            nn.GELU(),
            nn.Linear(field_dim, 1),
            nn.Sigmoid()
        )

        # ── Field Resonance ───────────────────────────────────────
        # Measures how much input "resonates" with core knowledge
        # High resonance = input relates to known fundamentals
        # Low resonance = genuinely new territory
        self.resonance_field = nn.Parameter(
            torch.randn(field_dim, field_dim) * 0.01,
            requires_grad=True  # Will be frozen after initial training
        )

        # Track parameter count
        self._param_count = sum(p.numel() for p in self.parameters())
        print(f"[FrozenCore] Initialized with {self._param_count:,} parameters")

    def forward(self, x: torch.Tensor) -> dict:
        """
        Process input through the frozen core.

        Args:
            x: Input tensor [batch, field_dim]

        Returns:
            dict containing:
            - core_representation: fundamental encoding
            - aleatoric_uncertainty: data noise level
            - epistemic_uncertainty: knowledge gap level
            - structure: deep structural features
            - resonance: how much input relates to known fundamentals
        """
        # Encode into fundamental space
        core_repr = self.field_encoder(x)

        # Estimate fundamental uncertainties
        uncertainties = self.uncertainty_principle(core_repr)
        aleatoric = uncertainties[:, 0:1]   # Data noise - irreducible
        epistemic = uncertainties[:, 1:2]   # Knowledge gap - reducible

        # Detect deep structure
        structure = self.structure_detector(core_repr)

        # Compute field resonance
        # How much does this input align with the fundamental field?
        resonance = torch.sigmoid(
            torch.einsum('bi,ij,bj->b', core_repr, self.resonance_field, core_repr)
        ).unsqueeze(1)

        return {
            'core_representation': core_repr,
            'aleatoric_uncertainty': aleatoric,
            'epistemic_uncertainty': epistemic,
            'structure': structure,
            'resonance': resonance,
            'total_uncertainty': aleatoric + epistemic
        }

    def check_consistency(
        self,
        new_info: torch.Tensor,
        core_repr: torch.Tensor
    ) -> torch.Tensor:
        """
        Check if new information is consistent with frozen core.

        This is the GUARDIAN function.
        Before BhaVi learns anything new, it checks:
        "Does this contradict my fundamental roots?"

        Returns:
            consistency score [0,1] per batch item
            1.0 = safe to learn, consistent with roots
            0.0 = contradicts fundamentals, do NOT learn
        """
        combined = torch.cat([new_info, core_repr], dim=-1)
        return self.consistency_checker(combined)

    def freeze(self):
        """
        PERMANENTLY freeze all core parameters.
        Call this after initial training on fundamentals.
        After this, the core NEVER changes.
        """
        for param in self.parameters():
            param.requires_grad = False

        print("[FrozenCore] ⚠️  Core is now PERMANENTLY FROZEN")
        print("[FrozenCore] Roots are protected. Nothing can change them.")

    def is_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.parameters())

    def parameter_count(self) -> int:
        return self._param_count


class FrozenCoreFieldWithInputProjection(nn.Module):
    """
    Complete Frozen Core with flexible input handling.

    Accepts variable-size inputs and projects to field_dim.
    This is what other BhaVi layers actually use.
    """

    def __init__(self, input_dim: int, field_dim: int = 256):
        super().__init__()

        self.input_dim = input_dim
        self.field_dim = field_dim

        # Project any input size to field dimension
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, field_dim),
            nn.LayerNorm(field_dim),
            nn.GELU()
        )

        # The actual frozen core
        self.core = FrozenCoreField(field_dim)

        total = sum(p.numel() for p in self.parameters())
        print(f"[FrozenCore+Projection] Total: {total:,} parameters")

    def forward(self, x: torch.Tensor) -> dict:
        # Project input to field dimension
        projected = self.input_projection(x)
        # Process through frozen core
        return self.core(projected)

    def freeze_core(self):
        """Freeze only the core, keep projection trainable."""
        self.core.freeze()

    def freeze_all(self):
        """Freeze everything."""
        for param in self.parameters():
            param.requires_grad = False
