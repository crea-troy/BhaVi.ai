"""
BhaVi - Complete Architecture
==============================
Connects all five layers into one unified system.

Flow:
  Input
    ↓
  Frozen Core Field       (fundamentals, uncertainty, roots)
    ↓
  Observer Core           (three-zone memory, safe learning)
    ↓
  Curiosity Engine        (gap detection, learning prioritization)
    ↓
  Superposition Cloud     (multiple interpretations simultaneously)
    ↓
  Collapse Decision       (late-binding answer, feedback loop)
    ↓
  Output

Properties:
  - Runs on any hardware (even phone/RPi)
  - ~50M parameters total
  - Self-improving with protected roots
  - Cannot be manipulated away from fundamentals
  - Persistent memory across sessions

Author: BhaVi Project
"""

import torch
import torch.nn as nn
from typing import Optional
import os
import sys
import importlib.util

# ── Robust path fix (works even with spaces in folder names) ──────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _import_from(rel_path, module_name):
    """Import a module by file path — works with any folder name."""
    full_path = os.path.join(_HERE, rel_path)
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_frozen   = _import_from("core/frozen_core.py",                    "frozen_core")
_observer = _import_from("memory/observer_core.py",                 "observer_core")
_curious  = _import_from("curiosity/curiosity_engine.py",           "curiosity_engine")
_supercol = _import_from("superposition/superposition_collapse.py", "superposition_collapse")

FrozenCoreFieldWithInputProjection = _frozen.FrozenCoreFieldWithInputProjection
ObserverCore                        = _observer.ObserverCore
CuriosityEngine                     = _curious.CuriosityEngine
SuperpositionCloud                  = _supercol.SuperpositionCloud
CollapseDecision                    = _supercol.CollapseDecision


class BhaVi(nn.Module):
    """
    BhaVi - Complete Neural Field Architecture

    A genuinely novel AI architecture that:
    1. Thinks in continuous fields, not discrete tokens
    2. Has protected roots that cannot be corrupted
    3. Self-improves through curiosity-driven learning
    4. Holds multiple interpretations simultaneously
    5. Runs efficiently on any hardware
    """

    def __init__(
        self,
        input_dim: int = 256,
        field_dim: int = 256,
        output_dim: int = 256,
        active_memory_slots: int = 512,
        consolidated_memory_slots: int = 1024,
        num_hypotheses: int = 8,
        memory_path: str = "bhavi_memory.pt"
    ):
        super().__init__()

        self.input_dim = input_dim
        self.field_dim = field_dim
        self.output_dim = output_dim
        self.memory_path = memory_path

        print("\n" + "="*50)
        print("  Initializing BhaVi Architecture")
        print("="*50)

        # ── Layer 1: Frozen Core Field ─────────────────────────────
        print("\n[Layer 1] Frozen Core Field")
        self.frozen_core = FrozenCoreFieldWithInputProjection(
            input_dim=input_dim,
            field_dim=field_dim
        )

        # ── Layer 2 (integrated): Dual Precision ──────────────────
        # Handled inside FrozenCore (aleatoric + epistemic split)

        # ── Layer 3: Observer Core ─────────────────────────────────
        print("\n[Layer 3] Observer Core")
        self.observer_core = ObserverCore(
            field_dim=field_dim,
            active_slots=active_memory_slots,
            consolidated_slots=consolidated_memory_slots
        )

        # ── Layer 4: Curiosity Engine ──────────────────────────────
        print("\n[Layer 4] Curiosity Engine")
        self.curiosity_engine = CuriosityEngine(field_dim=field_dim)

        # ── Layer 5: Superposition Cloud ───────────────────────────
        print("\n[Layer 5] Superposition Cloud")
        self.superposition = SuperpositionCloud(
            field_dim=field_dim,
            num_hypotheses=num_hypotheses
        )

        # ── Layer 6: Collapse Decision ─────────────────────────────
        print("\n[Layer 6] Collapse Decision")
        self.collapse = CollapseDecision(
            field_dim=field_dim,
            output_dim=output_dim
        )

        # ── Parameter Count ────────────────────────────────────────
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("\n" + "="*50)
        print(f"  Total Parameters:     {total:>12,}")
        print(f"  Trainable Parameters: {trainable:>12,}")
        print(f"  Memory footprint:     {total * 4 / 1024 / 1024:.1f} MB (FP32)")
        print(f"  Memory footprint:     {total * 1 / 1024 / 1024:.1f} MB (INT8)")
        print("="*50 + "\n")

        # Load persistent memory if exists
        self.observer_core.load_memory(memory_path)

    def forward(
        self,
        x: torch.Tensor,
        force_collapse: bool = False,
        learn: bool = True
    ) -> dict:
        """
        Full BhaVi forward pass.

        Args:
            x: Input tensor [batch, input_dim]
            force_collapse: Force a decision even if uncertain
            learn: Whether to update memory (False for inference only)

        Returns:
            Complete BhaVi output dictionary
        """

        # ── Layer 1: Frozen Core Processing ───────────────────────
        core_output = self.frozen_core(x)
        core_repr = core_output['core_representation']

        # ── Consistency Check ──────────────────────────────────────
        # If core is not yet frozen: allow ALL learning (no blocking)
        # If core is frozen: check consistency against roots
        if self.frozen_core.core.is_frozen():
            consistency = self.frozen_core.core.check_consistency(
                core_repr, core_repr
            )
        else:
            # Core not frozen yet — trust everything, learn freely
            consistency = torch.ones(core_repr.shape[0], 1)
        core_output['consistency_score'] = consistency

        # ── Layer 3: Observer Core ─────────────────────────────────
        observer_output = self.observer_core(
            current_input=core_repr,
            core_representation=core_repr,
            consistency_score=consistency
        ) if learn else self._read_memory_only(core_repr)

        memory_state = observer_output['memory_state']
        memory_confidence = observer_output.get(
            'consolidation_confidence',
            torch.zeros(x.shape[0], 1)
        )

        # ── Layer 4: Curiosity Engine ──────────────────────────────
        curiosity_output = self.curiosity_engine(
            core_output=core_output,
            memory_confidence=memory_confidence
        )

        # ── Layer 5: Superposition Cloud ───────────────────────────
        superposition_output = self.superposition(
            memory_state=memory_state,
            uncertainty=core_output['total_uncertainty']
        )

        # ── Layer 6: Collapse Decision ─────────────────────────────
        collapse_output = self.collapse(
            superposition_output=superposition_output,
            curiosity_output=curiosity_output,
            force_collapse=force_collapse
        )

        # ── Compile full output ────────────────────────────────────
        return {
            # Main output
            'output': collapse_output['decision'],
            'confidence': collapse_output['confidence'],

            # Uncertainty breakdown
            'aleatoric_uncertainty': core_output['aleatoric_uncertainty'],
            'epistemic_uncertainty': core_output['epistemic_uncertainty'],

            # Memory state
            'memory_state': memory_state,
            'memory_confidence': memory_confidence,

            # Curiosity state
            'gap_score': curiosity_output['gap_score'],
            'surprise': curiosity_output['surprise'],
            'should_learn': curiosity_output['should_learn'],
            'learning_blocked': curiosity_output['blocked'],

            # Superposition state
            'superposition_entropy': superposition_output['entropy'],
            'coherence': superposition_output['coherence'],
            'dominant_hypothesis': superposition_output['dominant_hypothesis'],

            # Decision metadata
            'readiness': collapse_output['readiness'],
            'collapsed': collapse_output['collapsed'],

            # Consistency with roots
            'consistency_with_roots': consistency
        }

    def _read_memory_only(self, core_repr: torch.Tensor) -> dict:
        """Read from memory without updating (inference mode)."""
        active_mem = self.observer_core.active_memory(core_repr)
        consolidated_mem, confidence = self.observer_core.consolidated_memory(core_repr)
        fused = self.observer_core.memory_fusion(
            torch.cat([core_repr, active_mem, consolidated_mem], dim=-1)
        )
        return {
            'memory_state': fused,
            'consolidation_confidence': confidence
        }

    def freeze_core(self):
        """
        Permanently freeze the core after initial training.
        Call this when you are satisfied with fundamental training.
        """
        self.frozen_core.freeze_core()
        print("\n[BhaVi] ⚠️  Core frozen. Roots are permanently protected.")

    def save(self, path: str = "bhavi_checkpoint.pt"):
        """Save complete BhaVi state."""
        torch.save({
            'model_state': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'field_dim': self.field_dim,
                'output_dim': self.output_dim,
            }
        }, path)
        # Also save memory separately (can grow independently)
        self.observer_core.save_memory(self.memory_path)
        print(f"[BhaVi] Saved to {path}")

    def load(self, path: str):
        """Load BhaVi state."""
        if os.path.exists(path):
            state = torch.load(path, weights_only=True)
            self.load_state_dict(state['model_state'])
            self.observer_core.load_memory(self.memory_path)
            print(f"[BhaVi] Loaded from {path}")

    def status(self):
        """Print full BhaVi status."""
        print("\n" + "="*50)
        print("  BhaVi Status Report")
        print("="*50)
        print(f"  Core frozen: {self.frozen_core.core.is_frozen()}")
        print(f"  Memory updates: {self.observer_core.update_count}")
        print(f"  Decisions made: {self.collapse.decision_count}")
        print(f"  Decision quality: {self.collapse.get_decision_quality():.3f}")
        self.curiosity_engine.curiosity_summary()
        print("="*50 + "\n")
