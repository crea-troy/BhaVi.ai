"""
BhaVi - Curiosity Engine
========================
The self-improvement driver of BhaVi.

Detects epistemic gaps (what BhaVi doesn't know well enough).
Prioritizes what to learn next.
CHECKS AGAINST ROOTS before allowing learning.
Never satisfied - always looking for gaps.

This is what makes BhaVi self-improving.

Author: BhaVi Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math


class EpistemicGapDetector(nn.Module):
    """
    Detects where BhaVi's knowledge has gaps.

    High gap score = BhaVi doesn't understand this well
    Low gap score = BhaVi knows this area well

    Uses the epistemic uncertainty from FrozenCore
    combined with memory confidence to find gaps.
    """

    def __init__(self, field_dim: int = 256):
        super().__init__()

        # Gap detection network
        self.gap_detector = nn.Sequential(
            nn.Linear(field_dim + 2, field_dim // 2),  # +2 for uncertainties
            nn.GELU(),
            nn.Linear(field_dim // 2, field_dim // 4),
            nn.GELU(),
            nn.Linear(field_dim // 4, 1),
            nn.Sigmoid()
        )

        # Gap characterizer - what KIND of gap is this?
        self.gap_characterizer = nn.Sequential(
            nn.Linear(field_dim, field_dim // 2),
            nn.GELU(),
            nn.Linear(field_dim // 2, 4),  # 4 gap types
            nn.Softmax(dim=-1)
        )
        # Gap types:
        # 0: factual gap (missing knowledge)
        # 1: reasoning gap (can't connect concepts)
        # 2: uncertainty gap (too noisy to understand)
        # 3: structural gap (missing fundamental understanding)

    def forward(
        self,
        core_repr: torch.Tensor,
        epistemic_uncertainty: torch.Tensor,
        aleatoric_uncertainty: torch.Tensor,
        memory_confidence: torch.Tensor
    ) -> dict:

        # Combine signals to detect gap
        gap_input = torch.cat([
            core_repr,
            epistemic_uncertainty,
            aleatoric_uncertainty
        ], dim=-1)

        # High epistemic uncertainty + low memory confidence = big gap
        gap_score = self.gap_detector(gap_input)

        # Adjust by memory confidence
        # If memory is confident, gap is smaller
        adjusted_gap = gap_score * (1.0 - memory_confidence * 0.5)

        # Characterize the gap
        gap_type = self.gap_characterizer(core_repr)

        return {
            'gap_score': adjusted_gap,
            'gap_type': gap_type,
            'raw_gap': gap_score
        }


class LearningPrioritizer(nn.Module):
    """
    Decides WHAT to learn next.

    Prioritizes based on:
    - Gap size (bigger gap = higher priority)
    - Gap type (structural gaps > factual gaps)
    - Consistency with roots (inconsistent = DO NOT learn)
    - Potential usefulness (will this help future reasoning?)
    """

    def __init__(self, field_dim: int = 256):
        super().__init__()

        # Priority scorer
        self.priority_scorer = nn.Sequential(
            nn.Linear(field_dim + 1 + 4 + 1, field_dim // 2),
            # field_dim + gap_score + gap_type + consistency
            nn.GELU(),
            nn.Linear(field_dim // 2, 1),
            nn.Sigmoid()
        )

        # Learning signal generator
        # What gradient direction should learning go?
        self.learning_signal = nn.Sequential(
            nn.Linear(field_dim, field_dim),
            nn.GELU(),
            nn.Linear(field_dim, field_dim),
            nn.Tanh()
        )

    def forward(
        self,
        core_repr: torch.Tensor,
        gap_score: torch.Tensor,
        gap_type: torch.Tensor,
        consistency_score: torch.Tensor
    ) -> dict:

        # Combine all signals
        priority_input = torch.cat([
            core_repr,
            gap_score,
            gap_type,
            consistency_score
        ], dim=-1)

        # Compute priority
        priority = self.priority_scorer(priority_input)

        # SAFETY CHECK: Zero out learning for inconsistent inputs
        # This is the guardian that protects roots
        safe_priority = priority * (consistency_score > 0.5).float()

        # Generate learning direction signal
        learn_signal = self.learning_signal(core_repr)

        return {
            'priority': priority,
            'safe_priority': safe_priority,
            'learning_signal': learn_signal,
            'blocked': (consistency_score <= 0.5).float()
        }


class CuriosityEngine(nn.Module):
    """
    BhaVi Curiosity Engine - Complete Self-Improvement Driver

    Always ON. Always looking for gaps.
    Always checking roots before learning.
    Drives continuous self-improvement.

    Total: ~10M parameters
    """

    def __init__(self, field_dim: int = 256):
        super().__init__()

        self.field_dim = field_dim

        # Gap detection
        self.gap_detector = EpistemicGapDetector(field_dim)

        # Learning prioritization
        self.prioritizer = LearningPrioritizer(field_dim)

        # Curiosity state - what is BhaVi currently curious about?
        # This evolves over time
        self.curiosity_state = nn.Parameter(
            torch.randn(field_dim) * 0.01
        )

        # Curiosity update mechanism
        self.curiosity_updater = nn.GRUCell(field_dim, field_dim)

        # Surprise detector
        # High surprise = encountered something very unexpected
        self.surprise_detector = nn.Sequential(
            nn.Linear(field_dim * 2, field_dim),
            nn.GELU(),
            nn.Linear(field_dim, 1),
            nn.Sigmoid()
        )

        # Learning history (lightweight)
        self.register_buffer('gap_history', torch.zeros(100))
        self.history_idx = 0

        # Statistics
        self.blocked_count = 0
        self.learned_count = 0
        self.total_count = 0

        total = sum(p.numel() for p in self.parameters())
        print(f"[CuriosityEngine] {total:,} parameters")

    def forward(
        self,
        core_output: dict,
        memory_confidence: torch.Tensor
    ) -> dict:
        """
        Process through curiosity engine.

        Args:
            core_output: Output from FrozenCoreField
            memory_confidence: Confidence from ObserverCore

        Returns:
            Curiosity analysis and learning directives
        """
        core_repr = core_output['core_representation']
        epistemic_unc = core_output['epistemic_uncertainty']
        aleatoric_unc = core_output['aleatoric_uncertainty']
        consistency = core_output.get('consistency_score',
                                       torch.ones(core_repr.shape[0], 1))

        # ── Detect knowledge gaps ──────────────────────────────────
        gap_info = self.gap_detector(
            core_repr,
            epistemic_unc,
            aleatoric_unc,
            memory_confidence
        )

        # ── Prioritize learning ────────────────────────────────────
        priority_info = self.prioritizer(
            core_repr,
            gap_info['gap_score'],
            gap_info['gap_type'],
            consistency
        )

        # ── Detect surprise ───────────────────────────────────────
        # Compare current input to curiosity state
        batch_size = core_repr.shape[0]
        curiosity_expanded = self.curiosity_state.unsqueeze(0).expand(batch_size, -1)
        surprise_input = torch.cat([core_repr, curiosity_expanded], dim=-1)
        surprise = self.surprise_detector(surprise_input)

        # ── Update curiosity state ─────────────────────────────────
        # Curiosity evolves: focus shifts to high-gap areas
        update_signal = (
            core_repr * gap_info['gap_score'] * priority_info['safe_priority']
        ).mean(0)  # Average across batch

        new_curiosity = self.curiosity_updater(
            update_signal.unsqueeze(0),
            self.curiosity_state.unsqueeze(0)
        ).squeeze(0)
        self.curiosity_state.data = new_curiosity.detach()

        # ── Track statistics ───────────────────────────────────────
        self.total_count += batch_size
        self.blocked_count += int(priority_info['blocked'].sum().item())
        self.learned_count += int(
            (priority_info['safe_priority'] > 0.5).sum().item()
        )

        # Track gap history
        avg_gap = gap_info['gap_score'].mean().item()
        self.gap_history[self.history_idx % 100] = avg_gap
        self.history_idx += 1

        return {
            'gap_score': gap_info['gap_score'],
            'gap_type': gap_info['gap_type'],
            'priority': priority_info['priority'],
            'safe_priority': priority_info['safe_priority'],
            'learning_signal': priority_info['learning_signal'],
            'surprise': surprise,
            'curiosity_state': self.curiosity_state,
            'blocked': priority_info['blocked'],
            'should_learn': priority_info['safe_priority'] > 0.5
        }

    def get_stats(self) -> dict:
        """Get curiosity engine statistics."""
        recent_gaps = self.gap_history[
            :min(self.history_idx, 100)
        ]
        return {
            'total_processed': self.total_count,
            'learning_blocked': self.blocked_count,
            'learning_allowed': self.learned_count,
            'block_rate': self.blocked_count / max(1, self.total_count),
            'avg_recent_gap': recent_gaps.mean().item() if len(recent_gaps) > 0 else 0,
            'current_curiosity_norm': self.curiosity_state.norm().item()
        }

    def curiosity_summary(self):
        """Print human-readable curiosity status."""
        stats = self.get_stats()
        print("\n[CuriosityEngine] Status:")
        print(f"  Total processed:    {stats['total_processed']}")
        print(f"  Learning allowed:   {stats['learning_allowed']}")
        print(f"  Learning blocked:   {stats['learning_blocked']} "
              f"(root protection)")
        print(f"  Block rate:         {stats['block_rate']:.1%}")
        print(f"  Avg knowledge gap:  {stats['avg_recent_gap']:.3f}")
        print(f"  Curiosity strength: {stats['current_curiosity_norm']:.3f}")
