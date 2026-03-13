"""
BhaVi - Observer Core
=====================
Three-zone persistent memory field.

ZONE 1: Permanent Memory  - frozen fundamentals (never changes)
ZONE 2: Consolidated Memory - slow field (changes with strong evidence)
ZONE 3: Active Memory - fast field (changes every interaction)

Information flows only ONE WAY:
Active → Consolidated → Permanent

Never backwards. This prevents corruption of roots.

Author: BhaVi Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import json
import os


class ActiveMemoryField(nn.Module):
    """
    Zone 3: Fast-updating working memory.
    
    Updates every interaction.
    Can be forgotten if not consolidated.
    Like human working memory.
    ~5M parameters
    """

    def __init__(self, field_dim: int = 256, memory_slots: int = 512):
        super().__init__()

        self.field_dim = field_dim
        self.memory_slots = memory_slots

        # Memory storage as a field
        # Each slot is a point in the field
        self.memory_field = nn.Parameter(
            torch.zeros(memory_slots, field_dim)
        )

        # Write gate - decides what gets stored
        self.write_gate = nn.Sequential(
            nn.Linear(field_dim * 2, field_dim),
            nn.GELU(),
            nn.Linear(field_dim, memory_slots),
            nn.Softmax(dim=-1)
        )

        # Read gate - decides what gets retrieved
        self.read_gate = nn.Sequential(
            nn.Linear(field_dim, field_dim),
            nn.GELU(),
            nn.Linear(field_dim, memory_slots),
            nn.Softmax(dim=-1)
        )

        # Update mechanism
        self.updater = nn.Sequential(
            nn.Linear(field_dim, field_dim),
            nn.GELU(),
            nn.Linear(field_dim, field_dim),
            nn.Tanh()
        )

        print(f"[ActiveMemory] {sum(p.numel() for p in self.parameters()):,} parameters")

    def write(self, new_info: torch.Tensor, query: torch.Tensor):
        """Write new information to active memory."""
        # Decide where to write
        combined = torch.cat([new_info, query], dim=-1)
        write_weights = self.write_gate(combined)  # [batch, slots]

        # Prepare update
        update = self.updater(new_info)  # [batch, field_dim]

        # Write to field using soft addressing
        # [batch, slots, 1] * [1, 1, field_dim]
        delta = write_weights.unsqueeze(-1) * update.unsqueeze(1)
        # Average across batch for field update
        self.memory_field.data += delta.mean(0) * 0.1  # Small learning rate

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Read from active memory based on query."""
        read_weights = self.read_gate(query)  # [batch, slots]
        # Weighted sum of memory slots
        # [batch, slots] x [slots, field_dim] = [batch, field_dim]
        retrieved = torch.matmul(read_weights, self.memory_field)
        return retrieved

    def forward(self, query: torch.Tensor, new_info: Optional[torch.Tensor] = None):
        if new_info is not None:
            self.write(new_info, query)
        return self.read(query)


class ConsolidatedMemoryField(nn.Module):
    """
    Zone 2: Slow-updating long-term memory.

    Updates only when evidence is strong and repeated.
    Like human long-term memory.
    More stable than active memory.
    ~8M parameters
    """

    def __init__(self, field_dim: int = 256, memory_slots: int = 1024):
        super().__init__()

        self.field_dim = field_dim
        self.memory_slots = memory_slots

        # Consolidated field - more stable
        self.consolidated_field = nn.Parameter(
            torch.zeros(memory_slots, field_dim)
        )

        # Evidence counter - tracks how many times something has been seen
        # Not a parameter, just a counter
        self.register_buffer(
            'evidence_counts',
            torch.zeros(memory_slots)
        )

        # Consolidation threshold - how much evidence needed to update
        self.consolidation_threshold = 3  # Must be seen 3+ times

        # Addressing mechanism
        self.address_encoder = nn.Sequential(
            nn.Linear(field_dim, field_dim),
            nn.GELU(),
            nn.Linear(field_dim, memory_slots),
            nn.Softmax(dim=-1)
        )

        # Content encoder
        self.content_encoder = nn.Sequential(
            nn.Linear(field_dim, field_dim),
            nn.LayerNorm(field_dim),
            nn.GELU(),
            nn.Linear(field_dim, field_dim)
        )

        print(f"[ConsolidatedMemory] {sum(p.numel() for p in self.parameters()):,} parameters")

    def consolidate(self, active_memory: torch.Tensor, importance: torch.Tensor):
        """
        Consolidate important information from active memory.

        Only updates if importance is high enough.
        This prevents noise from corrupting long-term memory.
        """
        # Only consolidate high-importance items
        mask = (importance > 0.7).float()  # Threshold

        if mask.sum() == 0:
            return  # Nothing important enough

        # Get addressing weights
        addresses = self.address_encoder(active_memory)  # [batch, slots]

        # Update evidence counts
        self.evidence_counts += (addresses * mask).mean(0)

        # Only update slots with enough evidence
        update_mask = (self.evidence_counts >= self.consolidation_threshold).float()

        # Prepare content update
        content = self.content_encoder(active_memory)  # [batch, field_dim]

        # Write with very slow learning rate (stable memory)
        delta = (addresses * mask).unsqueeze(-1) * content.unsqueeze(1)
        self.consolidated_field.data += (
            delta.mean(0) * update_mask.unsqueeze(-1) * 0.01  # Very slow
        )

    def read(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read from consolidated memory."""
        addresses = self.address_encoder(query)
        retrieved = torch.matmul(addresses, self.consolidated_field)

        # Also return confidence based on evidence counts
        confidence = (addresses * self.evidence_counts.unsqueeze(0)).sum(-1, keepdim=True)
        confidence = torch.sigmoid(confidence / 10.0)

        return retrieved, confidence

    def forward(self, query: torch.Tensor):
        return self.read(query)


class ObserverCore(nn.Module):
    """
    BhaVi Observer Core - Complete Three-Zone Memory System

    Zone 1: Permanent (from FrozenCore - passed in)
    Zone 2: Consolidated (slow learning)  
    Zone 3: Active (fast learning)

    Total: ~20M parameters
    """

    def __init__(
        self,
        field_dim: int = 256,
        active_slots: int = 512,
        consolidated_slots: int = 1024
    ):
        super().__init__()

        self.field_dim = field_dim

        # Zone 3: Active memory
        self.active_memory = ActiveMemoryField(field_dim, active_slots)

        # Zone 2: Consolidated memory
        self.consolidated_memory = ConsolidatedMemoryField(field_dim, consolidated_slots)

        # Importance estimator
        # Decides if something is worth consolidating
        self.importance_estimator = nn.Sequential(
            nn.Linear(field_dim * 2, field_dim),
            nn.GELU(),
            nn.Linear(field_dim, 1),
            nn.Sigmoid()
        )

        # Memory fusion
        # Combines all three zones into unified representation
        self.memory_fusion = nn.Sequential(
            nn.Linear(field_dim * 3, field_dim * 2),
            nn.GELU(),
            nn.Linear(field_dim * 2, field_dim),
            nn.LayerNorm(field_dim)
        )

        # Self-update tracker
        self.update_count = 0

        total = sum(p.numel() for p in self.parameters())
        print(f"[ObserverCore] Total: {total:,} parameters")

    def observe(
        self,
        current_input: torch.Tensor,
        core_representation: torch.Tensor,
        consistency_score: torch.Tensor
    ) -> dict:
        """
        Observe new information and update memory accordingly.

        Only updates memory if:
        1. Consistency score is high (not contradicting roots)
        2. Information is important enough

        Args:
            current_input: Current processed input [batch, field_dim]
            core_representation: Output from frozen core [batch, field_dim]
            consistency_score: How consistent with roots [batch, 1]

        Returns:
            Memory state dictionary
        """

        # ── Read from all memory zones ─────────────────────────────
        active_mem = self.active_memory(current_input)
        consolidated_mem, consolidation_confidence = self.consolidated_memory(current_input)

        # ── Estimate importance ────────────────────────────────────
        importance_input = torch.cat([current_input, core_representation], dim=-1)
        importance = self.importance_estimator(importance_input)

        # ── Safety check before writing ───────────────────────────
        # Only write to memory if consistent with frozen core
        safe_to_learn = (consistency_score > 0.5).float()

        # ── Update active memory (Zone 3) ──────────────────────────
        # Always updates if safe (fast memory)
        if safe_to_learn.mean() > 0.3:  # If most of batch is safe
            self.active_memory(
                current_input,
                new_info=current_input * safe_to_learn
            )

        # ── Consolidate to Zone 2 ──────────────────────────────────
        # Only consolidates if important AND consistent
        consolidation_signal = importance * safe_to_learn
        self.consolidated_memory.consolidate(active_mem, consolidation_signal.squeeze(-1))

        # ── Fuse all memory zones ──────────────────────────────────
        fused = self.memory_fusion(
            torch.cat([core_representation, active_mem, consolidated_mem], dim=-1)
        )

        self.update_count += 1

        return {
            'memory_state': fused,
            'active_memory': active_mem,
            'consolidated_memory': consolidated_mem,
            'consolidation_confidence': consolidation_confidence,
            'importance': importance,
            'safe_to_learn': safe_to_learn,
            'update_count': self.update_count
        }

    def forward(
        self,
        current_input: torch.Tensor,
        core_representation: torch.Tensor,
        consistency_score: torch.Tensor
    ) -> dict:
        return self.observe(current_input, core_representation, consistency_score)

    def save_memory(self, path: str):
        """Save memory state to disk for persistence."""
        torch.save({
            'active_field': self.active_memory.memory_field.data,
            'consolidated_field': self.consolidated_memory.consolidated_field.data,
            'evidence_counts': self.consolidated_memory.evidence_counts,
            'update_count': self.update_count
        }, path)
        print(f"[ObserverCore] Memory saved to {path}")

    def load_memory(self, path: str):
        """Load memory state from disk."""
        if os.path.exists(path):
            state = torch.load(path, weights_only=True)
            self.active_memory.memory_field.data = state['active_field']
            self.consolidated_memory.consolidated_field.data = state['consolidated_field']
            self.consolidated_memory.evidence_counts = state['evidence_counts']
            self.update_count = state['update_count']
            print(f"[ObserverCore] Memory loaded from {path}")
            print(f"[ObserverCore] Resuming from update #{self.update_count}")
        else:
            print(f"[ObserverCore] No saved memory found at {path}, starting fresh")
