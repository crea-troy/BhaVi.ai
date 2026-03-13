"""
BhaVi - Superposition Cloud + Collapse Decision
================================================
Superposition Cloud:
    Holds multiple interpretations simultaneously.
    Like quantum superposition - many states at once.
    Collapses only when evidence is strong enough.

Collapse Decision:
    Late-binding answer resolution.
    Never collapses prematurely.
    Learns from feedback to improve future decisions.

Author: BhaVi Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class SuperpositionCloud(nn.Module):
    """
    Holds multiple interpretations simultaneously.

    Ψ(x,t) = Σ αᵢ(t) · φᵢ(x)

    The structure of superposition is frozen (rules don't change).
    The weights αᵢ evolve with evidence.

    Total: ~10M parameters
    """

    def __init__(
        self,
        field_dim: int = 256,
        num_hypotheses: int = 8,
        hypothesis_dim: int = 128
    ):
        super().__init__()

        self.field_dim = field_dim
        self.num_hypotheses = num_hypotheses
        self.hypothesis_dim = hypothesis_dim

        # Hypothesis generators
        # Each generates one possible interpretation
        self.hypothesis_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(field_dim, hypothesis_dim),
                nn.GELU(),
                nn.Linear(hypothesis_dim, hypothesis_dim),
                nn.LayerNorm(hypothesis_dim)
            )
            for _ in range(num_hypotheses)
        ])

        # Amplitude estimator
        # αᵢ = how much does each hypothesis contribute?
        self.amplitude_estimator = nn.Sequential(
            nn.Linear(field_dim, field_dim // 2),
            nn.GELU(),
            nn.Linear(field_dim // 2, num_hypotheses),
            nn.Softmax(dim=-1)  # Amplitudes sum to 1
        )

        # Interference calculator
        # Some hypotheses reinforce each other, some cancel
        self.interference = nn.Bilinear(hypothesis_dim, hypothesis_dim, 1)

        # Superposition fusion
        self.fusion = nn.Sequential(
            nn.Linear(hypothesis_dim * num_hypotheses, field_dim),
            nn.GELU(),
            nn.Linear(field_dim, field_dim),
            nn.LayerNorm(field_dim)
        )

        # Coherence estimator
        # How coherent is the superposition?
        # Low coherence = very uncertain / ambiguous
        # High coherence = one interpretation dominates
        self.coherence_estimator = nn.Sequential(
            nn.Linear(num_hypotheses, 1),
            nn.Sigmoid()
        )

        total = sum(p.numel() for p in self.parameters())
        print(f"[SuperpositionCloud] {total:,} parameters")

    def forward(
        self,
        memory_state: torch.Tensor,
        uncertainty: torch.Tensor
    ) -> dict:
        """
        Generate superposition of interpretations.

        Args:
            memory_state: From ObserverCore [batch, field_dim]
            uncertainty: Total uncertainty [batch, 1]

        Returns:
            Superposition state
        """

        # Generate all hypotheses
        hypotheses = []
        for generator in self.hypothesis_generators:
            h = generator(memory_state)
            hypotheses.append(h)

        # Stack hypotheses [batch, num_hypotheses, hypothesis_dim]
        hypothesis_stack = torch.stack(hypotheses, dim=1)

        # Estimate amplitudes (uncertainty increases spread)
        amplitudes = self.amplitude_estimator(memory_state)

        # High uncertainty = more spread across hypotheses
        # Low uncertainty = one hypothesis dominates
        uncertainty_spread = uncertainty.expand_as(amplitudes)
        amplitudes = amplitudes * (1 - 0.5 * uncertainty_spread) + \
                     (1.0 / self.num_hypotheses) * 0.5 * uncertainty_spread

        # Weighted combination of hypotheses
        # [batch, num_hypotheses, 1] * [batch, num_hypotheses, hypothesis_dim]
        weighted = amplitudes.unsqueeze(-1) * hypothesis_stack

        # Flatten and fuse
        flat_weighted = weighted.reshape(memory_state.shape[0], -1)
        superposition_state = self.fusion(flat_weighted)

        # Estimate coherence (how certain are we?)
        coherence = self.coherence_estimator(amplitudes)

        # Entropy of amplitudes = how spread out are interpretations?
        entropy = -(amplitudes * torch.log(amplitudes + 1e-10)).sum(-1, keepdim=True)
        max_entropy = torch.log(torch.tensor(float(self.num_hypotheses)))
        normalized_entropy = entropy / max_entropy

        return {
            'superposition_state': superposition_state,
            'amplitudes': amplitudes,
            'hypotheses': hypothesis_stack,
            'coherence': coherence,
            'entropy': normalized_entropy,
            'dominant_hypothesis': amplitudes.argmax(dim=-1)
        }


class CollapseDecision(nn.Module):
    """
    Late-binding answer resolution.

    Never collapses the superposition prematurely.
    Waits for sufficient evidence.
    Learns from feedback to improve future decisions.

    Total: ~5M parameters
    """

    def __init__(self, field_dim: int = 256, output_dim: int = 256):
        super().__init__()

        self.field_dim = field_dim
        self.output_dim = output_dim

        # Collapse readiness estimator
        # Are we ready to collapse? Do we have enough evidence?
        self.readiness_estimator = nn.Sequential(
            nn.Linear(field_dim + 1 + 1, field_dim // 2),
            # state + coherence + uncertainty
            nn.GELU(),
            nn.Linear(field_dim // 2, 1),
            nn.Sigmoid()
        )

        # Decision maker - the actual collapse
        self.decision_maker = nn.Sequential(
            nn.Linear(field_dim, field_dim),
            nn.GELU(),
            nn.Linear(field_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        # Confidence estimator for the decision
        self.confidence_estimator = nn.Sequential(
            nn.Linear(field_dim + 1, field_dim // 2),
            nn.GELU(),
            nn.Linear(field_dim // 2, 1),
            nn.Sigmoid()
        )

        # Feedback integration
        # Learns from past decisions
        self.feedback_integrator = nn.GRUCell(output_dim, field_dim)
        self.feedback_state = nn.Parameter(torch.zeros(field_dim))

        # Decision quality tracker
        self.register_buffer('decision_quality_history', torch.zeros(50))
        self.decision_count = 0

        total = sum(p.numel() for p in self.parameters())
        print(f"[CollapseDecision] {total:,} parameters")

    def forward(
        self,
        superposition_output: dict,
        curiosity_output: dict,
        force_collapse: bool = False
    ) -> dict:
        """
        Decide whether to collapse superposition and generate output.

        Args:
            superposition_output: From SuperpositionCloud
            curiosity_output: From CuriosityEngine
            force_collapse: Force a decision regardless of readiness

        Returns:
            Decision output
        """
        state = superposition_output['superposition_state']
        coherence = superposition_output['coherence']
        uncertainty = 1.0 - coherence  # Low coherence = high uncertainty

        # Check if we should collapse
        readiness_input = torch.cat([state, coherence, uncertainty], dim=-1)
        readiness = self.readiness_estimator(readiness_input)

        # Collapse if ready or forced
        should_collapse = (readiness > 0.6) | force_collapse

        # Generate decision
        decision = self.decision_maker(state)

        # Estimate confidence in decision
        confidence_input = torch.cat([state, coherence], dim=-1)
        confidence = self.confidence_estimator(confidence_input)

        # Integrate feedback from past decisions
        batch_size = state.shape[0]
        feedback_state_expanded = self.feedback_state.unsqueeze(0).expand(batch_size, -1)
        updated_feedback = self.feedback_integrator(
            decision.detach(),
            feedback_state_expanded
        )
        self.feedback_state.data = updated_feedback.mean(0).detach()

        self.decision_count += 1

        return {
            'decision': decision,
            'confidence': confidence,
            'readiness': readiness,
            'should_collapse': should_collapse,
            'collapsed': should_collapse.any().item(),
            'decision_count': self.decision_count
        }

    def receive_feedback(self, feedback_signal: torch.Tensor):
        """
        Receive feedback on past decision quality.
        This is how CollapseDecision improves over time.
        """
        quality = feedback_signal.mean().item()
        idx = self.decision_count % 50
        self.decision_quality_history[idx] = quality

    def get_decision_quality(self) -> float:
        """Get recent decision quality."""
        recent = self.decision_quality_history[
            :min(self.decision_count, 50)
        ]
        return recent.mean().item() if len(recent) > 0 else 0.0
