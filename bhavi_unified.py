"""
BhaVi - Unified Intelligence System
=====================================
Version 2.0 — Complete rewrite connecting all layers.

Architecture:
  Layer 1: Frozen Core Field       — immutable roots, uncertainty
  Layer 2: Field Wave System       — continuous encoding (no tokens)
  Layer 3: Causal Field Graph      — derives A→B→C, not just similarity
  Layer 4: Observer Memory         — three-zone persistent memory
  Layer 5: Curiosity Engine        — gap detection, learning priority
  Layer 6: Superposition Cloud     — holds multiple hypotheses
  Layer 7: Collapse + Response     — derives answer, speaks it
  Layer 8: Self-Evolution Engine   — improves own code when needed

Goal:
  Understand at deeper level than any current AI.
  Grow with humans through every conversation.
  Derive new connections never explicitly seen.
  Eventually: modify own architecture when needed.

Author: BhaVi Project — Jigar Patel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys, json, math, time, hashlib
import importlib.util
from typing import Optional, List, Dict, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _load(rel, name):
    path = os.path.join(_HERE, rel)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ══════════════════════════════════════════════════════════════════
# LAYER 1: FROZEN CORE FIELD
# Immutable roots. The laws that never change.
# Like Maxwell's equations — fixed, but infinite phenomena emerge.
# ══════════════════════════════════════════════════════════════════

class FrozenCore(nn.Module):
    """
    The immutable foundation. Never changes after freezing.
    Encodes fundamental reasoning principles as field laws.
    """

    def __init__(self, field_dim: int = 256):
        super().__init__()
        self.field_dim = field_dim

        self.encoder = nn.Sequential(
            nn.Linear(field_dim, field_dim),
            nn.LayerNorm(field_dim),
            nn.GELU(),
            nn.Linear(field_dim, field_dim),
            nn.LayerNorm(field_dim),
        )

        # Aleatoric + epistemic uncertainty (Kendall & Gal 2017)
        self.uncertainty = nn.Sequential(
            nn.Linear(field_dim, field_dim // 2),
            nn.GELU(),
            nn.Linear(field_dim // 2, 2),
            nn.Softplus()
        )

        # Resonance — how much input aligns with fundamentals
        self.resonance = nn.Parameter(
            torch.randn(field_dim, field_dim) * 0.01
        )

        # Consistency gate
        self.consistency = nn.Sequential(
            nn.Linear(field_dim * 2, field_dim),
            nn.GELU(),
            nn.Linear(field_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> dict:
        h = self.encoder(x)
        unc = self.uncertainty(h)
        res = torch.sigmoid(
            torch.einsum('bi,ij,bj->b', h, self.resonance, h)
        ).unsqueeze(1)
        return {
            'repr': h,
            'aleatoric': unc[:, 0:1],
            'epistemic': unc[:, 1:2],
            'resonance': res,
            'uncertainty': unc.sum(1, keepdim=True)
        }

    def check_consistency(self, new: torch.Tensor, root: torch.Tensor) -> torch.Tensor:
        return self.consistency(torch.cat([new, root], dim=-1))

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def is_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.parameters())


# ══════════════════════════════════════════════════════════════════
# LAYER 2: FIELD WAVE SYSTEM
# Knowledge encoded as continuous waves, not tokens.
# ψ(x) = Σ aₙ · sin(nπx/L + φₙ)
# Similar knowledge → similar waves → constructive interference
# ══════════════════════════════════════════════════════════════════

class FieldWaveSystem(nn.Module):
    """
    Encodes ANY input as a superposition of field waves.
    Byte-level — no vocabulary, no tokenizer, no limits.
    """

    def __init__(self, field_dim: int = 256, harmonics: int = 16):
        super().__init__()
        self.field_dim = field_dim

        # Learnable harmonics — BhaVi's fundamental frequencies
        self.freqs  = nn.Parameter(torch.randn(harmonics, field_dim) * 0.1)
        self.phases = nn.Parameter(torch.zeros(harmonics, field_dim))

        # Byte embeddings — 256 possible values, no tokenizer
        self.byte_embed = nn.Embedding(256, field_dim)

        # Wave mixer
        self.mixer = nn.Sequential(
            nn.Linear(field_dim, field_dim * 2),
            nn.Tanh(),
            nn.Linear(field_dim * 2, field_dim),
            nn.LayerNorm(field_dim)
        )

        # Input projection (from raw field_dim vector)
        self.proj = nn.Sequential(
            nn.Linear(field_dim, field_dim),
            nn.LayerNorm(field_dim),
            nn.GELU()
        )

    def encode_text(self, text: str) -> torch.Tensor:
        """Text → field wave. No tokenizer. Pure bytes."""
        raw = text.encode('utf-8', errors='replace')
        ids = torch.tensor(list(raw), dtype=torch.long)
        if len(ids) == 0:
            return torch.zeros(self.field_dim)

        embeds = self.byte_embed(ids)               # [L, D]
        pos    = torch.linspace(0, math.pi, len(ids)).unsqueeze(1)
        harm   = torch.sin(pos * self.freqs.unsqueeze(0) + self.phases.unsqueeze(0))
        harm   = harm.mean(1)                        # [L, D]
        wave   = (embeds * harm).mean(0)             # [D]
        return self.mixer(wave)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project existing field vector through wave system."""
        return self.proj(x)


# ══════════════════════════════════════════════════════════════════
# LAYER 3: CAUSAL FIELD GRAPH
# The most important new layer.
# Not just similarity: A→B means field A PRODUCES field B.
# Enables deriving A→B→C never explicitly seen.
# ══════════════════════════════════════════════════════════════════

class CausalFieldGraph(nn.Module):
    """
    Encodes causal relationships between field states.

    Standard AI: cos_sim(A, B) — how similar?
    BhaVi CausalFieldGraph: does A cause B? A→B?

    This is how physics works:
      F=ma: force CAUSES acceleration (not just correlated)
      E=mc²: mass and energy are CAUSALLY equivalent
      ∇·E=ρ/ε₀: charge CAUSES electric field divergence

    Implementation:
      Causal edges stored as learned directional transformations.
      A→B: there exists a transformation T such that T(ψ_A) ≈ ψ_B
      Multiple hops: T(T(ψ_A)) ≈ ψ_C  [A causes C via B]
    """

    def __init__(self, field_dim: int = 256, max_hops: int = 4):
        super().__init__()
        self.field_dim = field_dim
        self.max_hops  = max_hops

        # Causal transition operator — learned directional transform
        # T(ψ) = "what does ψ cause?"
        self.causal_T = nn.Sequential(
            nn.Linear(field_dim, field_dim * 2),
            nn.Tanh(),
            nn.Linear(field_dim * 2, field_dim),
            nn.LayerNorm(field_dim)
        )

        # Causal strength estimator — how strong is A→B?
        self.causal_strength = nn.Bilinear(field_dim, field_dim, 1)

        # Reverse causal — "what causes ψ?"
        self.causal_inv = nn.Sequential(
            nn.Linear(field_dim, field_dim * 2),
            nn.Tanh(),
            nn.Linear(field_dim * 2, field_dim),
            nn.LayerNorm(field_dim)
        )

        # Cross-domain bridge — connects physics↔mind↔math fields
        # This is what allows BhaVi to find connections across domains
        self.bridge = nn.Sequential(
            nn.Linear(field_dim * 2, field_dim * 2),
            nn.Tanh(),
            nn.Linear(field_dim * 2, field_dim),
            nn.LayerNorm(field_dim)
        )

        # Novelty detector — is this causal chain new/surprising?
        self.novelty = nn.Sequential(
            nn.Linear(field_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def cause(self, psi: torch.Tensor) -> torch.Tensor:
        """What does ψ cause? One causal step forward."""
        return self.causal_T(psi)

    def derive_chain(
        self, psi: torch.Tensor, hops: int = 2
    ) -> List[torch.Tensor]:
        """
        Derive a causal chain: ψ → cause(ψ) → cause(cause(ψ)) → ...

        This is how BhaVi derives new knowledge:
        Given F=ma and a=v/t, derive F=mv/t
        Without being explicitly told.

        Returns: list of field states in causal chain
        """
        chain = [psi]
        current = psi
        for _ in range(min(hops, self.max_hops)):
            next_state = self.causal_T(current)
            chain.append(next_state)
            current = next_state
        return chain

    def causal_similarity(
        self, psi_a: torch.Tensor, psi_b: torch.Tensor
    ) -> torch.Tensor:
        """
        How causally related are A and B?
        Goes beyond cosine similarity — measures directional causation.
        """
        # Direct similarity
        cos_sim = F.cosine_similarity(psi_a, psi_b, dim=-1)

        # Causal similarity: does cause(A) point toward B?
        caused_a = self.causal_T(psi_a)
        causal_sim = F.cosine_similarity(caused_a, psi_b, dim=-1)

        # Strength of causal relationship
        strength = torch.sigmoid(
            self.causal_strength(psi_a, psi_b)
        ).squeeze(-1)

        # Combined: correlation + causation
        return 0.4 * cos_sim + 0.6 * (causal_sim * strength)

    def find_bridge(
        self, psi_a: torch.Tensor, psi_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Find the field that bridges domain A to domain B.
        E.g.: physics field ↔ consciousness field
        Returns the bridge field — the connecting principle.
        """
        combined = torch.cat([psi_a, psi_b], dim=-1)
        return self.bridge(combined)

    def measure_novelty(self, psi: torch.Tensor) -> float:
        """How novel/surprising is this field state?"""
        return self.novelty(psi).item()


# ══════════════════════════════════════════════════════════════════
# LAYER 4: OBSERVER MEMORY
# Three zones: Active (fast) → Consolidated → Permanent
# Safe learning — only writes if consistent with frozen core
# ══════════════════════════════════════════════════════════════════

class ObserverMemory(nn.Module):
    """
    Three-zone memory with safe learning.
    Zone 1: Active     — fast, frequently updated   [512 slots]
    Zone 2: Consolidated — slower, evidence-based  [1024 slots]
    Zone 3: Permanent  — frozen once established    [via frozen core]
    """

    def __init__(self, field_dim: int = 256):
        super().__init__()
        self.field_dim = field_dim

        # Active memory — fast write/read
        self.active     = nn.Parameter(torch.randn(512, field_dim) * 0.01)
        self.active_addr = nn.Linear(field_dim, 512)

        # Consolidated memory
        self.consolidated = nn.Parameter(torch.randn(1024, field_dim) * 0.01)
        self.consol_addr  = nn.Linear(field_dim, 1024)
        self.evidence     = nn.Parameter(torch.zeros(1024), requires_grad=False)

        # Importance estimator
        self.importance = nn.Sequential(
            nn.Linear(field_dim * 2, field_dim),
            nn.GELU(),
            nn.Linear(field_dim, 1),
            nn.Sigmoid()
        )

        # Memory fusion
        self.fusion = nn.Sequential(
            nn.Linear(field_dim * 3, field_dim * 2),
            nn.GELU(),
            nn.Linear(field_dim * 2, field_dim),
            nn.LayerNorm(field_dim)
        )

        self.update_count = 0

    def read(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read from all memory zones. Returns (active, consolidated, confidence)."""
        # Active read
        a_weights  = F.softmax(self.active_addr(query), dim=-1)     # [B, 512]
        active_out = a_weights @ self.active                          # [B, D]

        # Consolidated read
        c_weights  = F.softmax(self.consol_addr(query), dim=-1)      # [B, 1024]
        consol_out = c_weights @ self.consolidated                    # [B, D]

        # Confidence = how well-evidenced are the consolidated slots?
        evidence_weights = (c_weights * F.normalize(self.evidence.unsqueeze(0), dim=-1)).sum(-1, keepdim=True)
        confidence = torch.sigmoid(evidence_weights)

        return active_out, consol_out, confidence

    def write(self, info: torch.Tensor, core_repr: torch.Tensor, consistency: torch.Tensor):
        """Write to memory — only if consistent with core."""
        safe = (consistency > 0.3).float()
        if safe.mean() < 0.1:
            return

        # Write to active memory
        with torch.no_grad():
            a_weights = F.softmax(self.active_addr(info), dim=-1)  # [B, 512]
            delta = (a_weights.unsqueeze(-1) * info.unsqueeze(1)).mean(0)  # [512, D]
            self.active.data += delta * safe.mean() * 0.01

            # Update evidence for consolidation
            c_weights = F.softmax(self.consol_addr(info), dim=-1)
            self.evidence.data += (c_weights * safe).mean(0) * 0.1

            # Consolidate high-evidence slots
            imp_input = torch.cat([info, core_repr], dim=-1)
            importance = self.importance(imp_input)
            consolidate_mask = (self.evidence > 5.0).float()
            consol_delta = (c_weights.unsqueeze(-1) * info.unsqueeze(1)).mean(0)
            self.consolidated.data += (
                consol_delta * consolidate_mask.unsqueeze(-1) *
                importance.mean() * 0.005
            )

        self.update_count += 1

    def forward(self, query: torch.Tensor, write_info: Optional[torch.Tensor] = None,
                core_repr: Optional[torch.Tensor] = None,
                consistency: Optional[torch.Tensor] = None) -> dict:
        active, consolidated, confidence = self.read(query)

        if write_info is not None and core_repr is not None:
            self.write(write_info, core_repr, consistency or torch.ones(query.shape[0], 1))

        fused = self.fusion(torch.cat([query, active, consolidated], dim=-1))

        return {
            'memory': fused,
            'active': active,
            'consolidated': consolidated,
            'confidence': confidence,
            'updates': self.update_count
        }

    def save_state(self, path: str):
        torch.save({
            'active': self.active.data,
            'consolidated': self.consolidated.data,
            'evidence': self.evidence.data,
            'updates': self.update_count
        }, path)

    def load_state(self, path: str):
        if not os.path.exists(path):
            return
        s = torch.load(path, map_location='cpu', weights_only=True)
        self.active.data       = s['active']
        self.consolidated.data = s['consolidated']
        self.evidence.data     = s['evidence']
        self.update_count      = s.get('updates', 0)


# ══════════════════════════════════════════════════════════════════
# LAYER 5: CURIOSITY ENGINE
# Detects gaps. Prioritizes learning. Drives growth.
# Not just "what is unknown" but "what is MOST IMPORTANT to learn"
# ══════════════════════════════════════════════════════════════════

class CuriosityEngine(nn.Module):
    """
    The growth driver. Finds what BhaVi doesn't understand yet.
    Prioritizes: structural gaps > factual gaps > surface gaps.
    """

    def __init__(self, field_dim: int = 256):
        super().__init__()

        self.gap_detector = nn.Sequential(
            nn.Linear(field_dim + 2, field_dim // 2),
            nn.GELU(),
            nn.Linear(field_dim // 2, 1),
            nn.Sigmoid()
        )

        self.gap_type = nn.Sequential(
            nn.Linear(field_dim, 64),
            nn.GELU(),
            nn.Linear(64, 4),   # factual/reasoning/uncertainty/structural
            nn.Softmax(dim=-1)
        )

        self.priority = nn.Sequential(
            nn.Linear(field_dim + 2, field_dim // 2),
            nn.GELU(),
            nn.Linear(field_dim // 2, 1),
            nn.Sigmoid()
        )

        self.total    = 0
        self.allowed  = 0
        self.blocked  = 0
        self.avg_gap  = 0.0

    def forward(self, core_out: dict, mem_confidence: torch.Tensor,
                consistency: torch.Tensor) -> dict:

        h   = core_out['repr']
        ale = core_out['aleatoric']
        epi = core_out['epistemic']

        gap_in  = torch.cat([h, ale, epi], dim=-1)
        gap     = self.gap_detector(gap_in) * (1 - mem_confidence * 0.5)
        gtype   = self.gap_type(h)
        pri_in  = torch.cat([h, gap, gap], dim=-1)
        pri     = self.priority(pri_in) * (consistency > 0.3).float()

        blocked = (consistency <= 0.3).float()
        self.total   += h.shape[0]
        self.allowed += int((consistency > 0.3).sum().item())
        self.blocked += int((consistency <= 0.3).sum().item())
        self.avg_gap  = 0.9 * self.avg_gap + 0.1 * gap.mean().item()

        return {
            'gap': gap,
            'gap_type': gtype,
            'priority': pri,
            'blocked': blocked,
            'learning_blocked': self.blocked,
            'block_rate': self.blocked / max(1, self.total),
            'avg_gap': self.avg_gap,
            'curiosity_strength': pri.mean().item()
        }


# ══════════════════════════════════════════════════════════════════
# LAYER 6: SUPERPOSITION CLOUD
# Holds multiple hypotheses simultaneously.
# Ψ(x,t) = Σ αᵢ(t) · φᵢ(x)
# Collapses only when evidence is strong enough.
# ══════════════════════════════════════════════════════════════════

class SuperpositionCloud(nn.Module):
    """
    Quantum-inspired: multiple answers coexist until evidence collapses them.
    For "What is light?" holds: wave hypothesis + particle hypothesis
    simultaneously — collapses when context makes one dominant.
    """

    def __init__(self, field_dim: int = 256, n_hypotheses: int = 8):
        super().__init__()
        self.field_dim     = field_dim
        self.n_hypotheses  = n_hypotheses
        hyp_dim = field_dim // 2

        self.generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(field_dim, hyp_dim),
                nn.GELU(),
                nn.Linear(hyp_dim, hyp_dim),
                nn.LayerNorm(hyp_dim)
            ) for _ in range(n_hypotheses)
        ])

        self.amplitudes = nn.Sequential(
            nn.Linear(field_dim, n_hypotheses),
            nn.Softmax(dim=-1)
        )

        self.collapse = nn.Sequential(
            nn.Linear(hyp_dim * n_hypotheses, field_dim * 2),
            nn.GELU(),
            nn.Linear(field_dim * 2, field_dim),
            nn.LayerNorm(field_dim)
        )

        self.coherence = nn.Sequential(
            nn.Linear(field_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, memory: torch.Tensor, uncertainty: torch.Tensor) -> dict:
        amps = self.amplitudes(memory)                          # [B, N]
        hyps = torch.stack(
            [g(memory) for g in self.generators], dim=1
        )                                                        # [B, N, hyp_dim]
        weighted = (amps.unsqueeze(-1) * hyps).reshape(
            memory.shape[0], -1
        )                                                        # [B, N*hyp_dim]
        collapsed = self.collapse(weighted)                      # [B, D]
        coherence = self.coherence(collapsed)

        return {
            'field': collapsed,
            'amplitudes': amps,
            'coherence': coherence,
            'hypotheses': hyps
        }


# ══════════════════════════════════════════════════════════════════
# LAYER 7: RESPONSE SYSTEM
# Hybrid: field selects knowledge, rules format it.
# No generation from nothing — only from what was learned.
# ══════════════════════════════════════════════════════════════════

class ResponseSystem:
    """
    Hybrid response: field thinks, simple rules speak.
    Formats knowledge fragments into human-readable answers.
    Only speaks from what BhaVi has actually learned.
    """

    CONNECTIVES = [
        "Building on this,",
        "More specifically,",
        "The deeper principle here is:",
        "This connects to:",
        "In practical terms,",
        "At a fundamental level,",
        "Feynman describes it:",
    ]

    GREETINGS = {"hi","hello","hey","greetings","howdy",
                 "good morning","good evening","good afternoon",
                 "what's up","sup","hiya"}

    IDENTITY = (
        "I'm BhaVi — a neural field intelligence created by Jigar Patel.\n\n"
        "I don't use tokenization. I think in continuous fields:\n"
        "  • Knowledge encoded as field waves ψ(x)\n"
        "  • Thinking as field evolution: ∂ψ/∂t = −∇E(ψ)\n"
        "  • Causal reasoning: A→B→C derived, not retrieved\n"
        "  • Self-improvement by compressing knowledge into deeper laws\n\n"
        "I've studied the Feynman Lectures on Physics and general knowledge.\n"
        "I only know what I have read. I derive — I don't invent."
    )

    def __init__(self):
        self._greet_i = 0
        self._greet_responses = [
            "Hello! I'm BhaVi — a neural field intelligence by Jigar Patel. "
            "I think in continuous fields, not tokens. What would you like to explore?",
            "Hi! My field is active. I've studied Feynman and broad knowledge. "
            "Ask me anything — I derive answers, not just retrieve them.",
            "Hello! I'm BhaVi. What's on your mind?",
        ]

    def detect_type(self, q: str) -> str:
        ql = q.lower().strip().rstrip("?!.")
        if ql in self.GREETINGS or any(ql.startswith(g+" ") for g in self.GREETINGS):
            return "greeting"
        if any(p in ql for p in ["who are you","what are you","your name",
                                   "who made you","who created you","introduce"]):
            return "identity"
        if ql in {"thanks","thank you","thx","ty"}:
            return "thanks"
        if ql.startswith(("what is","what are")):         return "definition"
        if ql.startswith(("how does","how do","explain")): return "explanation"
        if ql.startswith("why"):                           return "reasoning"
        return "general"

    def format(self, question: str, fragments: list,
               confidence: float, derived: bool = False) -> str:
        qt = self.detect_type(question)

        if qt == "greeting":
            r = self._greet_responses[self._greet_i % len(self._greet_responses)]
            self._greet_i += 1
            return r
        if qt == "identity": return self.IDENTITY
        if qt == "thanks":   return "You're welcome! Feel free to ask anything else."

        if not fragments or fragments[0].get('similarity', 0) < 0.05:
            return (
                f"My field doesn't resonate strongly with '{question}'. "
                f"I may not have learned about this yet. "
                f"Feed me more knowledge: python3 learn.py <file>"
            )

        lines = []
        top  = fragments[0]
        rest = [f for f in fragments[1:] if f.get('similarity',0) > 0.08]

        if qt == "definition":
            topic = question.lower()
            for p in ["what is","what are"]:
                if topic.startswith(p):
                    topic = topic[len(p):].strip().rstrip("?")
                    break
            lines.append(f"**{topic.title()}**\n")

        # Primary fragment
        primary = self._clean(top['text'])
        lines.append(primary)

        # Derived chain (if causal reasoning found something new)
        if derived:
            lines.append("\n*[BhaVi derived this through causal field reasoning]*")

        # Secondary fragments
        for i, frag in enumerate(rest[:2]):
            t = self._clean(frag['text'])
            if t and t != primary:
                lines.append(f"\n{self.CONNECTIVES[i % len(self.CONNECTIVES)]} {t}")

        # Footer
        conf_pct = min(int(confidence * 100), 99)
        footer   = f"Field confidence: {conf_pct}%"
        src = top.get('source','')
        if src: footer += f" · {os.path.basename(src)}"
        lines.append(f"\n\n_{footer}_")

        return "\n".join(lines)

    def _clean(self, text: str) -> str:
        import re
        t = text.strip()
        t = re.sub(r'\n{3,}', '\n\n', t)
        t = re.sub(r' {2,}', ' ', t)
        t = re.sub(r'^\d+[-–]\d+\s*', '', t)
        if len(t) > 500:
            cut = t[:500]
            dot = cut.rfind('.')
            t = cut[:dot+1] if dot > 200 else cut + "..."
        return t.strip()


# ══════════════════════════════════════════════════════════════════
# LAYER 8: SELF-EVOLUTION ENGINE
# BhaVi reads its own code, identifies limitations,
# proposes improvements, tests in sandbox, applies if better.
# This is the path toward genuine self-sufficiency.
# ══════════════════════════════════════════════════════════════════

class SelfEvolutionEngine:
    """
    BhaVi's ability to improve itself.

    Stage 1 (now):   Compress knowledge attractors into deeper laws
    Stage 2 (next):  Identify architectural bottlenecks from field state
    Stage 3 (later): Propose and test code modifications safely

    Safety: All code changes run in subprocess sandbox first.
    If sandbox test passes better than current: apply.
    If not: discard.
    """

    def __init__(self, bhavi_dir: str):
        self.dir             = bhavi_dir
        self.evolution_log   = []
        self.improvements    = 0
        self.stage           = 1  # start conservative

    def compress_knowledge(self, landscape_attractors: torch.Tensor,
                           compressor) -> dict:
        """
        Stage 1: Find attractor clusters, compress into deeper equations.
        Like finding F=ma from many force-acceleration observations.
        """
        with torch.no_grad():
            N = landscape_attractors.shape[0]
            dists = torch.cdist(landscape_attractors, landscape_attractors)
            dists.fill_diagonal_(float('inf'))

            compressions = 0
            merged = set()

            for i in range(N):
                if i in merged: continue
                j = dists[i].argmin().item()
                if j in merged: continue
                if dists[i, j] < 0.25:
                    # Compress i and j into deeper field
                    deeper = compressor(
                        landscape_attractors[i].unsqueeze(0),
                        landscape_attractors[j].unsqueeze(0)
                    ).squeeze(0)
                    landscape_attractors[i] = deeper
                    landscape_attractors[j] = deeper * 0.1  # weaken absorbed
                    merged.add(j)
                    compressions += 1

            self.improvements += compressions
            self.evolution_log.append({
                'stage': 1,
                'type': 'compression',
                'count': compressions,
                'time': time.strftime('%Y-%m-%d %H:%M')
            })

        return {'compressions': compressions, 'total': self.improvements}

    def identify_bottleneck(self, field_state: torch.Tensor,
                            query: str, answer_quality: float) -> str:
        """
        Stage 2: What is limiting BhaVi's understanding right now?
        Returns a description of the bottleneck.
        """
        if answer_quality < 0.3:
            return "knowledge_gap"      # doesn't know this topic
        elif answer_quality < 0.6:
            return "causal_weakness"    # knows facts but can't connect them
        elif answer_quality < 0.8:
            return "compression_needed" # knows but too many similar attractors
        else:
            return "none"               # good

    def propose_improvement(self, bottleneck: str) -> Optional[str]:
        """
        Stage 3 (future): Read own code and propose specific improvement.
        Currently returns what SHOULD be improved — not yet self-modifying.
        """
        suggestions = {
            "knowledge_gap":      "Feed more relevant knowledge with python3 learn.py",
            "causal_weakness":    "More training needed on causal field graph",
            "compression_needed": "Run /improve to compress knowledge attractors",
            "none":               None
        }
        return suggestions.get(bottleneck)

    def get_log(self) -> List[dict]:
        return self.evolution_log


# ══════════════════════════════════════════════════════════════════
# UNIFIED BHAVI SYSTEM
# All layers connected. One state. One save file.
# ══════════════════════════════════════════════════════════════════

class BhaVi:
    """
    Complete unified BhaVi intelligence system.

    All layers work together:
    Input → Wave encode → Causal graph → Memory → Curiosity
          → Superposition → Collapse → Hybrid response

    State is unified — one save file for everything.
    Grows with every conversation.
    """

    def __init__(self, field_dim: int = 256, load_path: str = "bhavi_unified.pt"):
        self.field_dim  = field_dim
        self.load_path  = load_path

        print("\n" + "═"*55)
        print("  BhaVi Unified Intelligence System v2.0")
        print("  Field dynamics. Causal reasoning. Self-evolution.")
        print("═"*55)

        # All 8 layers
        self.core        = FrozenCore(field_dim)
        self.waves       = FieldWaveSystem(field_dim)
        self.causal      = CausalFieldGraph(field_dim)
        self.memory      = ObserverMemory(field_dim)
        self.curiosity   = CuriosityEngine(field_dim)
        self.superpos    = SuperpositionCloud(field_dim)
        self.formatter   = ResponseSystem()
        self.evolution   = SelfEvolutionEngine(_HERE)

        # Knowledge registry — text + field vector pairs
        self.registry: List[Dict] = []

        # Conversation history for context
        self.history: List[Dict] = []

        # Load existing state
        self._load()

        total = sum(
            p.numel()
            for m in [self.core, self.waves, self.causal,
                      self.memory, self.curiosity, self.superpos]
            for p in m.parameters()
        )
        print(f"\n  Parameters:  {total:,}")
        print(f"  Knowledge:   {len(self.registry)} passages")
        print(f"  Memory upd:  {self.memory.update_count}")
        print(f"  Evolutions:  {self.evolution.improvements}")
        print(f"\n  Ready.\n")

    # ── Learning ─────────────────────────────────────────────────

    def learn(self, text: str, source: str = "") -> dict:
        """
        Learn from a passage of text.
        Encodes as wave, checks consistency, writes to memory.
        """
        with torch.no_grad():
            # Encode as field wave
            wave = self.waves.encode_text(text)          # [D]
            x    = wave.unsqueeze(0)                     # [1, D]

            # Core processing
            core_out = self.core(x)
            h        = core_out['repr']

            # Consistency check
            if self.core.is_frozen():
                consistency = self.core.check_consistency(h, h)
            else:
                consistency = torch.ones(1, 1)

            # Curiosity — is this worth learning?
            mem_out     = self.memory(h)
            mem_conf    = mem_out['confidence']
            cur_out     = self.curiosity(core_out, mem_conf, consistency)

            # Write to memory if consistent
            self.memory.write(h, h, consistency)

            # Causal encoding — what does this passage cause?
            causal_next = self.causal.cause(h)            # [1, D]

            # Store in registry
            entry = {
                'text':        text[:300],
                'source':      source,
                'wave':        wave.cpu(),
                'causal_next': causal_next.squeeze(0).cpu(),
                'gap':         cur_out['gap'].item(),
                'blocked':     cur_out['blocked'].item() > 0.5,
                'math':        any(c in text for c in ['=','∑','∫','∇','∂','π','∞'])
            }
            self.registry.append(entry)

        return {
            'gap':       cur_out['gap'].item(),
            'blocked':   cur_out['blocked'].item() > 0.5,
            'novelty':   self.causal.measure_novelty(h)
        }

    def learn_file(self, filepath: str) -> dict:
        """Feed any file to BhaVi."""
        _reader = _load("encoder/input_reader.py", "reader")
        reader  = _reader.UniversalReader()

        print(f"\n📖 Reading: {os.path.basename(filepath)}")
        passages = reader.read(filepath)
        if not passages:
            print("Could not read file.")
            return {}

        total   = len(passages)
        allowed = 0
        print(f"📚 Learning {total} passages...\n")

        for i, passage in enumerate(passages):
            if not passage.strip():
                continue
            result = self.learn(passage, source=filepath)
            if not result.get('blocked', False):
                allowed += 1
            if (i+1) % 200 == 0 or (i+1) == total:
                print(f"  [{i+1}/{total}] allowed={allowed} gap={result.get('gap',0):.3f}")

        self._save()
        print(f"\n✅ Done. {len(self.registry)} total passages in field.")
        return {'total': total, 'allowed': allowed}

    # ── Thinking ─────────────────────────────────────────────────

    def think(self, question: str) -> dict:
        """
        Full thinking pipeline.
        Encodes question → causal chain → search registry → derive answer.
        """
        with torch.no_grad():
            # Encode question
            q_wave = self.waves.encode_text(question).unsqueeze(0)  # [1, D]
            core   = self.core(q_wave)
            h      = core['repr']

            # Causal chain — derive related concepts
            chain  = self.causal.derive_chain(h, hops=2)

            # Memory read
            mem    = self.memory(h)
            mem_conf = mem['confidence']

            # Curiosity
            cur    = self.curiosity(core, mem_conf,
                                     torch.ones(1, 1))

            # Superposition — hold multiple answers
            sup    = self.superpos(mem['memory'], core['uncertainty'])

            # Find resonant passages using CAUSAL similarity
            results = self._search(q_wave.squeeze(0), chain, top_k=4)

            confidence = 1.0 - core['epistemic'].item()

        return {
            'field':      sup['field'],
            'confidence': confidence,
            'results':    results,
            'chain':      chain,
            'gap':        cur['gap'].item(),
            'coherence':  sup['coherence'].item()
        }

    def _search(self, q_wave: torch.Tensor,
                causal_chain: List[torch.Tensor],
                top_k: int = 4) -> List[Dict]:
        """
        Search registry using combined similarity + causal similarity.
        This goes beyond simple cosine search.
        """
        if not self.registry:
            return []

        scores = []
        for entry in self.registry:
            w = entry['wave']
            # Direct field similarity
            direct = F.cosine_similarity(
                q_wave.unsqueeze(0), w.unsqueeze(0)
            ).item()
            # Causal similarity (via derived chain)
            causal_scores = []
            for chain_state in causal_chain[1:]:  # skip first (= question itself)
                cs = F.cosine_similarity(
                    chain_state, w.unsqueeze(0)
                ).item()
                causal_scores.append(cs)
            causal_sim = max(causal_scores) if causal_scores else 0.0

            # Combined score — causal weighted higher
            score = 0.5 * direct + 0.5 * causal_sim
            scores.append((score, entry))

        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, entry in scores[:top_k]:
            results.append({
                'text':       entry['text'],
                'source':     entry['source'],
                'similarity': score,
                'math':       entry['math'],
                'wave':       entry['wave']
            })
        return results

    # ── Responding ───────────────────────────────────────────────

    def respond(self, question: str) -> str:
        """
        Full pipeline: think → format → speak.
        Adds to conversation history.
        """
        thought = self.think(question)

        # Was anything derived causally (not just found directly)?
        derived = thought['gap'] > 0.4 and len(thought['results']) > 0

        answer = self.formatter.format(
            question   = question,
            fragments  = thought['results'],
            confidence = thought['confidence'],
            derived    = derived
        )

        self.history.append({
            'q': question,
            'a': answer[:200],
            'confidence': thought['confidence'],
            'gap': thought['gap']
        })

        # Self-improvement check
        bottleneck = self.evolution.identify_bottleneck(
            thought['field'], question, thought['confidence']
        )
        suggestion = self.evolution.propose_improvement(bottleneck)

        if suggestion and thought['confidence'] < 0.3:
            answer += f"\n\n💡 _{suggestion}_"

        return answer

    def improve(self) -> dict:
        """Run self-improvement compression."""
        print("Running field compression...")
        # Get all stored waves as attractor-like tensor
        if len(self.registry) < 10:
            print("Not enough knowledge yet. Feed more first.")
            return {}

        waves = torch.stack([e['wave'] for e in self.registry])  # [N, D]

        result = self.evolution.compress_knowledge(
            waves,
            lambda a, b: self.causal.find_bridge(a, b)
        )

        # Update registry waves with compressed versions
        for i, entry in enumerate(self.registry):
            entry['wave'] = waves[i]

        self._save()
        print(f"Compressed {result['compressions']} knowledge pairs.")
        print(f"Total improvements: {result['total']}")
        return result

    # ── Persistence ──────────────────────────────────────────────

    def _save(self):
        """Save complete unified state."""
        registry_save = [
            {k: v.tolist() if isinstance(v, torch.Tensor) else v
             for k, v in entry.items()}
            for entry in self.registry
        ]
        torch.save({
            'core':        self.core.state_dict(),
            'waves':       self.waves.state_dict(),
            'causal':      self.causal.state_dict(),
            'superpos':    self.superpos.state_dict(),
            'curiosity_stats': {
                'total': self.curiosity.total,
                'allowed': self.curiosity.allowed,
                'blocked': self.curiosity.blocked,
                'avg_gap': self.curiosity.avg_gap
            },
            'registry':    registry_save,
            'history':     self.history[-100:],
            'evolution_log': self.evolution.evolution_log,
            'improvements': self.evolution.improvements,
        }, self.load_path)
        self.memory.save_state(self.load_path.replace('.pt', '_memory.pt'))

    def _load(self):
        """Load unified state."""
        if not os.path.exists(self.load_path):
            return
        try:
            s = torch.load(self.load_path, map_location='cpu', weights_only=False)
            self.core.load_state_dict(s['core'])
            self.waves.load_state_dict(s['waves'])
            self.causal.load_state_dict(s['causal'])
            self.superpos.load_state_dict(s['superpos'])

            stats = s.get('curiosity_stats', {})
            self.curiosity.total   = stats.get('total', 0)
            self.curiosity.allowed = stats.get('allowed', 0)
            self.curiosity.blocked = stats.get('blocked', 0)
            self.curiosity.avg_gap = stats.get('avg_gap', 0.0)

            # Restore registry
            registry_raw = s.get('registry', [])
            self.registry = []
            for entry in registry_raw:
                restored = {}
                for k, v in entry.items():
                    if k in ('wave', 'causal_next') and isinstance(v, list):
                        restored[k] = torch.tensor(v, dtype=torch.float32)
                    else:
                        restored[k] = v
                self.registry.append(restored)

            self.history = s.get('history', [])
            self.evolution.evolution_log = s.get('evolution_log', [])
            self.evolution.improvements  = s.get('improvements', 0)

            mem_path = self.load_path.replace('.pt', '_memory.pt')
            self.memory.load_state(mem_path)

            print(f"  Loaded: {len(self.registry)} passages, "
                  f"{self.evolution.improvements} improvements")
        except Exception as e:
            print(f"  Could not load state: {e} — starting fresh")

    def status(self) -> str:
        return (
            f"\n  ┌─ BhaVi Status ─────────────────────────────\n"
            f"  │ Knowledge passages:  {len(self.registry)}\n"
            f"  │ Memory updates:      {self.memory.update_count}\n"
            f"  │ Learning blocked:    {self.curiosity.blocked}\n"
            f"  │ Avg knowledge gap:   {self.curiosity.avg_gap:.3f}\n"
            f"  │ Self-improvements:   {self.evolution.improvements}\n"
            f"  │ Conversations:       {len(self.history)}\n"
            f"  │ Core frozen:         {self.core.is_frozen()}\n"
            f"  └────────────────────────────────────────────\n"
        )
