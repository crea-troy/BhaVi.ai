"""
BhaVi - Knowledge Feeder
=========================
The main interface for teaching BhaVi from any source.

This connects:
  UniversalReader     (reads files)
        +
  NeuralFieldEncoder  (converts text to field)
        +
  BhaVi               (learns and stores)

Usage:
    feeder = KnowledgeFeeder(bhavi_model)
    
    # Feed a physics book
    feeder.feed_file("physics_textbook.pdf")
    
    # Feed direct text
    feeder.feed_text("Newton's law states F = ma")
    
    # Feed anything
    feeder.feed_text(\"\"\"
        E = mc² is Einstein's mass-energy equivalence.
        It means mass and energy are interchangeable.
        c is the speed of light: 3 × 10⁸ m/s
    \"\"\")
    
    # See what BhaVi learned
    feeder.learning_report()

Author: BhaVi Project
"""

import torch
import os
import sys
import importlib.util
import time
from typing import List, Optional

# ── Path setup ────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _import_from(rel_path, module_name):
    full_path = os.path.join(_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_reader_mod  = _import_from("encoder/input_reader.py", "input_reader")
_encoder_mod = _import_from("encoder/neural_field_encoder.py", "neural_field_encoder")

UniversalReader            = _reader_mod.UniversalReader
UniversalNeuralFieldEncoder = _encoder_mod.UniversalNeuralFieldEncoder


class LearningTracker:
    """Tracks what BhaVi has learned over time."""

    def __init__(self):
        self.sessions = []
        self.total_passages = 0
        self.total_learned = 0
        self.total_blocked = 0
        self.gap_history = []
        self.sources = []

    def record(self, passage: str, output: dict, source: str = ""):
        gap = output['gap_score'].mean().item()
        learned = output['should_learn'].float().mean().item()
        blocked = output['learning_blocked'].float().mean().item()
        math = output.get('math_detected', False)

        self.total_passages += 1
        self.total_learned += learned
        self.total_blocked += blocked
        self.gap_history.append(gap)

        if source and source not in self.sources:
            self.sources.append(source)

    def report(self):
        print("\n" + "="*55)
        print("  BhaVi Learning Report")
        print("="*55)
        print(f"  Sources fed:        {len(self.sources)}")
        for s in self.sources:
            print(f"    → {os.path.basename(s)}")
        print(f"\n  Total passages:     {self.total_passages}")
        print(f"  Learning allowed:   {self.total_learned:.0f} passages")
        print(f"  Learning blocked:   {self.total_blocked:.0f} passages")

        if self.gap_history:
            early  = sum(self.gap_history[:10]) / min(10, len(self.gap_history))
            recent = sum(self.gap_history[-10:]) / min(10, len(self.gap_history))
            reduction = (early - recent) / max(early, 1e-8) * 100
            print(f"\n  Initial knowledge gap:  {early:.4f}")
            print(f"  Current knowledge gap:  {recent:.4f}")
            print(f"  Gap reduction:          {reduction:.1f}%")
            if reduction > 5:
                print(f"  → BhaVi is learning well ✅")
            elif reduction > 0:
                print(f"  → BhaVi is starting to learn")
            else:
                print(f"  → Need more data or repetition")
        print("="*55 + "\n")


class KnowledgeFeeder:
    """
    Main interface for feeding knowledge to BhaVi.
    
    Handles the complete pipeline:
    File/Text → Reader → Encoder → BhaVi → Memory
    """

    def __init__(self, bhavi_model, field_dim: int = 256):
        self.bhavi = bhavi_model
        self.encoder = UniversalNeuralFieldEncoder(
            field_dim=field_dim,
            output_dim=field_dim
        )
        self.reader = UniversalReader()
        self.tracker = LearningTracker()

        print("[KnowledgeFeeder] Ready")
        print("[KnowledgeFeeder] Accepts: PDF, TXT, MD, direct text")
        print("[KnowledgeFeeder] No tokenization — pure field encoding\n")

    def feed_file(
        self,
        filepath: str,
        show_progress: bool = True,
        save_after: bool = True
    ):
        """
        Feed any file to BhaVi.
        
        Args:
            filepath:      Path to PDF, TXT, MD file
            show_progress: Print progress while learning
            save_after:    Save memory after feeding
        """
        print(f"\n{'='*55}")
        print(f"  Feeding: {os.path.basename(filepath)}")
        print(f"{'='*55}")

        # Read file into passages
        passages = self.reader.read(filepath)

        if not passages:
            print("[Feeder] No content extracted. Check file path.")
            return

        # Feed each passage to BhaVi
        self._feed_passages(passages, source=filepath, show_progress=show_progress)

        if save_after:
            self.bhavi.observer_core.save_memory(self.bhavi.memory_path)
            print(f"\n[Feeder] Memory saved ✅")

        self.tracker.report()

    def feed_text(
        self,
        text: str,
        source_name: str = "direct_input",
        show_progress: bool = False,
        save_after: bool = True
    ):
        """
        Feed any text directly to BhaVi.
        
        Args:
            text:         Any string — equations, paragraphs, mixed
            source_name:  Label for tracking
            show_progress: Print each passage
            save_after:   Save memory after feeding
        """
        passages = self.reader.read_text(text)

        if not passages:
            # Feed as single passage
            passages = [text.strip()]

        self._feed_passages(
            passages,
            source=source_name,
            show_progress=show_progress
        )

        if save_after:
            self.bhavi.observer_core.save_memory(self.bhavi.memory_path)

    def feed_folder(self, folder_path: str):
        """
        Feed all supported files in a folder.
        Great for feeding an entire library.
        """
        supported = ['.txt', '.pdf', '.md', '.tex']
        files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in supported
        ]

        if not files:
            print(f"[Feeder] No supported files in {folder_path}")
            return

        print(f"[Feeder] Found {len(files)} files to feed")

        for i, filepath in enumerate(files):
            print(f"\n[Feeder] File {i+1}/{len(files)}")
            self.feed_file(filepath, save_after=(i == len(files) - 1))

        self.tracker.report()

    def _feed_passages(
        self,
        passages: List[str],
        source: str = "",
        show_progress: bool = True
    ):
        """Internal method — feeds list of passages to BhaVi."""

        total = len(passages)
        start_time = time.time()

        for i, passage in enumerate(passages):
            if not passage.strip():
                continue

            # Encode passage to field vector
            with torch.no_grad():
                enc_result = self.encoder(passage)

            encoding = enc_result['encoding'].unsqueeze(0)  # [1, 256]
            math_detected = enc_result['math_detected']

            # Feed to BhaVi
            output = self.bhavi(encoding, learn=True, force_collapse=True)

            # Add math detection to output for tracking
            output['math_detected'] = math_detected

            # Track learning
            self.tracker.record(passage, output, source)

            # Progress display
            if show_progress and (i % 5 == 0 or i == total - 1):
                gap = output['gap_score'].mean().item()
                blocked = output['learning_blocked'].float().mean().item()
                elapsed = time.time() - start_time

                math_tag = "🔢" if math_detected else "📝"
                block_tag = "🛡️ " if blocked > 0.5 else "✅"

                print(
                    f"  [{i+1:>4}/{total}] {math_tag} {block_tag} "
                    f"gap={gap:.3f} | "
                    f"{elapsed:.1f}s | "
                    f"{passage[:50].strip()}..."
                    if len(passage) > 50
                    else f"  [{i+1:>4}/{total}] {math_tag} {block_tag} "
                         f"gap={gap:.3f} | {passage[:50]}"
                )

        elapsed = time.time() - start_time
        print(f"\n[Feeder] Done: {total} passages in {elapsed:.1f}s")
        print(f"[Feeder] Speed: {total/elapsed:.1f} passages/sec")

    def learning_report(self):
        """Print full learning report."""
        self.tracker.report()
        self.bhavi.status()

    def ask(self, question: str) -> dict:
        """
        Ask BhaVi a question using its learned knowledge.
        Encodes question and retrieves from memory.
        """
        print(f"\n[BhaVi] Question: {question}")

        with torch.no_grad():
            enc_result = self.encoder(question)
            encoding = enc_result['encoding'].unsqueeze(0)
            output = self.bhavi(encoding, learn=False, force_collapse=True)

        print(f"[BhaVi] Confidence:          {output['confidence'].mean().item():.3f}")
        print(f"[BhaVi] Epistemic gap:        {output['epistemic_uncertainty'].mean().item():.3f}")
        print(f"[BhaVi] Coherence:            {output['coherence'].mean().item():.3f}")
        print(f"[BhaVi] Memory confidence:    {output['memory_confidence'].mean().item():.3f}")

        return output
