"""
BhaVi - Chat Interface
=======================
Ask BhaVi questions and get answers from its learned knowledge.

Usage:
    python3 chat.py                    # interactive chat
    python3 chat.py --feed book.pdf    # feed a file first then chat

How it works:
    1. You type a question
    2. BhaVi encodes it as a neural field
    3. BhaVi searches its knowledge store for relevant passages
    4. BhaVi ranks them by field similarity
    5. BhaVi shows you the most relevant knowledge it has

Author: BhaVi Project
"""

import torch
import os
import sys
import json
import importlib.util
from typing import List

# ── Path setup ────────────────────────────────────────────────────
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


# Load all BhaVi modules
_bhavi   = _load("bhavi.py",                              "bhavi_main")
_enc     = _load("encoder/neural_field_encoder.py",       "nfe")
_reader  = _load("encoder/input_reader.py",               "reader")
_store   = _load("encoder/knowledge_store.py",            "kstore")

BhaVi                      = _bhavi.BhaVi
UniversalNeuralFieldEncoder = _enc.UniversalNeuralFieldEncoder
UniversalReader             = _reader.UniversalReader
KnowledgeStore              = _store.KnowledgeStore


# ── Knowledge Feeder (inline, no separate file needed) ────────────
class BhaViChat:
    """
    Complete BhaVi chat system.

    Feed knowledge → ask questions → get answers.
    All knowledge persists across sessions.
    """

    def __init__(self):
        print("\n" + "="*55)
        print("  Initializing BhaVi Chat")
        print("="*55)

        # Core model
        self.bhavi = BhaVi(memory_path="bhavi_memory.pt")

        # Encoder — converts text to field
        self.encoder = UniversalNeuralFieldEncoder(
            field_dim=256, output_dim=256
        )

        # Reader — reads files
        self.reader = UniversalReader()

        # Knowledge store — stores passages for retrieval
        self.store = KnowledgeStore(store_path="bhavi_knowledge")

        print(f"\n[Chat] Knowledge base: {len(self.store)} passages stored")
        print(f"[Chat] Ready.\n")

    # ── Feeding Knowledge ─────────────────────────────────────────

    def feed_file(self, filepath: str):
        """Feed any file (PDF, TXT, MD) to BhaVi."""
        print(f"\n📖 Reading: {os.path.basename(filepath)}")
        passages = self.reader.read(filepath)
        if not passages:
            print("Could not read file. Check path and format.")
            return
        self._feed_passages(passages, source=filepath)
        self.store.save()
        self.bhavi.observer_core.save_memory("bhavi_memory.pt")
        print(f"\n✅ Done. BhaVi now knows {len(self.store)} passages total.")

    def feed_text(self, text: str, source: str = "manual"):
        """Feed any text directly to BhaVi."""
        passages = self.reader.read_text(text)
        if not passages:
            passages = [text.strip()]
        self._feed_passages(passages, source=source)
        self.store.save()
        self.bhavi.observer_core.save_memory("bhavi_memory.pt")

    def _feed_passages(self, passages: List[str], source: str = ""):
        total = len(passages)
        print(f"📚 Teaching BhaVi {total} passages...\n")

        for i, passage in enumerate(passages):
            if not passage.strip():
                continue

            # Encode to field
            with torch.no_grad():
                enc_result = self.encoder(passage)

            encoding = enc_result['encoding']
            math     = enc_result['math_detected']

            # BhaVi processes and learns
            output = self.bhavi(
                encoding.unsqueeze(0),
                learn=True,
                force_collapse=True
            )

            gap        = output['gap_score'].mean().item()
            importance = output['confidence'].mean().item()
            blocked    = output['learning_blocked'].float().mean().item()

            # Store in knowledge base
            self.store.add(
                text         = passage.strip(),
                encoding     = encoding,
                source       = source,
                math_detected= math,
                importance   = importance,
                gap          = gap
            )

            # Progress
            tag = "🔢" if math else "📝"
            blk = "🛡️ " if blocked > 0.5 else "✅"
            preview = passage.strip()[:60].replace('\n', ' ')
            print(f"  {tag}{blk} [{i+1}/{total}] gap={gap:.3f} | {preview}...")

    # ── Asking Questions ──────────────────────────────────────────

    def ask(self, question: str, top_k: int = 3) -> str:
        """
        Ask BhaVi a question.
        Returns a formatted answer from its knowledge.
        """
        if len(self.store) == 0:
            return (
                "I have no knowledge yet. "
                "Feed me some text first using /feed or /learn."
            )

        # Encode question as field
        with torch.no_grad():
            enc_result = self.encoder(question)
            q_encoding = enc_result['encoding']

            # BhaVi processes question (no learning)
            output = self.bhavi(
                q_encoding.unsqueeze(0),
                learn=False,
                force_collapse=True
            )

        # Search knowledge store
        results = self.store.search(q_encoding, top_k=top_k)

        if not results:
            return "I could not find relevant knowledge for this question."

        # Build answer
        confidence   = output['confidence'].mean().item()
        epistemic    = output['epistemic_uncertainty'].mean().item()
        mem_conf     = output['memory_confidence'].mean().item()
        best_sim     = results[0]['similarity']

        answer_lines = []

        # Header
        answer_lines.append(f"\n{'─'*55}")
        answer_lines.append(f"🧠 BhaVi's Answer")
        answer_lines.append(f"{'─'*55}")

        # If best match is good enough
        if best_sim > 0.3:
            answer_lines.append(
                f"\nMost relevant knowledge I have:\n"
            )
            for rank, result in enumerate(results):
                sim  = result['similarity']
                text = result['text'].strip()
                src  = os.path.basename(result['source']) if result['source'] else "direct input"
                math = "🔢 " if result['math'] else ""

                answer_lines.append(
                    f"  [{rank+1}] {math}Relevance: {sim:.1%} | Source: {src}"
                )
                answer_lines.append(f"  {text}\n")
        else:
            answer_lines.append(
                "\n  I don't have strong knowledge about this topic yet."
            )
            answer_lines.append(
                "  Feed me more relevant content with /feed <file>"
            )

        # Confidence footer
        answer_lines.append(f"{'─'*55}")
        answer_lines.append(
            f"Confidence: {confidence:.1%} | "
            f"Uncertainty: {epistemic:.1%} | "
            f"Memory: {mem_conf:.1%} | "
            f"Best match: {best_sim:.1%}"
        )
        answer_lines.append(f"{'─'*55}")

        return "\n".join(answer_lines)

    def knowledge_stats(self):
        """Show what BhaVi knows."""
        print(f"\n{'='*55}")
        print(f"  BhaVi Knowledge Base")
        print(f"{'='*55}")
        self.store.stats()
        self.bhavi.status()

    # ── Interactive Chat Loop ─────────────────────────────────────

    def chat_loop(self):
        """
        Interactive chat interface.

        Commands:
            /feed <filepath>   — feed a file to BhaVi
            /learn <text>      — type knowledge directly
            /stats             — show knowledge stats
            /save              — save current state
            /quit              — exit
            anything else      — ask a question
        """
        print("\n" + "🧠 " * 15)
        print("  BhaVi Chat — Ask me anything I've learned")
        print("🧠 " * 15)
        print("\nCommands:")
        print("  /feed <filepath>  — teach BhaVi from a file")
        print("  /learn            — type knowledge directly")
        print("  /stats            — show what BhaVi knows")
        print("  /save             — save memory")
        print("  /quit             — exit")
        print("\nOr just type any question!\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nGoodbye!")
                break

            if not user_input:
                continue

            # ── Commands ──────────────────────────────────────────
            if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                print("Saving and exiting...")
                self.store.save()
                self.bhavi.observer_core.save_memory("bhavi_memory.pt")
                print("Goodbye!")
                break

            elif user_input.lower().startswith('/feed '):
                filepath = user_input[6:].strip()
                if os.path.exists(filepath):
                    self.feed_file(filepath)
                else:
                    print(f"File not found: {filepath}")

            elif user_input.lower() == '/learn':
                print("Type or paste your text (type END on a new line to finish):")
                lines = []
                while True:
                    line = input()
                    if line.strip().upper() == 'END':
                        break
                    lines.append(line)
                text = "\n".join(lines)
                if text.strip():
                    self.feed_text(text, source="manual_input")
                    print("✅ Learned!")

            elif user_input.lower() == '/stats':
                self.knowledge_stats()

            elif user_input.lower() == '/save':
                self.store.save()
                self.bhavi.observer_core.save_memory("bhavi_memory.pt")
                print("✅ Saved!")

            # ── Question ──────────────────────────────────────────
            else:
                answer = self.ask(user_input)
                print(answer)


# ── Entry Point ───────────────────────────────────────────────────
def main():
    chat = BhaViChat()

    # Check for --feed argument
    if len(sys.argv) > 2 and sys.argv[1] == '--feed':
        filepath = sys.argv[2]
        chat.feed_file(filepath)

    # Start interactive chat
    chat.chat_loop()


if __name__ == "__main__":
    main()
