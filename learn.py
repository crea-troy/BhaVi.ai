"""
BhaVi - Knowledge Learner
==========================
Feeds any file or text to BhaVi and saves to knowledge store.

Usage:
    python3 learn.py                          # built-in physics demo
    python3 learn.py /path/to/book.pdf        # feed a PDF
    python3 learn.py /path/to/notes.txt       # feed a text file

After running, do:
    python3 export_knowledge.py
    git add docs/bhavi_knowledge.json
    git push
"""

import sys
import os
import importlib.util

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


def main():
    _chat_mod = _load("chat.py", "chat_main")
    BhaViChat = _chat_mod.BhaViChat

    print("\n" + "="*55)
    print("  BhaVi Knowledge Learner")
    print("="*55)

    chat = BhaViChat()

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            sys.exit(1)
        print(f"\nFeeding: {filepath}\n")
        chat.feed_file(filepath)
        print(f"\n✅ Done! Knowledge store: {len(chat.store)} passages")
        print(f"\nNext step: python3 export_knowledge.py")
        return

    # Built-in demo
    print("\nRunning built-in physics demo\n")
    texts = [
        "Newton's First Law: An object at rest stays at rest unless acted upon by a force. This is the law of inertia.",
        "Newton's Second Law: F = ma. Force equals mass times acceleration.",
        "Newton's Third Law: For every action there is an equal and opposite reaction.",
        "Conservation of Energy: Energy cannot be created or destroyed, only transformed. KE = half m v squared.",
        "Einstein Special Relativity: E = mc squared. Mass and energy are equivalent. c = 3 times 10 to the 8 m/s.",
        "Heisenberg Uncertainty Principle: delta-x times delta-p >= h-bar over 2. Cannot know position and momentum exactly.",
        "Wave-particle duality: Quantum objects are both wave and particle. de Broglie wavelength = h divided by momentum.",
        "Second Law of Thermodynamics: Entropy always increases. Heat flows from hot to cold. This gives time its direction.",
        "Maxwell Equations: Four equations unify electricity, magnetism and light. Changing E fields create B fields and vice versa.",
        "Quantum superposition: A quantum system exists in multiple states until measured. Measurement collapses the wave function.",
    ]

    for i, text in enumerate(texts):
        print(f"[{i+1}/{len(texts)}] {text[:60]}...")
        chat.feed_text(text, source_name="physics_demo", show_progress=False)

    print(f"\n✅ Done! Knowledge store: {len(chat.store)} passages")
    print(f"\nNext step: python3 export_knowledge.py")


if __name__ == "__main__":
    main()
