"""
BhaVi - Knowledge Demo
=======================
Shows how to feed any knowledge to BhaVi.

Run:
    python3 learn.py

Or feed your own file:
    python3 learn.py path/to/your/book.pdf

Author: BhaVi Project
"""

import sys
import os
import importlib.util
import torch

# ── Path setup ────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _import_from(rel_path, module_name):
    full_path = os.path.join(_HERE, rel_path)
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    # Load BhaVi
    _bhavi_mod   = _import_from("bhavi.py", "bhavi_main")
    _feeder_mod  = _import_from("encoder/knowledge_feeder.py", "knowledge_feeder")

    BhaVi          = _bhavi_mod.BhaVi
    KnowledgeFeeder = _feeder_mod.KnowledgeFeeder

    print("\n" + "🧠 " * 20)
    print("  BhaVi Knowledge Learning Demo")
    print("🧠 " * 20)

    # Initialize BhaVi
    bhavi = BhaVi(memory_path="bhavi_memory.pt")

    # Initialize feeder
    feeder = KnowledgeFeeder(bhavi)

    # ── Check if file provided as argument ──────────────────────────
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"\nFeeding file: {filepath}")
        feeder.feed_file(filepath)
        feeder.learning_report()
        return

    # ── Demo: Feed physics knowledge directly ───────────────────────
    print("\n" + "="*55)
    print("  Demo: Teaching BhaVi Physics")
    print("="*55)

    physics_knowledge = [

        # Classical Mechanics
        """
        Newton's First Law of Motion states that an object at rest 
        stays at rest, and an object in motion stays in motion with 
        the same speed and in the same direction, unless acted upon 
        by an unbalanced force. This is also known as the law of inertia.
        """,

        """
        Newton's Second Law: F = ma
        Force equals mass times acceleration.
        If you apply a force F to an object with mass m,
        it will accelerate at rate a = F/m.
        A heavier object requires more force to achieve the same acceleration.
        """,

        """
        Newton's Third Law: For every action there is an equal 
        and opposite reaction. When object A exerts a force on object B,
        object B simultaneously exerts a force equal in magnitude 
        and opposite in direction on object A.
        """,

        # Energy
        """
        The law of conservation of energy states that energy cannot 
        be created or destroyed, only transformed from one form to another.
        Total energy of an isolated system remains constant.
        Kinetic energy KE = ½mv²
        Potential energy PE = mgh
        Total mechanical energy E = KE + PE = constant
        """,

        # Special Relativity
        """
        Einstein's Special Relativity (1905):
        E = mc²
        Energy equals mass times the speed of light squared.
        c = 3 × 10⁸ m/s (speed of light)
        This means mass and energy are equivalent and interchangeable.
        A small amount of mass contains enormous amounts of energy.
        """,

        """
        Time dilation: Moving clocks run slower.
        t' = t / √(1 - v²/c²)
        Where t is proper time, v is velocity, c is speed of light.
        At velocities close to c, time passes more slowly.
        This has been experimentally verified with atomic clocks on aircraft.
        """,

        # Quantum Mechanics
        """
        The Heisenberg Uncertainty Principle states that we cannot 
        simultaneously know both the exact position and exact momentum 
        of a particle with perfect precision.
        ΔxΔp ≥ ℏ/2
        Where ℏ is the reduced Planck constant.
        This is not a limitation of measurement — it is fundamental to nature.
        """,

        """
        Wave-particle duality: Every quantum entity exhibits both 
        wave and particle properties.
        de Broglie wavelength: λ = h/p
        Where h is Planck's constant and p is momentum.
        Electrons, photons, and even atoms show interference patterns.
        """,

        # Thermodynamics
        """
        The Second Law of Thermodynamics states that the total entropy 
        of an isolated system can never decrease over time.
        Entropy S is a measure of disorder.
        ΔS ≥ 0 for any spontaneous process.
        Heat flows naturally from hot to cold, never the reverse.
        This gives time its direction — the arrow of time.
        """,

        # Electromagnetism
        """
        Maxwell's Equations describe all of classical electromagnetism:
        ∇·E = ρ/ε₀         (Gauss's law — electric charges create fields)
        ∇·B = 0             (No magnetic monopoles exist)
        ∇×E = -∂B/∂t        (Faraday — changing B creates E)
        ∇×B = μ₀J + μ₀ε₀∂E/∂t  (Ampere — currents and changing E create B)
        These four equations unify electricity, magnetism, and light.
        """,
    ]

    print(f"\nFeeding {len(physics_knowledge)} physics concepts to BhaVi...\n")

    for i, knowledge in enumerate(physics_knowledge):
        topic = knowledge.strip().split('\n')[0].strip()[:50]
        print(f"\n[{i+1}/{len(physics_knowledge)}] Teaching: {topic}...")
        feeder.feed_text(knowledge, source_name="physics_demo", show_progress=True)

    # ── Learning Report ─────────────────────────────────────────────
    print("\n" + "="*55)
    print("  What BhaVi Learned")
    print("="*55)
    feeder.learning_report()

    # ── Test: Ask BhaVi questions ───────────────────────────────────
    print("\n" + "="*55)
    print("  Testing BhaVi's Understanding")
    print("="*55)

    questions = [
        "What is Newton's second law?",
        "What does E = mc² mean?",
        "What is the Heisenberg uncertainty principle?",
        "What is entropy?",
    ]

    for question in questions:
        feeder.ask(question)
        print()

    print("\n" + "✅ " * 20)
    print("  BhaVi has learned physics!")
    print("  Memory saved to: bhavi_memory.pt")
    print("  Next session will remember all of this.")
    print("✅ " * 20 + "\n")

    print("To feed your own book:")
    print("  python3 learn.py path/to/book.pdf")
    print("  python3 learn.py path/to/notes.txt\n")


if __name__ == "__main__":
    main()
