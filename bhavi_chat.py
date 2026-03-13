"""
BhaVi - Unified Chat Interface
================================
Single entry point for all BhaVi interactions.

Commands:
  /feed <file>    — learn from any file (pdf, txt, py, etc)
  /improve        — compress knowledge into deeper equations
  /status         — show field status
  /freeze         — freeze core (lock fundamentals permanently)
  /quit           — save and exit

Run:
  conda activate AI
  cd ~/BhaVi.ai
  python3 bhavi_chat.py

Author: BhaVi Project — Jigar Patel
"""

import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bhavi_unified import BhaVi


def main():
    bhavi = BhaVi(load_path=os.path.join(_HERE, "bhavi_unified.pt"))

    print("  Commands: /feed <file>  /improve  /status  /freeze  /quit\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBhaVi: Saving field state. Goodbye.")
            bhavi._save()
            break

        if not user:
            continue

        # Commands
        if user.lower() in {"/quit", "/exit", "quit", "exit"}:
            bhavi._save()
            print("BhaVi: Field state saved. Goodbye.")
            break

        elif user.lower().startswith("/feed "):
            path = user[6:].strip()
            if not os.path.exists(path):
                print(f"BhaVi: File not found: {path}")
            else:
                bhavi.learn_file(path)

        elif user.lower() == "/improve":
            result = bhavi.improve()
            if result:
                print(f"BhaVi: Compressed {result.get('compressions', 0)} "
                      f"knowledge pairs into deeper field equations.")

        elif user.lower() == "/status":
            print(bhavi.status())

        elif user.lower() == "/freeze":
            confirm = input("  Freeze core permanently? (yes/no): ").strip().lower()
            if confirm == "yes":
                bhavi.core.freeze()
                bhavi._save()
                print("BhaVi: Core is now permanently frozen. Roots protected.")
            else:
                print("BhaVi: Core remains unfrozen.")

        else:
            answer = bhavi.respond(user)
            print(f"\nBhaVi: {answer}\n")


if __name__ == "__main__":
    main()
