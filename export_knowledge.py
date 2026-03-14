"""
BhaVi - Static Knowledge Exporter
===================================
Exports BhaVi's learned knowledge to static JSON files
that can be hosted on GitHub Pages with NO server needed.

Run this after feeding BhaVi knowledge:
    python3 export_knowledge.py

This creates:
    docs/bhavi_knowledge.json   ← all passages + encodings
    docs/bhavi_config.json      ← model config

Then push to GitHub — your site works instantly.

Author: BhaVi Project
"""

import torch
import json
import os
import sys
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


def export_knowledge(
    store_path: str = "bhavi_knowledge",
    output_dir: str = "docs"
):
    """Export knowledge store to static JSON for GitHub Pages."""

    enc_path  = store_path + "_encodings.pt"
    meta_path = store_path + "_entries.json"

    if not os.path.exists(meta_path):
        print(f"[Export] No knowledge found at {meta_path}")
        print(f"[Export] Run: python3 learn.py first")
        return

    # Load entries
    with open(meta_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    # Load encodings
    encodings = None
    if os.path.exists(enc_path):
        encodings = torch.load(enc_path, weights_only=True)

    print(f"[Export] Found {len(entries)} knowledge entries")

    # Build export structure
    export_data = {
        "version": "1.0",
        "total_passages": len(entries),
        "passages": []
    }

    for i, entry in enumerate(entries):
        passage_data = {
            "id": i,
            "text": entry["text"],
            "source": os.path.basename(entry.get("source", "")),
            "math": entry.get("math_detected", False),
            "importance": entry.get("importance", 0.5),
        }

        # Include encoding as list (for cosine similarity in JS)
        if encodings is not None and i < encodings.shape[0]:
            enc = encodings[i].tolist()
            # Round to 4 decimal places to reduce file size
            enc = [round(x, 4) for x in enc]
            passage_data["encoding"] = enc

        export_data["passages"].append(passage_data)

    # Save to docs/ folder (GitHub Pages serves from here)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "bhavi_knowledge.json")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[Export] Saved to {output_path}")
    print(f"[Export] File size: {size_kb:.1f} KB")
    print(f"[Export] Ready for GitHub Pages ✅")

    # Also save config
    config = {
        "encoding_dim": 256,
        "total_passages": len(entries),
        "version": "bhavi_1.0"
    }
    config_path = os.path.join(output_dir, "bhavi_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nNext steps:")
    print(f"  1. git add docs/")
    print(f"  2. git commit -m 'Update BhaVi knowledge'")
    print(f"  3. git push")
    print(f"  4. Your site at crea-troy.github.io/BhaVi.ai updates automatically")


if __name__ == "__main__":
    export_knowledge()
