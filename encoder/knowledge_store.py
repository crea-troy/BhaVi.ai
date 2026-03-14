"""
BhaVi - Knowledge Store
========================
Stores text passages alongside their field encodings.

When BhaVi learns something, we store:
  - The original text (so we can return it)
  - The field encoding (so we can search it)
  - Metadata (source, importance, math content)

When asked a question:
  - Encode the question as a field
  - Find closest stored passages by field similarity
  - Return the most relevant ones

This is BhaVi's retrievable long-term knowledge base.

Author: BhaVi Project
"""

import torch
import json
import os
from typing import List, Optional
from dataclasses import dataclass, asdict


@dataclass
class KnowledgeEntry:
    """One piece of stored knowledge."""
    text: str
    source: str
    math_detected: bool
    importance: float
    gap_at_learning: float
    index: int


class KnowledgeStore:
    """
    Stores passages + encodings for retrieval.

    Two parts:
    1. encodings.pt  — tensor of all field vectors [N, 256]
    2. entries.json  — list of original texts + metadata
    """

    def __init__(self, store_path: str = "bhavi_knowledge"):
        self.store_path = store_path
        self.enc_path   = store_path + "_encodings.pt"
        self.meta_path  = store_path + "_entries.json"

        self.encodings: Optional[torch.Tensor] = None  # [N, 256]
        self.entries: List[KnowledgeEntry] = []

        self._load()

    def add(
        self,
        text: str,
        encoding: torch.Tensor,
        source: str = "",
        math_detected: bool = False,
        importance: float = 0.5,
        gap: float = 0.5
    ):
        """Store a passage with its encoding."""
        idx = len(self.entries)

        entry = KnowledgeEntry(
            text=text,
            source=source,
            math_detected=math_detected,
            importance=importance,
            gap_at_learning=gap,
            index=idx
        )
        self.entries.append(entry)

        enc = encoding.detach().cpu().unsqueeze(0)  # [1, 256]
        if self.encodings is None:
            self.encodings = enc
        else:
            self.encodings = torch.cat([self.encodings, enc], dim=0)

    def search(
        self,
        query_encoding: torch.Tensor,
        top_k: int = 5
    ) -> List[dict]:
        """
        Find most relevant passages for a query.
        Uses cosine similarity between field encodings.
        """
        if self.encodings is None or len(self.entries) == 0:
            return []

        query = query_encoding.detach().cpu()
        if query.dim() == 2:
            query = query.squeeze(0)

        # Normalize
        query_norm = torch.nn.functional.normalize(query.unsqueeze(0), dim=-1)
        store_norm = torch.nn.functional.normalize(self.encodings, dim=-1)

        # Cosine similarity
        similarities = torch.matmul(store_norm, query_norm.T).squeeze(-1)  # [N]

        # Top k
        k = min(top_k, len(self.entries))
        top_scores, top_indices = torch.topk(similarities, k)

        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            entry = self.entries[idx]
            results.append({
                'text': entry.text,
                'source': entry.source,
                'similarity': score,
                'math': entry.math_detected,
                'importance': entry.importance,
                'index': idx
            })

        return results

    def save(self):
        """Save store to disk."""
        if self.encodings is not None:
            torch.save(self.encodings, self.enc_path)

        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(
                [asdict(e) for e in self.entries],
                f, ensure_ascii=False, indent=2
            )

        print(f"[KnowledgeStore] Saved {len(self.entries)} entries")

    def _load(self):
        """Load store from disk."""
        if os.path.exists(self.enc_path) and os.path.exists(self.meta_path):
            self.encodings = torch.load(self.enc_path, weights_only=True)

            with open(self.meta_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            self.entries = [KnowledgeEntry(**e) for e in raw]

            print(f"[KnowledgeStore] Loaded {len(self.entries)} entries from disk")
        else:
            print("[KnowledgeStore] Starting fresh knowledge store")

    def __len__(self):
        return len(self.entries)

    def stats(self):
        if not self.entries:
            print("[KnowledgeStore] Empty — no knowledge stored yet")
            return
        math_count = sum(1 for e in self.entries if e.math_detected)
        sources = list(set(e.source for e in self.entries if e.source))
        print(f"[KnowledgeStore] {len(self.entries)} passages stored")
        print(f"[KnowledgeStore] {math_count} contain math/equations")
        print(f"[KnowledgeStore] Sources: {sources}")
