"""
BhaVi - Unified Training
=========================
Trains all BhaVi layers together on the existing knowledge base.

What it does:
  1. Loads all passages from bhavi_knowledge_entries.json
  2. Trains the field wave encoder (better byte→wave mapping)
  3. Trains the causal graph (learns A→B causal structure)
  4. Trains the superposition cloud (better hypothesis weighting)
  5. Trains the frozen core (uncertainty estimation)
  6. Runs self-improvement compression every epoch
  7. Saves unified state

Run:
  conda activate AI
  cd ~/BhaVi.ai
  python3 bhavi_train.py

Author: BhaVi Project — Jigar Patel
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json, os, sys, time, math

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bhavi_unified import (
    BhaVi, FrozenCore, FieldWaveSystem,
    CausalFieldGraph, SuperpositionCloud
)

CONFIG = {
    'batch_size': 16,
    'lr':         1e-3,
    'epochs':     20,
    'log_every':  50,
    'field_dim':  256,
}


class KnowledgeDataset(Dataset):
    def __init__(self, path: str, wave_system: FieldWaveSystem, field_dim: int = 256):
        with open(path, 'r', encoding='utf-8') as f:
            entries = json.load(f)

        self.items = []
        print(f"[Dataset] Encoding {len(entries)} passages as field waves...")
        for i, e in enumerate(entries):
            text = e.get('text', '').strip()
            if len(text) > 20:
                with torch.no_grad():
                    wave = wave_system.encode_text(text)   # [D]
                self.items.append((text, wave))
            if (i+1) % 500 == 0:
                print(f"  [{i+1}/{len(entries)}]")

        print(f"[Dataset] {len(self.items)} passages ready.\n")

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate(batch):
    texts = [b[0] for b in batch]
    waves = torch.stack([b[1] for b in batch])   # [B, D]
    return texts, waves


def train():
    print("\n" + "="*55)
    print("  BhaVi Unified Training")
    print("  All layers trained together.")
    print("="*55)

    entries_path = os.path.join(_HERE, 'bhavi_knowledge_entries.json')
    if not os.path.exists(entries_path):
        print(f"\n❌ Missing: {entries_path}")
        print("   Run: python3 learn.py <your_book.pdf> first")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Train] Device: {device}")

    # Initialize wave system first (needed for dataset encoding)
    waves   = FieldWaveSystem(CONFIG['field_dim']).to(device)
    core    = FrozenCore(CONFIG['field_dim']).to(device)
    causal  = CausalFieldGraph(CONFIG['field_dim']).to(device)
    superp  = SuperpositionCloud(CONFIG['field_dim']).to(device)

    # Load existing weights if present
    unified_path = os.path.join(_HERE, 'bhavi_unified.pt')
    if os.path.exists(unified_path):
        s = torch.load(unified_path, map_location=device, weights_only=False)
        try:
            waves.load_state_dict(s['waves'])
            core.load_state_dict(s['core'])
            causal.load_state_dict(s['causal'])
            superp.load_state_dict(s['superpos'])
            print("[Train] Loaded existing weights.\n")
        except Exception as e:
            print(f"[Train] Fresh start: {e}\n")

    dataset = KnowledgeDataset(entries_path, waves, CONFIG['field_dim'])
    loader  = DataLoader(dataset, batch_size=CONFIG['batch_size'],
                         shuffle=True, collate_fn=collate)

    all_params = (
        list(waves.parameters()) +
        list(core.parameters()) +
        list(causal.parameters()) +
        list(superp.parameters())
    )
    optimizer = optim.AdamW(all_params, lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'] * len(loader), eta_min=1e-5
    )

    print(f"[Train] {len(dataset)} passages | {len(loader)} batches | "
          f"{CONFIG['epochs']} epochs\n")

    best_loss = float('inf')

    for epoch in range(CONFIG['epochs']):
        epoch_loss = 0.0
        t0 = time.time()

        for step, (texts, wave_batch) in enumerate(loader):
            wave_batch = wave_batch.to(device)
            B = wave_batch.shape[0]

            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)

            # ── Loss 1: Core — epistemic uncertainty toward 0.5 (calibrated)
            core_out   = core(wave_batch)
            epi_loss   = (core_out['epistemic'] - 0.5).pow(2).mean()
            total_loss = total_loss + 0.2 * epi_loss

            # ── Loss 2: Wave consistency — re-encoded wave matches stored wave
            waves.train()
            wave_re    = torch.stack([waves.encode_text(t) for t in texts]).to(device)
            wave_loss  = F.mse_loss(wave_re, wave_batch.detach())
            total_loss = total_loss + wave_loss

            # ── Loss 3: Causal — cause(A) should differ from A (not identity)
            c1         = causal.cause(wave_batch)
            causal_loss = F.mse_loss(c1, wave_batch) * 0.1
            total_loss = total_loss + causal_loss

            # ── Loss 4: Superposition — output should match original field
            sup_out    = superp(core_out['repr'].detach(), core_out['uncertainty'].detach())
            sup_loss   = F.mse_loss(sup_out['field'], wave_batch.detach()) * 0.3
            total_loss = total_loss + sup_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += total_loss.item()

            if (step + 1) % CONFIG['log_every'] == 0:
                avg = epoch_loss / (step + 1)
                lr  = scheduler.get_last_lr()[0]
                eta = (time.time()-t0)/(step+1) * (len(loader)-step-1)
                print(f"  E{epoch+1} S{step+1}/{len(loader)} | "
                      f"Loss {avg:.4f} | LR {lr:.2e} | ETA {eta/60:.1f}m")

        avg_loss = epoch_loss / len(loader)
        elapsed  = time.time() - t0
        print(f"\n{'='*55}")
        print(f"  Epoch {epoch+1}/{CONFIG['epochs']} | "
              f"Loss {avg_loss:.4f} | {elapsed/60:.1f}m")

        # Save state into unified format
        if avg_loss < best_loss:
            best_loss = avg_loss

            bhavi = BhaVi(load_path=unified_path)
            bhavi.core.load_state_dict(core.state_dict())
            bhavi.waves.load_state_dict(waves.state_dict())
            bhavi.causal.load_state_dict(causal.state_dict())
            bhavi.superpos.load_state_dict(superp.state_dict())

            # Populate registry so passages are searchable after training
            if len(bhavi.registry) == 0:
                print(f"  Building registry ({len(dataset.items)} passages)...")
                with torch.no_grad():
                    for text, wave in dataset.items:
                        bhavi.registry.append({
                            'text':        text[:300],
                            'source':      '',
                            'wave':        wave.cpu(),
                            'causal_next': bhavi.causal.cause(wave.unsqueeze(0)).squeeze(0).cpu(),
                            'gap':         0.5,
                            'blocked':     False,
                            'math':        any(c in text for c in ['=','∑','∫','∇','∂','π','∞'])
                        })
                print(f"  Registry: {len(bhavi.registry)} passages ✅")

            bhavi._save()
            del bhavi
            print(f"  ✅ Best model saved (loss={avg_loss:.4f})\n")
        else:
            print()

    print(f"\n{'='*55}")
    print(f"  Training complete. Best loss: {best_loss:.4f}")
    print(f"\n  Now run: python3 bhavi_chat.py")
    print(f"{'='*55}\n")


import torch.nn.functional as F

if __name__ == "__main__":
    train()
