"""
BhaVi - Field Decoder Trainer
================================
Trains BhaVi's own text generation from its knowledge base.

What happens:
  1. Loads all stored Feynman + general knowledge passages
  2. For each passage: encodes it as a field vector (the input)
  3. Also encodes the passage text as target characters (the output)
  4. Trains FieldDecoder to regenerate the passage from its field
  5. After training: BhaVi can generate answers in its own words

Run:
    conda activate AI
    cd ~/BhaVi.ai
    python3 train_decoder.py

This takes 2-4 hours on CPU. Run it overnight.
After training: python3 chat.py  (BhaVi will now generate answers)

Author: BhaVi Project
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import sys
import importlib.util
import time
import math

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


# ── Hyperparameters ───────────────────────────────────────────────
CONFIG = {
    'field_dim':      256,
    'hidden_dim':     512,
    'seed_seq_len':   32,
    'max_chars':      256,     # max chars per training sample
    'batch_size':     16,
    'learning_rate':  3e-4,
    'epochs':         10,
    'save_every':     1,       # save checkpoint every N epochs
    'decoder_path':   'bhavi_decoder.pt',
    'log_every':      50,      # print loss every N steps
}


# ── Dataset ───────────────────────────────────────────────────────
class KnowledgeDataset(Dataset):
    """
    Each item: (field_vector, char_sequence)
    - field_vector: pre-computed 256-dim encoding of the passage
    - char_sequence: the passage text as byte values (target)
    """

    def __init__(
        self,
        entries_path: str,
        encodings_path: str,
        max_chars: int = 256
    ):
        self.max_chars = max_chars

        # Load text entries
        print(f"[Dataset] Loading entries from {entries_path}...")
        with open(entries_path, 'r', encoding='utf-8') as f:
            entries = json.load(f)

        # Load field encodings
        print(f"[Dataset] Loading encodings from {encodings_path}...")
        encodings = torch.load(encodings_path, weights_only=True)

        # Build pairs
        self.pairs = []
        for i, entry in enumerate(entries):
            if i >= encodings.shape[0]:
                break
            text = entry.get('text', '').strip()
            if len(text) < 20:
                continue  # skip very short passages

            field_vec = encodings[i]                    # [256]

            # Convert text to byte sequence, clipped to max_chars
            text_bytes = text.encode('utf-8', errors='replace')[:max_chars]

            # Pad to max_chars with zeros
            padded = list(text_bytes) + [0] * (max_chars - len(text_bytes))
            char_tensor = torch.tensor(padded, dtype=torch.long)

            self.pairs.append((field_vec, char_tensor))

        print(f"[Dataset] {len(self.pairs)} training pairs ready")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# ── Training loop ─────────────────────────────────────────────────
def train():
    print("\n" + "="*55)
    print("  BhaVi Field Decoder Training")
    print("="*55)

    # Check required files exist
    entries_path   = os.path.join(_HERE, 'bhavi_knowledge_entries.json')
    encodings_path = os.path.join(_HERE, 'bhavi_knowledge_encodings.pt')

    if not os.path.exists(entries_path):
        print(f"\n❌ Missing: {entries_path}")
        print("   Run: python3 learn.py <your_book.pdf> first")
        sys.exit(1)

    if not os.path.exists(encodings_path):
        print(f"\n❌ Missing: {encodings_path}")
        print("   Run: python3 learn.py <your_book.pdf> first")
        sys.exit(1)

    # Load FieldDecoder
    _decoder_mod = _load("decoder/field_decoder.py", "field_decoder")
    FieldDecoder  = _decoder_mod.FieldDecoder

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Train] Device: {device}")

    # Dataset + DataLoader
    dataset = KnowledgeDataset(
        entries_path   = entries_path,
        encodings_path = encodings_path,
        max_chars      = CONFIG['max_chars']
    )
    loader = DataLoader(
        dataset,
        batch_size  = CONFIG['batch_size'],
        shuffle     = True,
        num_workers = 0,
        pin_memory  = False
    )

    # Model
    model = FieldDecoder(
        field_dim    = CONFIG['field_dim'],
        hidden_dim   = CONFIG['hidden_dim'],
        seed_seq_len = CONFIG['seed_seq_len'],
    ).to(device)

    # Load existing checkpoint if available
    decoder_path = os.path.join(_HERE, CONFIG['decoder_path'])
    start_epoch  = 0
    if os.path.exists(decoder_path):
        print(f"[Train] Loading checkpoint: {decoder_path}")
        ckpt = torch.load(decoder_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt.get('epoch', 0)
        print(f"[Train] Resuming from epoch {start_epoch}")

    # Optimizer with cosine LR schedule
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = CONFIG['learning_rate'],
        weight_decay = 1e-4
    )

    total_steps = CONFIG['epochs'] * len(loader)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-5
    )

    # ── Training epochs ───────────────────────────────────────────
    print(f"\n[Train] Starting training for {CONFIG['epochs']} epochs")
    print(f"[Train] {len(dataset)} samples, {len(loader)} batches/epoch")
    print(f"[Train] Batch size: {CONFIG['batch_size']}")
    print(f"[Train] This will take 2-4 hours on CPU — run overnight\n")

    best_loss   = float('inf')
    global_step = 0

    for epoch in range(start_epoch, start_epoch + CONFIG['epochs']):
        model.train()
        epoch_loss  = 0.0
        epoch_start = time.time()

        for step, (field_vecs, char_seqs) in enumerate(loader):
            field_vecs = field_vecs.to(device)    # [B, 256]
            char_seqs  = char_seqs.to(device)     # [B, max_chars]

            # For context field: use the same field vector
            # (in real use, this would be retrieved knowledge context)
            # During training, we teach: given field → reproduce its text
            context_field = field_vecs + torch.randn_like(field_vecs) * 0.05

            optimizer.zero_grad()

            output = model(
                question_field = field_vecs,
                context_field  = context_field,
                target_chars   = char_seqs
            )

            loss = output['loss']
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            # Logging
            if (step + 1) % CONFIG['log_every'] == 0:
                avg   = epoch_loss / (step + 1)
                lr    = scheduler.get_last_lr()[0]
                ppl   = math.exp(min(avg, 10))       # perplexity
                elapsed = time.time() - epoch_start
                eta   = elapsed / (step + 1) * (len(loader) - step - 1)
                print(
                    f"  Epoch {epoch+1} | Step {step+1}/{len(loader)} | "
                    f"Loss {avg:.4f} | PPL {ppl:.1f} | "
                    f"LR {lr:.2e} | ETA {eta/60:.1f}m"
                )

        # ── End of epoch ──────────────────────────────────────────
        avg_loss = epoch_loss / len(loader)
        elapsed  = time.time() - epoch_start
        ppl      = math.exp(min(avg_loss, 10))

        print(f"\n{'='*55}")
        print(f"  Epoch {epoch+1} complete")
        print(f"  Loss:       {avg_loss:.4f}")
        print(f"  Perplexity: {ppl:.1f}  (lower = better, target <5)")
        print(f"  Time:       {elapsed/60:.1f} minutes")
        print(f"{'='*55}\n")

        # Save checkpoint
        if (epoch + 1) % CONFIG['save_every'] == 0:
            ckpt = {
                'epoch':       epoch + 1,
                'model_state': model.state_dict(),
                'loss':        avg_loss,
                'config':      CONFIG,
            }
            torch.save(ckpt, decoder_path)
            print(f"[Train] Checkpoint saved → {decoder_path}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = decoder_path.replace('.pt', '_best.pt')
                torch.save(ckpt, best_path)
                print(f"[Train] New best model → {best_path}")

        # Quick generation test every epoch
        print("[Train] Quick generation test...")
        test_generate(model, dataset, device)

    print(f"\n{'='*55}")
    print(f"  Training complete!")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Model saved: {decoder_path}")
    print(f"\n  Now run: python3 chat.py")
    print(f"  BhaVi will generate answers in its own words.")
    print(f"{'='*55}\n")


def test_generate(model, dataset, device, num_tests=2):
    """Quick generation test to see how BhaVi is doing."""
    model.eval()
    with torch.no_grad():
        for i in range(min(num_tests, len(dataset))):
            field_vec, char_seq = dataset[i]
            field_vec = field_vec.unsqueeze(0).to(device)

            # Use same field as context (self-reproduction test)
            context_field = field_vec.clone()

            generated = model.generate(
                question_field = field_vec,
                context_field  = context_field,
                max_chars      = 120,
                temperature    = 0.7
            )

            # Show target vs generated
            target_bytes = char_seq.numpy().tolist()
            target_text  = bytes(
                [b for b in target_bytes if b > 0]
            ).decode('utf-8', errors='replace')[:80]

            print(f"  Target:    {target_text!r}")
            print(f"  Generated: {generated[:80]!r}")
            print()

    model.train()


if __name__ == "__main__":
    train()
