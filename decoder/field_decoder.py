"""
BhaVi - Field Decoder
======================
Generates text from BhaVi's field representations.

No tokenizer. No vocabulary lookup.
Field state → character stream, one byte at a time.

Architecture:
  field_state [256]
        ↓
  FieldToSequence  (expands field into sequence of vectors)
        ↓
  FieldRNN         (autoregressive field dynamics)
        ↓
  CharHead         (256 possible byte values)
        ↓
  character stream → answer text

Author: BhaVi Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FieldToSequence(nn.Module):
    """
    Expands a single field vector into a sequence of field states.
    This seeds the decoder with the question + context information.
    """

    def __init__(self, field_dim: int = 256, seq_len: int = 32):
        super().__init__()
        self.field_dim = field_dim
        self.seq_len   = seq_len

        # Project field into seq_len separate vectors
        self.expander = nn.Sequential(
            nn.Linear(field_dim, field_dim * 2),
            nn.GELU(),
            nn.Linear(field_dim * 2, field_dim * seq_len),
        )

        # Positional field — each position gets a unique bias
        self.pos_field = nn.Parameter(
            torch.randn(seq_len, field_dim) * 0.02
        )

        self.norm = nn.LayerNorm(field_dim)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """
        Args:
            field: [batch, field_dim]
        Returns:
            sequence: [batch, seq_len, field_dim]
        """
        B = field.shape[0]
        expanded = self.expander(field)                    # [B, seq_len * field_dim]
        seq = expanded.view(B, self.seq_len, self.field_dim)  # [B, seq_len, field_dim]
        seq = seq + self.pos_field.unsqueeze(0)            # add positional field
        return self.norm(seq)


class FieldRNN(nn.Module):
    """
    Autoregressive field dynamics.
    At each step: current field + previous char embedding → next field state.
    This is the core generative engine — pure field dynamics, no attention.
    """

    def __init__(self, field_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.field_dim  = field_dim
        self.hidden_dim = hidden_dim

        # Char embedding — maps previous byte value to field space
        self.char_embed = nn.Embedding(256, field_dim)

        # GRU operates in field space
        self.gru = nn.GRU(
            input_size=field_dim * 2,  # current field + char embed
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Project hidden back to field_dim
        self.field_proj = nn.Sequential(
            nn.Linear(hidden_dim, field_dim),
            nn.LayerNorm(field_dim)
        )

    def forward(
        self,
        context_seq: torch.Tensor,
        char_ids: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context_seq: [batch, seq_len, field_dim]  — from FieldToSequence
            char_ids:    [batch, seq_len]              — previous char byte values
            hidden:      GRU hidden state (None for fresh start)
        Returns:
            field_out:   [batch, seq_len, field_dim]
            hidden:      updated GRU hidden state
        """
        # Embed previous characters
        char_emb = self.char_embed(char_ids)          # [B, seq_len, field_dim]

        # Concatenate context and char embeddings
        rnn_input = torch.cat([context_seq, char_emb], dim=-1)  # [B, seq_len, field_dim*2]

        # Run through GRU
        gru_out, hidden = self.gru(rnn_input, hidden)  # [B, seq_len, hidden_dim]

        # Project to field space
        field_out = self.field_proj(gru_out)           # [B, seq_len, field_dim]

        return field_out, hidden


class CharHead(nn.Module):
    """
    Maps field state → probability distribution over 256 byte values.
    This is what actually produces the output characters.
    """

    def __init__(self, field_dim: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(field_dim, field_dim * 2),
            nn.GELU(),
            nn.Linear(field_dim * 2, 256)   # 256 possible byte values
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """
        Args:
            field: [batch, seq_len, field_dim] or [batch, field_dim]
        Returns:
            logits: [batch, seq_len, 256] or [batch, 256]
        """
        return self.head(field)


class FieldDecoder(nn.Module):
    """
    Complete BhaVi Field Decoder.

    Takes a field representation (question + knowledge context)
    and generates an answer as a character stream.

    No tokenizer. No vocabulary. Pure field → bytes.
    """

    def __init__(
        self,
        field_dim: int   = 256,
        hidden_dim: int  = 512,
        seed_seq_len: int = 32,
    ):
        super().__init__()

        self.field_dim    = field_dim
        self.hidden_dim   = hidden_dim
        self.seed_seq_len = seed_seq_len

        # Field merger — combines question field + context field
        self.field_merger = nn.Sequential(
            nn.Linear(field_dim * 2, field_dim * 2),
            nn.GELU(),
            nn.Linear(field_dim * 2, field_dim),
            nn.LayerNorm(field_dim)
        )

        # Expand merged field into seed sequence
        self.field_to_seq = FieldToSequence(field_dim, seed_seq_len)

        # Autoregressive field dynamics
        self.field_rnn = FieldRNN(field_dim, hidden_dim)

        # Character output head
        self.char_head = CharHead(field_dim)

        total = sum(p.numel() for p in self.parameters())
        print(f"[FieldDecoder] {total:,} parameters")

    def merge_fields(
        self,
        question_field: torch.Tensor,
        context_field: torch.Tensor
    ) -> torch.Tensor:
        """
        Merge question field + knowledge context into one field state.

        Args:
            question_field: [batch, field_dim]
            context_field:  [batch, field_dim]  (avg of retrieved passages)
        Returns:
            merged: [batch, field_dim]
        """
        combined = torch.cat([question_field, context_field], dim=-1)
        return self.field_merger(combined)

    def forward(
        self,
        question_field: torch.Tensor,
        context_field: torch.Tensor,
        target_chars: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Training forward pass.

        Args:
            question_field: [batch, field_dim]
            context_field:  [batch, field_dim]
            target_chars:   [batch, seq_len]  — target byte values for training

        Returns:
            dict with loss, logits, merged_field
        """
        # Merge question + context into one field
        merged = self.merge_fields(question_field, context_field)

        # Expand into seed sequence
        seed_seq = self.field_to_seq(merged)           # [B, seed_len, field_dim]

        if target_chars is not None:
            # Training: teacher forcing
            # Prepend BOS (0) and shift targets right
            B, T = target_chars.shape
            bos   = torch.zeros(B, 1, dtype=torch.long, device=target_chars.device)
            input_chars = torch.cat([bos, target_chars[:, :-1]], dim=1)  # [B, T]

            # Pad/trim seed sequence to match T
            if seed_seq.shape[1] < T:
                pad = seed_seq[:, -1:, :].expand(B, T - seed_seq.shape[1], -1)
                ctx = torch.cat([seed_seq, pad], dim=1)
            else:
                ctx = seed_seq[:, :T, :]

            field_out, _ = self.field_rnn(ctx, input_chars)
            logits        = self.char_head(field_out)  # [B, T, 256]

            # Cross-entropy loss over characters
            loss = F.cross_entropy(
                logits.reshape(-1, 256),
                target_chars.reshape(-1)
            )

            return {
                'loss':         loss,
                'logits':       logits,
                'merged_field': merged
            }

        return {'merged_field': merged}

    @torch.no_grad()
    def generate(
        self,
        question_field: torch.Tensor,
        context_field: torch.Tensor,
        max_chars: int    = 500,
        temperature: float = 0.7,
        stop_token: int   = ord('\n\n'),
    ) -> str:
        """
        Generate an answer character by character.

        Args:
            question_field: [1, field_dim]
            context_field:  [1, field_dim]
            max_chars:      maximum characters to generate
            temperature:    sampling temperature (lower = more focused)

        Returns:
            generated text string
        """
        self.eval()
        device = question_field.device

        # Merge fields
        merged   = self.merge_fields(question_field, context_field)
        seed_seq = self.field_to_seq(merged)           # [1, seed_len, field_dim]

        generated_bytes = []
        hidden     = None
        prev_char  = torch.zeros(1, 1, dtype=torch.long, device=device)  # BOS

        # Expand seed to running context
        ctx = seed_seq

        for step in range(max_chars):
            # Get field state for this step
            if step < ctx.shape[1]:
                step_ctx = ctx[:, step:step+1, :]     # [1, 1, field_dim]
            else:
                step_ctx = ctx[:, -1:, :]             # repeat last

            # Run one RNN step
            field_out, hidden = self.field_rnn(step_ctx, prev_char, hidden)

            # Get char distribution
            logits = self.char_head(field_out[:, 0, :])  # [1, 256]

            # Sample
            if temperature > 0:
                probs     = F.softmax(logits / temperature, dim=-1)
                next_char = torch.multinomial(probs, 1)   # [1, 1]
            else:
                next_char = logits.argmax(dim=-1, keepdim=True)

            byte_val = next_char.item()
            generated_bytes.append(byte_val)

            # Stop on double newline (end of answer signal)
            if len(generated_bytes) >= 4:
                last4 = bytes(generated_bytes[-4:])
                if last4 == b'\n\n\n\n' or last4 == b'    ':
                    break

            # Update prev_char for next step
            prev_char = next_char.unsqueeze(0) if next_char.dim() == 1 else next_char

        # Decode bytes to string, ignoring errors
        return bytes(generated_bytes).decode('utf-8', errors='replace').strip()
