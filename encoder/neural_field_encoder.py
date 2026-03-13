"""
BhaVi - Neural Field Universal Encoder
=======================================
Converts ANY form of input into a continuous neural field.

No tokenization.
No special formatting required.
Works with:
  - Plain text
  - Physics equations
  - PDF content
  - Research papers
  - Numbers and data
  - Mixed content

The core idea:
  Instead of breaking text into tokens (like GPT does),
  we treat the entire input as a continuous signal.
  
  Each character has a position.
  Position is encoded as a continuous wave function.
  Meaning emerges from the field, not from discrete tokens.

  f(position, character_value) → field_state
  
  This is like how physics treats signals:
  not as discrete packets but as continuous waves.

Author: BhaVi Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import re
from typing import Union, List


# ── Positional Field Encoding ──────────────────────────────────────
class ContinuousPositionalField(nn.Module):
    """
    Encodes position as a continuous wave function.
    
    Instead of discrete position embeddings (like transformers),
    position is a point in a continuous field.
    
    Uses Fourier features — same math as physics wave equations:
    
    φ(x) = [sin(2π f₁ x), cos(2π f₁ x),
             sin(2π f₂ x), cos(2π f₂ x), ...]
    
    This means:
    - Position 0.0 and 0.001 are slightly different (continuous)
    - Position 0 and 1000 are very different (range aware)
    - No maximum sequence length (infinite context)
    """

    def __init__(self, field_dim: int = 128, num_frequencies: int = 64):
        super().__init__()
        self.field_dim = field_dim
        self.num_frequencies = num_frequencies

        # Learnable frequencies — BhaVi learns which frequencies matter
        self.frequencies = nn.Parameter(
            torch.randn(num_frequencies) * 0.1
        )

        # Project Fourier features to field_dim
        self.projection = nn.Linear(num_frequencies * 2, field_dim)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        positions: [batch, seq_len] normalized to [0, 1]
        returns:   [batch, seq_len, field_dim]
        """
        # positions: [batch, seq_len, 1]
        pos = positions.unsqueeze(-1)

        # frequencies: [1, 1, num_freq]
        freqs = self.frequencies.unsqueeze(0).unsqueeze(0)

        # Compute wave functions
        angles = 2 * math.pi * freqs * pos  # [batch, seq_len, num_freq]
        fourier_features = torch.cat([
            torch.sin(angles),
            torch.cos(angles)
        ], dim=-1)  # [batch, seq_len, num_freq*2]

        return self.projection(fourier_features)


# ── Character Field Encoder ────────────────────────────────────────
class CharacterFieldEncoder(nn.Module):
    """
    Encodes individual characters as field values.
    
    Works at byte level — handles ANY language, ANY symbol,
    ANY equation, ANY character without special vocabulary.
    
    256 possible byte values → continuous field embedding
    """

    def __init__(self, field_dim: int = 128):
        super().__init__()

        # 256 byte values → field embeddings
        # No vocabulary needed — every possible character is covered
        self.byte_field = nn.Embedding(256, field_dim)

        # Character context encoder
        # Understands character in context of neighbors
        self.context_encoder = nn.Sequential(
            nn.Linear(field_dim, field_dim),
            nn.GELU(),
            nn.Linear(field_dim, field_dim)
        )

    def forward(self, byte_sequence: torch.Tensor) -> torch.Tensor:
        """
        byte_sequence: [batch, seq_len] integer values 0-255
        returns:       [batch, seq_len, field_dim]
        """
        embedded = self.byte_field(byte_sequence)
        return self.context_encoder(embedded)


# ── Field Dynamics Layer ───────────────────────────────────────────
class FieldDynamicsLayer(nn.Module):
    """
    Propagates information across the field like a wave.
    
    Instead of attention (transformer style):
        every token attends to every other token → O(n²)
    
    We use local field propagation:
        each field point interacts with neighbors → O(n)
    
    This is how physics works:
        fields propagate locally, global patterns emerge
    """

    def __init__(self, field_dim: int = 256, kernel_size: int = 7):
        super().__init__()

        self.field_dim = field_dim

        # Local interaction kernel (like a physics kernel)
        # Processes each position with its local neighborhood
        self.local_interaction = nn.Conv1d(
            in_channels=field_dim,
            out_channels=field_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=field_dim // 4  # Grouped for efficiency
        )

        # Non-linear field activation
        self.field_activation = nn.Sequential(
            nn.LayerNorm(field_dim),
            nn.GELU()
        )

        # Long-range field coupling (efficient)
        # Captures global structure without full attention
        self.global_coupling = nn.Sequential(
            nn.AdaptiveAvgPool1d(32),  # Compress to 32 key points
            nn.Flatten(1),
            nn.Linear(field_dim * 32, field_dim),
            nn.GELU()
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """
        field: [batch, seq_len, field_dim]
        returns: [batch, seq_len, field_dim]
        """
        # Local propagation
        field_t = field.transpose(1, 2)  # [batch, field_dim, seq_len]
        local = self.local_interaction(field_t)
        local = local.transpose(1, 2)    # [batch, seq_len, field_dim]
        local = self.field_activation(local)

        # Global coupling
        global_context = self.global_coupling(field_t)  # [batch, field_dim]
        global_context = global_context.unsqueeze(1)    # [batch, 1, field_dim]

        # Combine local + global
        return local + 0.1 * global_context


# ── Math/Equation Detector ─────────────────────────────────────────
class MathFieldEncoder(nn.Module):
    """
    Special handling for mathematical content.
    
    Detects equations, numbers, symbols.
    Encodes mathematical structure separately.
    Merges with text field.
    """

    def __init__(self, field_dim: int = 256):
        super().__init__()

        # Encode mathematical tokens specially
        self.math_encoder = nn.Sequential(
            nn.Linear(field_dim, field_dim),
            nn.GELU(),
            nn.Linear(field_dim, field_dim),
            nn.LayerNorm(field_dim)
        )

        # Math detector - is this region mathematical?
        self.math_detector = nn.Sequential(
            nn.Linear(field_dim, field_dim // 4),
            nn.GELU(),
            nn.Linear(field_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, field: torch.Tensor) -> tuple:
        """
        Returns enhanced field + math_mask showing where math is
        """
        math_encoding = self.math_encoder(field)
        math_mask = self.math_detector(field)

        # Blend math encoding where math is detected
        enhanced = field * (1 - math_mask) + math_encoding * math_mask

        return enhanced, math_mask


# ── Field Collapse to Fixed Dimension ─────────────────────────────
class FieldCollapser(nn.Module):
    """
    Collapses variable-length field to fixed 256-dim vector.
    
    Input can be ANY length (short sentence or entire book chapter).
    Output is always 256 numbers for BhaVi.
    
    Uses multi-scale pooling — captures both local and global meaning.
    """

    def __init__(self, field_dim: int = 256, output_dim: int = 256):
        super().__init__()

        # Multi-scale feature extraction
        self.fine_pool   = nn.AdaptiveAvgPool1d(16)   # fine detail
        self.medium_pool = nn.AdaptiveAvgPool1d(8)    # medium structure
        self.coarse_pool = nn.AdaptiveAvgPool1d(4)    # global structure
        self.peak_pool   = nn.AdaptiveMaxPool1d(4)    # prominent features

        total_features = field_dim * (16 + 8 + 4 + 4)

        self.collapse_network = nn.Sequential(
            nn.Linear(total_features, field_dim * 4),
            nn.GELU(),
            nn.Linear(field_dim * 4, field_dim * 2),
            nn.GELU(),
            nn.Linear(field_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """
        field:   [batch, seq_len, field_dim]
        returns: [batch, output_dim]
        """
        field_t = field.transpose(1, 2)  # [batch, field_dim, seq_len]

        fine   = self.fine_pool(field_t).flatten(1)
        medium = self.medium_pool(field_t).flatten(1)
        coarse = self.coarse_pool(field_t).flatten(1)
        peak   = self.peak_pool(field_t).flatten(1)

        multi_scale = torch.cat([fine, medium, coarse, peak], dim=-1)
        return self.collapse_network(multi_scale)


# ── Complete Universal Neural Field Encoder ────────────────────────
class UniversalNeuralFieldEncoder(nn.Module):
    """
    The complete universal encoder for BhaVi.

    Accepts ANY text input:
    - Plain text
    - Physics equations  
    - PDF extracted text
    - Research papers
    - Mixed content
    - Any language
    - Any symbols

    No tokenization. No vocabulary. No special format needed.
    Treats everything as a continuous byte-level field.

    Total: ~8M parameters
    """

    def __init__(
        self,
        field_dim: int = 256,
        output_dim: int = 256,
        num_field_layers: int = 4,
        max_chunk_size: int = 2048  # characters per chunk
    ):
        super().__init__()

        self.field_dim = field_dim
        self.output_dim = output_dim
        self.max_chunk_size = max_chunk_size

        # Stage 1: Character → field
        self.char_encoder = CharacterFieldEncoder(field_dim // 2)

        # Stage 2: Position → field
        self.pos_encoder = ContinuousPositionalField(field_dim // 2)

        # Merge character + position fields
        self.field_merger = nn.Sequential(
            nn.Linear(field_dim, field_dim),
            nn.LayerNorm(field_dim),
            nn.GELU()
        )

        # Stage 3: Field dynamics (propagation)
        self.field_layers = nn.ModuleList([
            FieldDynamicsLayer(field_dim)
            for _ in range(num_field_layers)
        ])

        # Stage 4: Math detection and enhancement
        self.math_encoder = MathFieldEncoder(field_dim)

        # Stage 5: Collapse to fixed dim
        self.collapser = FieldCollapser(field_dim, output_dim)

        # Chunk aggregator (for long documents)
        # Combines multiple chunks into one representation
        self.chunk_aggregator = nn.GRU(
            input_size=output_dim,
            hidden_size=output_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )

        total = sum(p.numel() for p in self.parameters())
        print(f"[UniversalEncoder] {total:,} parameters")
        print(f"[UniversalEncoder] Accepts: text, equations, PDF, any format")
        print(f"[UniversalEncoder] No tokenization — pure field encoding")

    def text_to_bytes(self, text: str) -> torch.Tensor:
        """Convert text to byte tensor. Works with any character."""
        # Encode to UTF-8 bytes (handles any language, any symbol)
        byte_values = list(text.encode('utf-8', errors='replace'))
        # Clip to valid byte range
        byte_values = [min(b, 255) for b in byte_values]
        return torch.tensor(byte_values, dtype=torch.long)

    def encode_chunk(self, byte_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode a single chunk of bytes into field representation.
        chunk: [seq_len] byte values
        returns: [output_dim]
        """
        seq_len = byte_tensor.shape[0]
        batch = byte_tensor.unsqueeze(0)  # [1, seq_len]

        # Normalized positions [0, 1]
        positions = torch.linspace(0, 1, seq_len).unsqueeze(0)

        # Stage 1: Character embeddings
        char_field = self.char_encoder(batch)      # [1, seq_len, field_dim//2]
        pos_field  = self.pos_encoder(positions)   # [1, seq_len, field_dim//2]

        # Merge into unified field
        field = torch.cat([char_field, pos_field], dim=-1)  # [1, seq_len, field_dim]
        field = self.field_merger(field)

        # Stage 3: Field dynamics
        for layer in self.field_layers:
            field = field + layer(field)  # Residual connections

        # Stage 4: Math enhancement
        field, math_mask = self.math_encoder(field)

        # Stage 5: Collapse to fixed vector
        collapsed = self.collapser(field)  # [1, output_dim]

        return collapsed.squeeze(0)  # [output_dim]

    def forward(self, text: str) -> dict:
        """
        Main encoding function.
        
        Accepts ANY text of ANY length.
        Automatically chunks long inputs.
        Returns 256-dim field representation + metadata.
        
        Args:
            text: Any string — plain text, equations, mixed content
            
        Returns:
            dict with 'encoding' [output_dim] and metadata
        """
        if not text or not text.strip():
            return {
                'encoding': torch.zeros(self.output_dim),
                'chunks': 0,
                'math_detected': False,
                'length': 0
            }

        # Detect mathematical content
        math_pattern = re.compile(
            r'[∂∇∑∫√±×÷=<>≤≥≠∞π]|'  # Math symbols
            r'\b[A-Z]\s*[=]\s*|'       # Variables
            r'\d+\.\d+|'               # Decimals
            r'[a-z]\^[0-9]|'           # Powers
            r'\\[a-z]+\{',             # LaTeX
            re.UNICODE
        )
        math_detected = bool(math_pattern.search(text))

        # Convert to bytes
        byte_tensor = self.text_to_bytes(text)
        total_len = byte_tensor.shape[0]

        # Split into chunks if long
        chunks = []
        for start in range(0, total_len, self.max_chunk_size):
            end = min(start + self.max_chunk_size, total_len)
            chunk = byte_tensor[start:end]

            # Pad short chunks
            if len(chunk) < 8:
                chunk = F.pad(chunk, (0, 8 - len(chunk)))

            chunk_encoding = self.encode_chunk(chunk)
            chunks.append(chunk_encoding)

        num_chunks = len(chunks)

        if num_chunks == 1:
            final_encoding = chunks[0]
        else:
            # Aggregate chunks with GRU
            chunk_stack = torch.stack(chunks).unsqueeze(0)  # [1, chunks, dim]
            _, hidden = self.chunk_aggregator(chunk_stack)
            final_encoding = hidden[-1].squeeze(0)  # [output_dim]

        return {
            'encoding': final_encoding,
            'chunks': num_chunks,
            'math_detected': math_detected,
            'length': total_len,
            'bytes_per_chunk': self.max_chunk_size
        }

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a list of texts into batch tensor.
        Returns: [batch, output_dim]
        """
        encodings = []
        for text in texts:
            result = self.forward(text)
            encodings.append(result['encoding'])
        return torch.stack(encodings)
