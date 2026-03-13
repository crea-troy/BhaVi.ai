"""
BhaVi - Universal Input Reader
================================
Reads ANY file format and converts to clean text for the encoder.

Supported formats:
  - .txt  (plain text)
  - .pdf  (books, papers, textbooks)
  - .md   (markdown documents)
  - direct string input (type anything)

After reading, text is:
  1. Cleaned
  2. Split into meaningful chunks (sentences/paragraphs)
  3. Fed to UniversalNeuralFieldEncoder
  4. BhaVi learns

Author: BhaVi Project
"""

import os
import re
from typing import List, Generator


class TextCleaner:
    """Cleans text from any source into clean readable form."""

    @staticmethod
    def clean(text: str) -> str:
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        # Fix common PDF artifacts
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # Fix hyphenated words
        text = re.sub(r'\f', '\n\n', text)             # Form feeds

        # Keep math symbols intact
        text = text.strip()
        return text

    @staticmethod
    def split_into_passages(text: str, passage_size: int = 500) -> List[str]:
        """
        Split text into meaningful passages.
        Tries to split on paragraph boundaries first,
        then sentence boundaries.
        """
        # Split on paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        passages = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) < passage_size:
                current += "\n\n" + para if current else para
            else:
                if current:
                    passages.append(current.strip())
                current = para

        if current:
            passages.append(current.strip())

        # Filter very short passages
        passages = [p for p in passages if len(p) > 20]

        return passages


class UniversalReader:
    """
    Reads any file and returns clean text passages.
    
    Usage:
        reader = UniversalReader()
        
        # From file
        passages = reader.read("physics_book.pdf")
        passages = reader.read("notes.txt")
        
        # Direct text
        passages = reader.read_text("E = mc² means energy equals mass times c squared")
    """

    def __init__(self, passage_size: int = 500):
        self.cleaner = TextCleaner()
        self.passage_size = passage_size

    def read(self, filepath: str) -> List[str]:
        """Read any file and return list of passages."""

        if not os.path.exists(filepath):
            print(f"[Reader] File not found: {filepath}")
            return []

        ext = os.path.splitext(filepath)[1].lower()

        print(f"[Reader] Reading: {filepath}")
        print(f"[Reader] Format: {ext}")

        if ext == '.pdf':
            text = self._read_pdf(filepath)
        elif ext in ['.txt', '.md', '.tex']:
            text = self._read_text_file(filepath)
        else:
            # Try reading as text anyway
            text = self._read_text_file(filepath)

        if not text:
            print(f"[Reader] Could not extract text from {filepath}")
            return []

        text = self.cleaner.clean(text)
        passages = self.cleaner.split_into_passages(text, self.passage_size)

        print(f"[Reader] Extracted {len(passages)} passages")
        print(f"[Reader] Total characters: {len(text):,}")

        return passages

    def read_text(self, text: str) -> List[str]:
        """Read directly from string input."""
        text = self.cleaner.clean(text)
        return self.cleaner.split_into_passages(text, self.passage_size)

    def _read_text_file(self, filepath: str) -> str:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except Exception as e:
            print(f"[Reader] Error reading text file: {e}")
            return ""

    def _read_pdf(self, filepath: str) -> str:
        """
        Read PDF using pypdf (install: pip install pypdf --break-system-packages)
        Falls back to pdfminer if available.
        """
        # Try pypdf first
        try:
            import pypdf
            text_parts = []
            with open(filepath, 'rb') as f:
                reader = pypdf.PdfReader(f)
                total_pages = len(reader.pages)
                print(f"[Reader] PDF has {total_pages} pages")

                for i, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                        if (i + 1) % 10 == 0:
                            print(f"[Reader] Processed {i+1}/{total_pages} pages...")
                    except Exception:
                        continue

            return "\n\n".join(text_parts)

        except ImportError:
            pass

        # Try pdfminer
        try:
            from pdfminer.high_level import extract_text
            print("[Reader] Using pdfminer...")
            return extract_text(filepath)
        except ImportError:
            pass

        print("[Reader] No PDF library found.")
        print("[Reader] Install with: pip install pypdf --break-system-packages")
        return ""
