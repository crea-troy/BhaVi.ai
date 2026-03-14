# BhaVi.ai — Neural Field Intelligence

An original AI architecture that thinks in **continuous fields**, not tokens.

- No transformers. No tokenization.
- Self-improving with protected roots.
- Runs locally on any machine (~50M params, 18MB).
- Deploys to GitHub Pages with zero server cost.

---

## Folder Structure

```
BhaVi.ai/
├── docs/                    ← GitHub Pages serves this
│   ├── index.html           ← The chat web interface
│   └── bhavi_knowledge.json ← Generated after training (git push this)
│
├── core/frozen_core.py      ← Layer 1: Immutable roots
├── memory/observer_core.py  ← Layer 3: Three-zone memory
├── curiosity/               ← Layer 4: Self-improvement engine
├── superposition/           ← Layer 5+6: Multi-hypothesis + decision
├── encoder/                 ← Universal input encoder (text/PDF/equations)
│
├── bhavi.py                 ← Complete BhaVi model
├── main.py                  ← Test BhaVi works
├── learn.py                 ← Feed knowledge to BhaVi
├── chat.py                  ← Local terminal chat
├── export_knowledge.py      ← Export learned knowledge → docs/
└── setup.sh                 ← Install dependencies
```

---

## Quick Start

### Step 1 — Install dependencies
```bash
bash setup.sh
```

### Step 2 — Test BhaVi runs
```bash
python3 main.py
```

### Step 3 — Feed BhaVi knowledge
```bash
# Built-in physics demo
python3 learn.py

# Your own book or PDF
python3 learn.py /path/to/book.pdf
python3 learn.py /path/to/notes.txt
```

### Step 4 — Chat locally
```bash
python3 chat.py
```

### Step 5 — Export to web
```bash
python3 export_knowledge.py
```

### Step 6 — Push to GitHub
```bash
git add docs/bhavi_knowledge.json
git commit -m "Update BhaVi knowledge"
git push
```

### Step 7 — Enable GitHub Pages
Go to your repo → **Settings** → **Pages**
→ Source: `main` branch, `/docs` folder → **Save**

Your site is live at: `https://crea-troy.github.io/BhaVi.ai`

---

## How It Works

```
Local (Python)                    Web (Browser)
──────────────                    ─────────────
BhaVi reads your books            Loads bhavi_knowledge.json
Neural field encoding             JS cosine similarity search
Saves to knowledge store          Returns ranked results
Export to docs/                   Shows confidence + uncertainty
git push → GitHub Pages           No server needed ever
```

---

## BhaVi Architecture

| Layer | Name | Function |
|---|---|---|
| 1 | Frozen Core Field | Immutable roots, uncertainty split |
| 2 | Superposition Cloud | Holds multiple interpretations |
| 3 | Observer Core | Three-zone persistent memory |
| 4 | Curiosity Engine | Gap detection, self-improvement |
| 5 | Collapse Decision | Late-binding answer resolution |

---

## Key Properties

- **~5M parameters** — tiny but architecturally intelligent
- **18MB RAM** — runs on phone, laptop, Raspberry Pi
- **Persistent memory** — remembers across sessions
- **Root protection** — core knowledge cannot be corrupted
- **No tokenization** — processes raw bytes as continuous field

---

## License
MIT
