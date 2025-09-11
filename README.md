# Type-from-Image Font Builder

Build a font (TTF/OTF) and a preview PNG from a single black‑on‑white image of your glyphs.

Requirements
- Python 3.10+, uv
- Tools on PATH: `magick` (ImageMagick), `potrace`, `fontforge`, `zip`

Quick start
- Run: `uv run make_font.py --image path/to/sheet.png --name MyFont --preview-text "HELLOWORLD"`
- Outputs in `MyFont/`:
  - `MyFont-Regular.ttf`, `MyFont-Regular.otf`, `MyFont-Regular.ufo/`
  - `characters/png/`, `characters/svg/`, `MyFont_preview.png`

Notes
- You’ll label detected tiles (paste a string or label per tile).
- On errors (missing labels/SVG/fonts), the script fails loudly; fix and re‑run.
