# Type-from-Image Font Builder

Turn a single image of your alphabet into a working font (TTF/OTF) and an instant preview image. The pipeline labels glyphs from an image, traces them to vectors, builds a UFO, exports TTF/OTF, and renders a preview of your chosen text using the actual built font.

Highlights
- Robust glyph detection and interactive labeling (fast paste or per-tile).
- Vectorization via potrace + SVG parsing with transform support.
- Consistent sizing and baseline alignment for uppercase and lowercase.
- Natural spacing from per-glyph sidebearings inferred from PNG edge “ink”.
- Fail-fast: if anything is missing or unrenderable, it crashes loudly.

## Quick Start

1) Generate a clean source image of the typeface (optional: with GPT-4o)
- Ask for a high-contrast sheet: black glyphs on white background, no gradients, consistent baseline, ample spacing between glyphs.
- Example prompt you can paste into GPT-4o:
  "Create a black-on-white sheet with uppercase A–Z, lowercase a–z, and digits 0–9 in the style of [describe your style]. Use solid black fills, no outlines, no shadows, and high contrast. Keep a consistent baseline and similar heights, each glyph clearly separated with white space between characters. 2000×1000 px or larger."

2) Build the font
- With uv (Python runner) installed:
  - `uv run make_font.py --image path/to/your_sheet.png --name MyFont --preview-text "HELLOWORLD"`
- The script will:
  - Detect and tile characters
  - Let you paste labels (or label per tile)
  - Trace to SVG (potrace)
  - Build UFO, export TTF/OTF (fontforge)
  - Render a preview image `<name>_preview.png` from the built font (text set by `--preview-text`)

3) Outputs
- A folder named after `--name`, e.g. `MyFont/` containing:
  - `MyFont-Regular.ttf` and `MyFont-Regular.otf`
  - `MyFont-Regular.ufo/` (source)
  - `characters/png/` and `characters/svg/` (per-glyph images)
  - `MyFont_preview.png` (rendered from the actual TTF/OTF)
  - `MyFont.zip` (archive of the folder)

## Requirements

- Python 3.10+
- uv (https://github.com/astral-sh/uv) to run the script: `uv run make_font.py ...`
- External tools (must be on PATH):
  - ImageMagick (`magick`)
  - potrace
  - fontforge
  - zip
- Optional terminal image viewers for previewing tiles: `icat` (kitty), `viu`, or `chafa`

### Install hints
- macOS (Homebrew): `brew install imagemagick potrace fontforge zip`
- Debian/Ubuntu: `sudo apt-get install -y imagemagick potrace fontforge zip`

## Usage

- Basic:
  - `uv run make_font.py --image path/to/sheet.png --name AUDIOGEN`
- Optional parameters:
  - `--style Regular` (default)
  - `--threshold`, `--close`, `--dilate`, `--min-area`, `--merge-iou`

During labeling:
- You can paste a continuous string (e.g., `ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789`) when prompted.
- For individual tiles, type a single character; type `space` for a space.

The build “screams and crashes” on problems — missing labels, unparseable SVGs, failed font export, or preview render. Fix what it tells you, then re-run.

## How spacing and sizing work

- Vertical alignment
  - Uppercase: aligned to a common baseline using a trimmed-bottom anchor; cap heights normalized across all caps.
  - Lowercase: anchored by percentiles; descenders (g/j/p/q/y) hang below baseline naturally.
- Spacing
  - Sidebearings are inferred per glyph from the PNG’s left/right ink edges for natural spacing (stems vs. round shapes) and no overlaps.
  - Advance width = outline width + per-glyph LSB/RSB.

## Troubleshooting

- “Bitmap missing for glyph” when previewing: a glyph didn’t make it into the font (likely unlabeled or empty outline). The script now fails early with explicit errors for required preview letters.
- “Unparseable SVG” or missing SVG file: ensure `magick` and `potrace` are installed and on PATH; re-run.
- Preview clipping or overlaps: report which letters; spacing and scaling can be tuned, but defaults should be sensible.

## Make it a public repo

- Initialize git and push wherever you like (GitHub, GitLab, etc.):
  - `git init`
  - `git add .`
  - `git commit -m "Initial commit: type-from-image font builder"`
  - Create a new empty repo on your hosting provider, then:
    - `git branch -M main`
    - `git remote add origin <your-remote-url>`
    - `git push -u origin main`

This repository’s `.gitignore` excludes all build artifacts so only the code and docs are committed.

## Notes

- The script writes checkpoints under `~/.make_font_checkpoints` so you can resume labeling. They’re outside the repo.
- If your source image is noisy or has uneven sizes, consider regenerating it (e.g., with GPT‑4o) for better results.
