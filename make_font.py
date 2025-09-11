#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "Pillow>=10.0", "numpy>=1.26", "opencv-python-headless>=4.10.0",
#   "ufoLib2>=0.16.0", "fonttools[ufo]>=4.53.0",
# ]
# ///

import sys, os, argparse, json, hashlib, subprocess, tempfile, re
import xml.etree.ElementTree as ET
import numpy as np, cv2, ufoLib2
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple

# Font metrics
UPM = 1000
ASCENT = 780
DESCENT = 220
LSB = 12
RSB = 12
# Dynamic sidebearing range (font units). Per-glyph values are derived from PNG edges.
SB_MIN = 4
SB_MAX = 16

# Image processing
def imread_gray(path): return np.array(Image.open(path).convert('L'))

def binarize(gray, thresh=None, close=0, dilate=0):
    if thresh is None: thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
    bw = (gray <= thresh).astype(np.uint8) * 255
    if close > 0: bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close, close)))
    if dilate > 0: bw = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate)))
    return bw

def find_boxes(bw, min_area=50, max_aspect=20.0):
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    H, W = bw.shape
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area: continue
        ar = max(w/h, h/w)
        if ar > max_aspect: continue
        # Convert to (x0, y0, x1, y1) format and clamp
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x+w), min(H, y+h)
        boxes.append((x0, y0, x1, y1))
    return boxes

def nms_merge(boxes, iou_thresh=0.15):
    """Merge overlapping/nearby boxes"""
    def iou(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0, ix1-ix0), max(0, iy1-iy0)
        inter = iw * ih
        if inter == 0: return 0.0
        area = (ax1-ax0)*(ay1-ay0) + (bx1-bx0)*(by1-by0) - inter
        return inter / max(area, 1)
    
    boxes = boxes[:]
    changed = True
    while changed:
        changed = False
        out = []
        while boxes:
            a = boxes.pop()
            merged = False
            for j, b in enumerate(boxes):
                if iou(a, b) >= iou_thresh:
                    nx0 = min(a[0], b[0])
                    ny0 = min(a[1], b[1])
                    nx1 = max(a[2], b[2])
                    ny1 = max(a[3], b[3])
                    boxes[j] = (nx0, ny0, nx1, ny1)
                    merged = True
                    changed = True
                    break
            if not merged: out.append(a)
        boxes = out
    return boxes

def merge_dots(boxes):
    """Merge small dots (like i, j dots) with their base characters"""
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))  # Sort by y, then x
    merged = []
    skip = set()
    
    for i, box in enumerate(boxes):
        if i in skip:
            continue
            
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        area = w * h
        
        # Check if this could be a dot (small and roughly square)
        if area < 1000 and max(w/h, h/w) < 3.0:  # Small and squarish (more generous)
            # Look for a base character below
            best_match = None
            best_gap = float('inf')
            
            for j, other in enumerate(boxes):
                if j == i or j in skip:
                    continue
                ox0, oy0, ox1, oy1 = other
                
                # Must be below the dot
                if oy0 <= y1:
                    continue
                
                # Check if horizontally aligned
                x_overlap = min(x1, ox1) - max(x0, ox0)
                if x_overlap > 0:  # Any x-overlap
                    # Check if vertically close (dot above base)
                    y_gap = oy0 - y1
                    if 0 <= y_gap < 50:  # Within 50 pixels
                        if y_gap < best_gap:
                            best_gap = y_gap
                            best_match = j
            
            if best_match is not None:
                ox0, oy0, ox1, oy1 = boxes[best_match]
                merged.append((min(x0, ox0), y0, max(x1, ox1), oy1))
                skip.add(i)
                skip.add(best_match)
                continue
            
        if i not in skip:
            merged.append(box)
    
    return merged

def sort_reading_order(boxes): return sorted(boxes, key=lambda b: (b[1]//50, b[0]))

def draw_preview(gray, boxes, path):
    """Draw preview image with bounding boxes"""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(rgb, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    Image.fromarray(rgb).save(path)

def save_tiles(gray, boxes, outdir):
    """Save individual character tiles"""
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for i, (x, y, w, h) in enumerate(boxes):
        tile = Image.fromarray(gray[y:y+h, x:x+w])
        path = os.path.join(outdir, f"{i:03d}.png")
        tile.save(path)
        paths.append(path)
    return paths

# SVG processing
def svg_to_ufo_paths(svg_file):
    """Parse SVG and apply transforms so path coords are absolute in SVG space.
    Supports translate, scale, and matrix transforms on groups and paths.
    """
    NS = '{http://www.w3.org/2000/svg}'

    def multiply_affines(a,b,c,d,e,f, ma,mb,mc,md,me,mf):
        # Compose two affines M * A (apply A then M)
        na = ma*a + mc*b
        nb = mb*a + md*b
        nc = ma*c + mc*d
        nd = mb*c + md*d
        ne = ma*e + mc*f + me
        nf = mb*e + md*f + mf
        return (na,nb,nc,nd,ne,nf)

    def apply_affine(M, x, y):
        a,b,c,d,e,f = M
        return (a*x + c*y + e, b*x + d*y + f)

    def parse_transform(tstr):
        if not tstr:
            return (1.0,0.0,0.0,1.0,0.0,0.0)
        a,b,c,d,e,f = 1.0,0.0,0.0,1.0,0.0,0.0
        for m in re.finditer(r'(matrix|translate|scale)\s*\(([^\)]*)\)', tstr):
            kind = m.group(1)
            nums = [float(x) for x in re.split(r'[\s,]+', m.group(2).strip()) if x]
            if kind == 'matrix' and len(nums) >= 6:
                a,b,c,d,e,f = multiply_affines(a,b,c,d,e,f, *nums[:6])
            elif kind == 'translate':
                tx = nums[0] if len(nums) >= 1 else 0.0
                ty = nums[1] if len(nums) >= 2 else 0.0
                a,b,c,d,e,f = multiply_affines(a,b,c,d,e,f, 1,0,0,1,tx,ty)
            elif kind == 'scale':
                sx = nums[0] if len(nums) >= 1 else 1.0
                sy = nums[1] if len(nums) >= 2 else sx
                a,b,c,d,e,f = multiply_affines(a,b,c,d,e,f, sx,0,0,sy,0,0)
        return (a,b,c,d,e,f)

    def walk(node, cur_M, out_paths):
        node_M = parse_transform(node.get('transform'))
        M = multiply_affines(node_M[0],node_M[1],node_M[2],node_M[3],node_M[4],node_M[5],
                             cur_M[0],cur_M[1],cur_M[2],cur_M[3],cur_M[4],cur_M[5])
        if node.tag == NS + 'path':
            d_attr = node.get('d', '')
            if d_attr:
                cmds = parse_svg_path(d_attr)
                t_cmds = []
                cur = None
                subpath_start = None
                for cmd, coords in cmds:
                    it = iter(coords)
                    if cmd == 'M':
                        # absolute move, subsequent pairs are treated as L
                        x, y = next(it, None), next(it, None)
                        if x is None or y is None:
                            continue
                        cur = (x, y)
                        subpath_start = cur
                        x2, y2 = apply_affine(M, x, y)
                        t_cmds.append(('M', [x2, y2]))
                        for x, y in zip(it, it):
                            cur = (x, y)
                            px, py = apply_affine(M, x, y)
                            t_cmds.append(('L', [px, py]))
                    elif cmd == 'm':
                        dx, dy = next(it, None), next(it, None)
                        if dx is None or dy is None:
                            continue
                        if cur is None:
                            cur = (dx, dy)
                        else:
                            cur = (cur[0] + dx, cur[1] + dy)
                        subpath_start = cur
                        x2, y2 = apply_affine(M, cur[0], cur[1])
                        t_cmds.append(('M', [x2, y2]))
                        for dx, dy in zip(it, it):
                            cur = (cur[0] + dx, cur[1] + dy)
                            px, py = apply_affine(M, cur[0], cur[1])
                            t_cmds.append(('L', [px, py]))
                    elif cmd == 'L':
                        for x, y in zip(it, it):
                            cur = (x, y)
                            px, py = apply_affine(M, x, y)
                            t_cmds.append(('L', [px, py]))
                    elif cmd == 'l':
                        for dx, dy in zip(it, it):
                            if cur is None:
                                cur = (dx, dy)
                            else:
                                cur = (cur[0] + dx, cur[1] + dy)
                            px, py = apply_affine(M, cur[0], cur[1])
                            t_cmds.append(('L', [px, py]))
                    elif cmd == 'C':
                        for x1,y1,x2,y2,x3,y3 in zip(it,it,it,it,it,it):
                            cur = (x3, y3)
                            nx1, ny1 = apply_affine(M, x1, y1)
                            nx2, ny2 = apply_affine(M, x2, y2)
                            nx3, ny3 = apply_affine(M, x3, y3)
                            t_cmds.append(('C', [nx1, ny1, nx2, ny2, nx3, ny3]))
                    elif cmd == 'c':
                        for dx1,dy1,dx2,dy2,dx3,dy3 in zip(it,it,it,it,it,it):
                            if cur is None:
                                cur = (0.0, 0.0)
                            x1, y1 = cur[0] + dx1, cur[1] + dy1
                            x2, y2 = cur[0] + dx2, cur[1] + dy2
                            x3, y3 = cur[0] + dx3, cur[1] + dy3
                            cur = (x3, y3)
                            nx1, ny1 = apply_affine(M, x1, y1)
                            nx2, ny2 = apply_affine(M, x2, y2)
                            nx3, ny3 = apply_affine(M, x3, y3)
                            t_cmds.append(('C', [nx1, ny1, nx2, ny2, nx3, ny3]))
                    elif cmd in ('Z','z'):
                        # Close path; replicate by explicit close in drawing function
                        t_cmds.append(('Z', []))
                        cur = subpath_start
                    else:
                        # Ignore unsupported commands
                        pass
                out_paths.append(t_cmds)
        for child in list(node):
            walk(child, M, out_paths)

    try:
        tree = ET.parse(svg_file)
        root = tree.getroot()
        out = []
        walk(root, (1.0,0.0,0.0,1.0,0.0,0.0), out)
        return out
    except Exception as e:
        print(f"⚠ SVG parse failed for {svg_file}: {e}")
        return []

def parse_svg_path(path_d):
    commands = []
    parts = re.split(r'([MmLlHhVvCcSsQqTtAaZz])', path_d)
    parts = [p.strip() for p in parts if p.strip()]
    
    i = 0
    while i < len(parts):
        if re.match(r'[MmLlHhVvCcSsQqTtAaZz]', parts[i]):
            cmd = parts[i]
            coords = []
            i += 1
            while i < len(parts) and not re.match(r'[MmLlHhVvCcSsQqTtAaZz]', parts[i]):
                coord_str = parts[i].replace(',', ' ')
                coord_nums = [float(x) for x in coord_str.split() if x]
                coords.extend(coord_nums)
                i += 1
            commands.append((cmd, coords))
        else: i += 1
    return commands

def _svg_paths_bbox(svg_paths):
    xs, ys = [], []
    for path_commands in svg_paths:
        for cmd, coords in path_commands:
            it = iter(coords)
            if cmd in ('M','L'):
                for x, y in zip(it, it):
                    xs.append(x); ys.append(y)
            elif cmd == 'C':
                for x1,y1,x2,y2,x3,y3 in zip(it,it,it,it,it,it):
                    xs.extend([x1,x2,x3]); ys.extend([y1,y2,y3])
    if not xs or not ys:
        raise ValueError("Empty SVG path data after transform")
    return min(xs), min(ys), max(xs), max(ys)

def _has_descender(ch: str) -> bool:
    return ch.isalpha() and ch.lower() in set("gjpqy")

def _quantile(vals, q: float) -> float:
    import numpy as _np
    arr = _np.array(vals, dtype=float)
    if arr.size == 0:
        return 0.0
    q = min(max(q, 0.0), 1.0)
    return float(_np.quantile(arr, q))

def _anchor_y_percentile(svg_paths, descender: bool) -> float:
    """Return a robust baseline anchor y from path points.
    For non-descenders, use a high percentile near the bottom (e.g., 95%).
    For descenders, use a lower percentile to ignore long tails (e.g., 70%).
    """
    ys = []
    for path_commands in svg_paths:
        for cmd, coords in path_commands:
            it = iter(coords)
            if cmd in ('M','L'):
                for _x, y in zip(it, it):
                    ys.append(y)
            elif cmd == 'C':
                for _x1,y1,_x2,y2,_x3,y3 in zip(it,it,it,it,it,it):
                    ys.extend([y1, y2, y3])
    if not ys:
        return 0.0
    return _quantile(ys, 0.70 if descender else 0.95)

def _anchor_caps_y(svg_paths, q: float = 0.99) -> float:
    """Anchor for uppercase caps: use a very high percentile of bottom
    to avoid tiny extremities but keep caps aligned tightly to baseline."""
    ys = []
    for path_commands in svg_paths:
        for cmd, coords in path_commands:
            it = iter(coords)
            if cmd in ('M','L'):
                for _x, y in zip(it, it):
                    ys.append(y)
            elif cmd == 'C':
                for _x1,y1,_x2,y2,_x3,y3 in zip(it,it,it,it,it,it):
                    ys.extend([y1, y2, y3])
    if not ys:
        return 0.0
    return _quantile(ys, q)

def _infer_sidebearings_from_png(png_path: str) -> tuple[int, int]:
    """Derive per-glyph left/right sidebearings from edge coverage.
    - Coverage = fraction of rows whose left/rightmost column contains ink
    - More coverage → flatter edge → larger sidebearing
    - Less coverage → round/diagonal edge → smaller sidebearing
    Returns (lsb_units, rsb_units) in font units.
    """
    try:
        img = Image.open(png_path).convert('L')
        arr = np.array(img)
        if arr.size == 0:
            raise ValueError('empty image')
        # Treat <=240 as ink
        left_col = arr[:, 0]
        right_col = arr[:, -1]
        left_cov = float((left_col <= 240).sum()) / max(1, left_col.shape[0])
        right_cov = float((right_col <= 240).sum()) / max(1, right_col.shape[0])
        # Map coverage [0..1] → [SB_MIN..SB_MAX]
        lsb = int(round(SB_MIN + (SB_MAX - SB_MIN) * left_cov))
        rsb = int(round(SB_MIN + (SB_MAX - SB_MIN) * right_cov))
        # Clamp
        lsb = max(SB_MIN, min(SB_MAX, lsb))
        rsb = max(SB_MIN, min(SB_MAX, rsb))
        return lsb, rsb
    except Exception:
        return LSB, RSB

def draw_svg_paths_into_glyph(glyph, svg_paths, lsb, ty, min_x, min_y, max_x, max_y, scale):
    pen = glyph.getPen()
    for path_commands in svg_paths:
        current_pos, path_started = None, False
        for cmd, coords in path_commands:
            if cmd == 'M' and len(coords) >= 2:
                x, y = coords[0], coords[1]
                ufo_x, ufo_y = (x - min_x) * scale + lsb, (max_y - y) * scale + ty
                pen.moveTo((ufo_x, ufo_y))
                current_pos, path_started = (ufo_x, ufo_y), True
            elif cmd == 'L' and len(coords) >= 2 and current_pos:
                x, y = coords[0], coords[1]
                ufo_x, ufo_y = (x - min_x) * scale + lsb, (max_y - y) * scale + ty
                pen.lineTo((ufo_x, ufo_y))
                current_pos = (ufo_x, ufo_y)
            elif cmd == 'C' and len(coords) >= 6 and current_pos:
                x1, y1, x2, y2, x3, y3 = coords[0:6]
                cp1_x, cp1_y = (x1 - min_x) * scale + lsb, (max_y - y1) * scale + ty
                cp2_x, cp2_y = (x2 - min_x) * scale + lsb, (max_y - y2) * scale + ty
                end_x, end_y = (x3 - min_x) * scale + lsb, (max_y - y3) * scale + ty
                pen.curveTo((cp1_x, cp1_y), (cp2_x, cp2_y), (end_x, end_y))
                current_pos = (end_x, end_y)
            elif cmd in ['Z', 'z']:
                if path_started:
                    pen.closePath()
                path_started = False
        if path_started:
            pen.closePath()

def find_svg_for_character(label, output_dir):
    svg_dir = os.path.join(output_dir, "characters", "svg")
    if not os.path.exists(svg_dir): return None
    
    if label.isalpha():
        svg_file = f"{label}_{'upper' if label.isupper() else 'lower'}.svg"
    elif label.isdigit():
        svg_file = f"{label}.svg"
    else:
        # For special characters, use ASCII code
        svg_file = f"char_{ord(label)}.svg"
    
    svg_path = os.path.join(svg_dir, svg_file)
    return svg_path if os.path.exists(svg_path) else None

def find_png_for_character(label, output_dir):
    png_dir = os.path.join(output_dir, "characters", "png")
    if not os.path.exists(png_dir):
        return None
    if label.isalpha():
        base = f"{label}_{'upper' if label.isupper() else 'lower'}"
    elif label.isdigit():
        base = label
    else:
        base = f"char_{ord(label)}"
    p = os.path.join(png_dir, f"{base}.png")
    return p if os.path.exists(p) else None

def save_character_images(gray_img, boxes, labels, output_dir):
    """Save individual character images as PNG and SVG"""
    os.makedirs(output_dir, exist_ok=True)
    png_dir = os.path.join(output_dir, "characters", "png")
    svg_dir = os.path.join(output_dir, "characters", "svg")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)
    
    def trim_horizontal(p: np.ndarray) -> np.ndarray:
        """Trim any fully-white columns from left and right.
        Keeps rows as-is. Assumes text is dark on light background.
        """
        if p.ndim != 2 or p.size == 0:
            return p
        # Consider pixels <= 240 as ink
        ink_cols = (p <= 240).any(axis=0)
        if not ink_cols.any():
            return p  # nothing to trim
        left = int(np.argmax(ink_cols))
        right = int(len(ink_cols) - np.argmax(ink_cols[::-1]))  # exclusive
        if right <= left:
            return p
        return p[:, left:right]

    for i, (x, y, w, h) in enumerate(boxes):
        if i not in labels: continue
        label = labels[i]
        
        # Save PNG
        char_img = gray_img[y:y+h, x:x+w]
        # Trim white-only columns on the sides so the edges are true edges
        char_img = trim_horizontal(char_img)
        if label.isalpha():
            base_filename = f"{label}_{'upper' if label.isupper() else 'lower'}"
        elif label.isdigit():
            base_filename = label
        else:
            base_filename = f"char_{ord(label)}"
        
        png_path = os.path.join(png_dir, f"{base_filename}.png")
        svg_path = os.path.join(svg_dir, f"{base_filename}.svg")
        
        Image.fromarray(char_img).save(png_path)
        
        # Create SVG using potrace
        try:
            with tempfile.NamedTemporaryFile(suffix='.pgm', delete=False) as tmp_pgm:
                tmp_pgm_path = tmp_pgm.name
                subprocess.run(["magick", png_path, "-threshold", "50%", tmp_pgm_path], check=True, capture_output=True)
                subprocess.run(["potrace", tmp_pgm_path, "-s", "-o", svg_path], check=True, capture_output=True)
                os.unlink(tmp_pgm_path)
        except: pass

# Checkpoint system
def get_checkpoint_path(image_path, params):
    img_hash = hashlib.md5(open(image_path, 'rb').read()).hexdigest()
    param_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    checkpoint_dir = os.path.expanduser("~/.make_font_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, f"{img_hash}_{param_hash}.json")

def save_checkpoint(image_path, params, labels):
    path = get_checkpoint_path(image_path, params)
    with open(path, 'w') as f:
        json.dump({"params": params, "labels": labels, "timestamp": os.path.getmtime(image_path)}, f)
    print(f"✓ Saved checkpoint")

def load_checkpoint(image_path, params):
    try:
        path = get_checkpoint_path(image_path, params)
        if not os.path.exists(path): return None
        with open(path, 'r') as f: data = json.load(f)
        if data.get("timestamp") != os.path.getmtime(image_path): return None
        if data.get("params") != params: return None
        return data.get("labels")
    except: return None

# Interactive labeling
def interactive_labeling(image_path, params, workdir, preview_text: str):
    existing = load_checkpoint(image_path, params)
    if existing:
        print(f"\n📁 Found checkpoint with {len(existing)} labeled characters")
        if input("Use existing checkpoint? [Y/n] ").strip().lower() != 'n':
            return {int(k): v for k, v in existing.items()}
    
    gray = imread_gray(image_path)
    bw = binarize(gray, thresh=params["threshold"], close=params["close"], dilate=params["dilate"])
    boxes = find_boxes(bw, min_area=params["min_area"])
    print(f"Found {len(boxes)} raw components before merging")
    boxes = merge_dots(boxes)
    print(f"Have {len(boxes)} components after merging dots")
    # Apply NMS merge if merge_iou is set
    if params.get("merge_iou", 0.0) > 0:
        boxes = nms_merge(boxes, params["merge_iou"])
        print(f"Have {len(boxes)} components after NMS merge")
    boxes = sort_reading_order(boxes)
    # Convert back to (x,y,w,h) format for drawing/saving
    boxes = [(x0, y0, x1-x0, y1-y0) for x0, y0, x1, y1 in boxes]
    
    preview_path = os.path.join(workdir, "_boxes.png")
    tiles_dir = os.path.join(workdir, "tiles")
    draw_preview(gray, boxes, preview_path)
    save_tiles(gray, boxes, tiles_dir)
    
    print(f"\nPreview saved: {preview_path}")
    print(f"Tiles saved in: {tiles_dir}")
    
    # Try to display preview
    viewers = [["kitten", "icat"], ["icat"], ["kitty", "+kitten", "icat"], ["viu"], ["chafa", "--size=80x40"]]
    for viewer_cmd in viewers:
        try:
            subprocess.run(viewer_cmd + [preview_path], check=True, capture_output=False)
            break
        except: continue
    
    # Interactive labeling
    labels = {}
    s = input(f"\nPaste labels for all {len(boxes)} tiles (or leave blank): ").strip()
    if s:
        for i, char in enumerate(s[:len(boxes)]):
            if char != ' ': labels[i] = char
    else:
        # Individual labeling
        for i in range(len(boxes)):
            tile_path = os.path.join(tiles_dir, f'{i:03d}.png')
            print(f"\nIndex {i:03d} → {tile_path}")
            for viewer_cmd in viewers:
                try:
                    subprocess.run(viewer_cmd + [tile_path], check=True, capture_output=False)
                    break
                except: continue
            val = input(" label: ").strip()
            if val.lower() == "space": val = " "
            if len(val) > 1: val = val[0]
            if val: labels[i] = val
    
    mapped = {i: ch for i, ch in labels.items() if ch}
    
    # Check for duplicates
    from collections import Counter
    char_counts = Counter(mapped.values())
    duplicates = {ch: count for ch, count in char_counts.items() if count > 1}
    
    if duplicates:
        print("\n⚠️  Duplicate characters found:")
        for ch, count in duplicates.items():
            indices = [i for i, c in mapped.items() if c == ch]
            print(f"  '{ch}' appears {count} times at indices: {indices}")
        print("\nYou can fix these duplicates now.")
        
        # Allow fixing duplicates
        for ch in duplicates:
            indices = [i for i, c in mapped.items() if c == ch]
            print(f"\nCharacter '{ch}' appears at indices: {indices}")
            for idx in indices:
                tile_path = os.path.join(tiles_dir, f'{idx:03d}.png')
                print(f"  Index {idx} → {tile_path}")
                # Try to display image
                for viewer_cmd in viewers:
                    try:
                        subprocess.run(viewer_cmd + [tile_path], check=True, capture_output=False)
                        break
                    except: continue
                new_val = input(f"  Enter new label for index {idx} (or press Enter to keep '{ch}'): ").strip()
                if new_val:
                    if new_val.lower() == "space": new_val = " "
                    if len(new_val) > 1:
                        print("Taking first char.")
                        new_val = new_val[0]
                    mapped[idx] = new_val
    
    print("\nFinal Summary:", mapped)
    # Early check: ensure preview text coverage in labels
    needed_chars = [ch for ch in preview_text if ch.strip()]
    req_missing = [ch for ch in needed_chars if ch not in mapped.values()]
    if req_missing:
        raise RuntimeError(
            "Labeling incomplete for preview text. Missing: " + ", ".join(req_missing)
        )
    yn = input("Proceed to build font with these labels? [y/N] ").strip().lower()
    if yn != "y": 
        print("Aborted.")
        sys.exit(0)
    
    save_checkpoint(image_path, params, mapped)
    return mapped

def generate_preview_with_ttf(ttf_path, text, output_path):
    """Generate high-quality preview using TTF font"""
    try:
        scale = 2
        font_size = 120 * scale
        margin = 40 * scale

        font = ImageFont.truetype(ttf_path, font_size)

        # Measure text to size canvas
        dummy = Image.new('RGB', (1, 1))
        d = ImageDraw.Draw(dummy)
        bbox = d.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        caption_h = 28 * scale
        img_w = max(1, text_w + 2 * margin)
        img_h = max(1, text_h + 2 * margin + caption_h)

        img = Image.new('RGB', (img_w, img_h), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((margin, margin), text, fill='black', font=font)

        # Caption
        try:
            caption = f"Font: {os.path.basename(ttf_path)}"
            draw.text((margin, margin + text_h + 8 * scale), caption, fill='gray', font=ImageFont.load_default())
        except:
            pass

        # Downsample for crisp text
        img_final = img.resize((max(1, img_w // scale), max(1, img_h // scale)), Image.LANCZOS)
        img_final.save(output_path, quality=95)
        print(f"✓ Saved {output_path}")
        return True
    except Exception as e:
        print(f"⚠ Could not generate preview: {e}")
        return False


def compile_font(image_path, labels, params, output_dir, family, style, preview_text: str):
    gray = imread_gray(image_path)
    bw = binarize(gray, thresh=params["threshold"], close=params["close"], dilate=params["dilate"])
    boxes = find_boxes(bw, min_area=params["min_area"])
    boxes = merge_dots(boxes)
    if params.get("merge_iou", 0.0) > 0:
        boxes = nms_merge(boxes, params["merge_iou"])
    boxes = sort_reading_order(boxes)
    
    # Convert to (x,y,w,h) format for save_character_images  
    boxes_xywh = [(x0, y0, x1-x0, y1-y0) for x0, y0, x1, y1 in boxes]
    save_character_images(gray, boxes_xywh, labels, output_dir)

    # Early validation: ensure preview text glyphs are labeled and traceable
    labeled_chars = set(labels.values())
    required = [ch for ch in preview_text if ch.strip()]
    missing = [ch for ch in required if ch not in labeled_chars]
    if missing:
        raise RuntimeError(
            "Missing required characters for preview: " + ", ".join(missing) +
            " — label these before building."
        )
    # Validate SVG availability and parseability for required chars
    for ch in required:
        svg_path = find_svg_for_character(ch, output_dir)
        if not svg_path or not os.path.exists(svg_path):
            raise FileNotFoundError(
                f"Missing SVG for '{ch}' at expected location. Re-run labeling/tracing. "
                f"Ensure 'magick' (ImageMagick) and 'potrace' are installed and on PATH."
            )
        svg_paths = svg_to_ufo_paths(svg_path)
        if not svg_paths:
            raise ValueError(f"Unparseable SVG for '{ch}' ({svg_path}) — cannot build valid glyph.")
    
    # Create UFO font using SVG paths
    font = ufoLib2.Font()
    font.info.familyName = family
    font.info.styleName = style
    font.info.unitsPerEm = UPM
    font.info.ascender = ASCENT
    font.info.descender = -DESCENT
    
    usable_h = ASCENT - DESCENT
    bot_pad = 40

    # Precompute global scale from reference glyphs (prefer uppercase)
    ref_ph = []
    ref_pw = []
    caps_extents = []  # anchor_y - min_y for uppercase (cap height in SVG units)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        if i not in labels: 
            continue
        ch = labels[i]
        svg_path = find_svg_for_character(ch, output_dir)
        if not svg_path or not os.path.exists(svg_path):
            continue
        svg_paths = svg_to_ufo_paths(svg_path)
        if not svg_paths:
            continue
        min_x, min_y, max_x, max_y = _svg_paths_bbox(svg_paths)
        pw, ph = (max_x - min_x), (max_y - min_y)
        if ch.isupper():
            ref_ph.append(ph)
            ref_pw.append(pw)
            # Use uppercase anchor to measure cap extent above baseline
            anchor_y = _anchor_caps_y(svg_paths, 0.99)
            cap_extent = max(1e-6, anchor_y - min_y)
            caps_extents.append(cap_extent)
    if not ref_ph:
        # fall back to all glyphs
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            if i not in labels: 
                continue
            ch = labels[i]
            svg_path = find_svg_for_character(ch, output_dir)
            if not svg_path or not os.path.exists(svg_path):
                continue
            svg_paths = svg_to_ufo_paths(svg_path)
            if not svg_paths:
                continue
            min_x, min_y, max_x, max_y = _svg_paths_bbox(svg_paths)
            pw, ph = (max_x - min_x), (max_y - min_y)
            ref_ph.append(ph)
            ref_pw.append(pw)
    if not ref_ph:
        raise RuntimeError("No valid SVG glyphs found to compute scaling")
    import statistics
    median_ph = statistics.median(ref_ph)
    avg_pw = statistics.mean(ref_pw) if ref_pw else median_ph
    global_scale = (usable_h - 2*bot_pad) / max(median_ph, 1e-6)
    # Cap-height normalization target (SVG units): make all caps share similar height
    cap_target_svg = statistics.median(caps_extents) if caps_extents else None
    
    # Use original (x0,y0,x1,y1) boxes for iteration; metrics from SVG bboxes
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        if i not in labels: continue
        label = labels[i]
        
        svg_path = find_svg_for_character(label, output_dir)
        if not svg_path or not os.path.exists(svg_path):
            raise FileNotFoundError(f"SVG file not found for character '{label}'. Expected: {svg_path}")
        
        svg_paths = svg_to_ufo_paths(svg_path)
        if not svg_paths:
            raise ValueError(f"Could not parse SVG paths from {svg_path}")
        # Compute bbox and use global scale for consistency
        min_x, min_y, max_x, max_y = _svg_paths_bbox(svg_paths)
        pw, ph = (max_x - min_x), (max_y - min_y)
        if ph <= 0 or pw <= 0:
            raise ValueError(f"Zero-area SVG bbox for '{label}' in {svg_path}")

        # Derive per-glyph sidebearings from PNG edges (fallback to defaults)
        png_path = find_png_for_character(label, output_dir)
        gl_lsb, gl_rsb = _infer_sidebearings_from_png(png_path) if png_path else (LSB, RSB)

        g = font.newGlyph(label)
        g.unicode = ord(label) if len(label) == 1 and label != " " else None

        # Baseline anchor: use absolute bottom for uppercase caps,
        # percentile-based anchor for others (handles curved bottoms/descenders)
        is_cap = bool(label.isalpha() and label.isupper())
        anchor_y = _anchor_caps_y(svg_paths, 0.99) if is_cap else _anchor_y_percentile(svg_paths, _has_descender(label))

        # Per-glyph scale: normalize cap height across uppercase to reduce tall/short variance
        if is_cap and cap_target_svg:
            cap_extent = max(1e-6, anchor_y - min_y)
            # Desired upm cap height is whatever global_scale would give to the median cap
            # So we adjust per-glyph scale proportionally to match the median cap extent
            scale = global_scale * (cap_target_svg / cap_extent)
            # Clamp to avoid extreme distortion
            lo, hi = global_scale * 0.92, global_scale * 1.08
            if scale < lo: scale = lo
            if scale > hi: scale = hi
        else:
            scale = global_scale
        # With mapping ufo_y = (max_y - y) * scale + ty, set anchor to baseline (0)
        ty = -int(round((max_y - anchor_y) * scale))
        draw_svg_paths_into_glyph(g, svg_paths, gl_lsb, ty, min_x, min_y, max_x, max_y, scale)
        adv_x = pw * scale
        # Use glyph outline width for advance (natural spacing)
        adj_adv_x = adv_x
        adv = int(gl_lsb + adj_adv_x + gl_rsb)
        g.width = max(adv, gl_lsb + gl_rsb + 1)
    
    if " " not in font:
        sp = font.newGlyph("space")
        sp.unicode = 0x20
        sp.width = int(UPM*0.33)
    
    # Save UFO font
    ufo_path = os.path.join(output_dir, f"{family}-{style}.ufo")
    font.save(ufo_path, overwrite=True)
    print(f"✓ Saved {ufo_path}")
    
    # Generate TTF and OTF using fontforge
    ttf_path = os.path.join(output_dir, f"{family}-{style}.ttf")
    otf_path = os.path.join(output_dir, f"{family}-{style}.otf")
    
    font_generated = False
    try:
        result_ttf = subprocess.run(["fontforge", "-lang=ff", "-c", f'Open("{ufo_path}"); Generate("{ttf_path}")'], 
                                    capture_output=True, text=True, timeout=60)
        if result_ttf.returncode == 0 and os.path.exists(ttf_path):
            print(f"✓ Generated {ttf_path}")
            font_generated = True
        result_otf = subprocess.run(["fontforge", "-lang=ff", "-c", f'Open("{ufo_path}"); Generate("{otf_path}")'], 
                                    capture_output=True, text=True, timeout=60)
        if result_otf.returncode == 0 and os.path.exists(otf_path):
            print(f"✓ Generated {otf_path}")
            font_generated = True or font_generated
        if not font_generated:
            raise RuntimeError(f"FontForge failed to generate TTF/OTF.\nTTF stderr:\n{result_ttf.stderr}\nOTF stderr:\n{result_otf.stderr}")
    except Exception as e:
        raise
    
    # Generate preview using any generated font (TTF preferred, then OTF)
    print(f"\nGenerating preview for text: '{preview_text}'...")
    preview_path = os.path.join(output_dir, f"{family}_preview.png")
    generated_preview = False
    # Try with generated fonts first
    for font_file in (ttf_path, otf_path):
        if os.path.exists(font_file):
            if generate_preview_with_ttf(font_file, preview_text, preview_path):
                generated_preview = True
                break
    if not generated_preview:
        raise RuntimeError("Preview generation failed: no usable font file found or glyphs not renderable")
    
    # Create zip archive
    print("\nCreating zip archive...")
    zip_name = f"{os.path.basename(output_dir)}.zip"
    zip_path = os.path.join(output_dir, zip_name)
    try:
        orig_dir = os.getcwd()
        os.chdir(output_dir)
        subprocess.run(["zip", "-r", zip_name, ".", "-x", zip_name], capture_output=True, check=True)
        os.chdir(orig_dir)
        print(f"✓ Created {zip_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to create zip archive: {e}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--style", default="Regular")
    p.add_argument("--threshold", type=int, default=None)
    p.add_argument("--close", type=int, default=0)
    p.add_argument("--dilate", type=int, default=0)
    p.add_argument("--min-area", type=int, default=50)
    p.add_argument("--merge-iou", type=float, default=0.0, help="IOU threshold for merging - 0 disables merging")
    p.add_argument("--preview-text", default="PREVIEW", help="Text to render in the final preview image")
    args = p.parse_args()
    
    params = {"threshold": args.threshold, "close": args.close, "dilate": args.dilate, "min_area": args.min_area, "merge_iou": args.merge_iou}
    
    with tempfile.TemporaryDirectory() as workdir:
        labels = interactive_labeling(args.image, params, workdir, args.preview_text)
        compile_font(args.image, labels, params, args.name, args.name, args.style, args.preview_text)
    
    print(f"\n✅ All outputs saved in: {args.name}/")

if __name__ == "__main__": main()
