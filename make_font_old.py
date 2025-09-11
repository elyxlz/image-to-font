#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "Pillow>=10.0",
#   "numpy>=1.26",
#   "opencv-python-headless>=4.10.0",
#   "ufoLib2>=0.16.0",
#   "fonttools[ufo]>=4.53.0",
# ]
# ///
import sys, os, math, argparse, json, pathlib, textwrap, shutil
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import xml.etree.ElementTree as ET
import re
import hashlib
import ufoLib2
from fontTools.fontBuilder import FontBuilder
from fontTools import subset

# Font metrics
UPM = 1000
ASCENT = 780
DESCENT = 220
LSB = 60
RSB = 60

def imread_gray(path:str)->np.ndarray:
    im = Image.open(path).convert("L")
    return np.array(im)

def binarize(gray: np.ndarray, thresh:int|None=None, close:int=3, dilate:int=1) -> np.ndarray:
    if thresh is None:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    if close>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    if dilate>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        bw = cv2.dilate(bw, k, iterations=1)
    return bw

def find_boxes(mask: np.ndarray, min_area:int=64, max_aspect:float=20.0) -> List[Tuple[int,int,int,int]]:
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    H,W=mask.shape
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < min_area: continue
        ar = max(w/h, h/w)
        if ar>max_aspect: continue
        # clamp
        x0=max(0,x); y0=max(0,y); x1=min(W,x+w); y1=min(H,y+h)
        boxes.append((x0,y0,x1,y1))
    return boxes

def nms_merge(boxes: List[Tuple[int,int,int,int]], iou_thresh:float=0.15)->List[Tuple[int,int,int,int]]:
    # greedy merge for overlapping/nearby boxes
    def iou(a,b):
        ax0,ay0,ax1,ay1=a; bx0,by0,bx1,by1=b
        ix0=max(ax0,bx0); iy0=max(ay0,by0); ix1=min(ax1,bx1); iy1=min(ay1,by1)
        iw=max(0,ix1-ix0); ih=max(0,iy1-iy0)
        inter=iw*ih
        if inter==0: return 0.0
        area=(ax1-ax0)*(ay1-ay0)+(bx1-bx0)*(by1-by0)-inter
        return inter/max(area,1)
    boxes=boxes[:]
    changed=True
    while changed:
        changed=False
        out=[]
        while boxes:
            a=boxes.pop()
            merged=False
            for j,b in enumerate(boxes):
                if iou(a,b)>=iou_thresh:
                    nx0=min(a[0],b[0]); ny0=min(a[1],b[1]); nx1=max(a[2],b[2]); ny1=max(a[3],b[3])
                    boxes[j]=(nx0,ny0,nx1,ny1); merged=True; changed=True; break
            if not merged: out.append(a)
        boxes=out
    return boxes

def merge_dots(boxes: List[Tuple[int,int,int,int]]) -> List[Tuple[int,int,int,int]]:
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

def sort_reading_order(boxes: List[Tuple[int,int,int,int]], line_tol:int=20)->List[Tuple[int,int,int,int]]:
    # sort by top-to-bottom (rows), then left-to-right
    byy=sorted(boxes, key=lambda b:(b[1], b[0]))
    rows=[]
    for b in byy:
        x0,y0,x1,y1=b
        placed=False
        for row in rows:
            # same row if y overlaps within tolerance of row first bbox midline
            ry0=min(bb[1] for bb in row); ry1=max(bb[3] for bb in row)
            if not (y1 < ry0-line_tol or y0 > ry1+line_tol):
                row.append(b); placed=True; break
        if not placed: rows.append([b])
    out=[]
    for row in rows:
        out.extend(sorted(row, key=lambda b:b[0]))
    return out

def draw_preview(gray: np.ndarray, boxes: List[Tuple[int,int,int,int]], path:str):
    rgb = Image.fromarray(gray).convert("RGB")
    draw = ImageDraw.Draw(rgb)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(12, min(rgb.size)//40))
    except:
        font = ImageFont.load_default()
    for i,(x0,y0,x1,y1) in enumerate(boxes):
        draw.rectangle([x0,y0,x1,y1], outline=(255,0,0), width=3)
        draw.rectangle([x0,y0-18,x0+38,y0], fill=(255,0,0))
        draw.text((x0+3,y0-17), str(i), fill=(255,255,255), font=font)
    rgb.save(path)

def save_tiles(src_gray: np.ndarray, boxes: List[Tuple[int,int,int,int]], outdir:str)->List[str]:
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    paths=[]
    for i,(x0,y0,x1,y1) in enumerate(boxes):
        tile = Image.fromarray(src_gray[y0:y1, x0:x1])
        p = os.path.join(outdir, f"{i:03d}.png")
        tile.save(p)
        paths.append(p)
    return paths

def signed_area(poly: np.ndarray)->float:
    x=poly[:,0]; y=poly[:,1]
    return 0.5*np.sum(x*np.roll(y,-1)-np.roll(x,-1)*y)

def parse_svg_path(path_d):
    """Parse SVG path data into a list of commands and coordinates"""
    commands = []
    # Split path data by command letters, keeping the letters
    parts = re.split(r'([MmLlHhVvCcSsQqTtAaZz])', path_d)
    parts = [p.strip() for p in parts if p.strip()]
    
    i = 0
    while i < len(parts):
        if re.match(r'[MmLlHhVvCcSsQqTtAaZz]', parts[i]):
            cmd = parts[i]
            coords = []
            i += 1
            # Get coordinates until next command or end
            while i < len(parts) and not re.match(r'[MmLlHhVvCcSsQqTtAaZz]', parts[i]):
                coord_str = parts[i].replace(',', ' ')
                coord_nums = [float(x) for x in coord_str.split() if x]
                coords.extend(coord_nums)
                i += 1
            commands.append((cmd, coords))
        else:
            i += 1
    return commands

def svg_to_ufo_paths(svg_file_path, img_height):
    """Convert SVG paths to UFO-compatible path commands"""
    try:
        tree = ET.parse(svg_file_path)
        root = tree.getroot()
        
        # Find all path elements
        paths = []
        for path_elem in root.findall('.//{http://www.w3.org/2000/svg}path'):
            path_d = path_elem.get('d', '')
            if path_d:
                commands = parse_svg_path(path_d)
                paths.append(commands)
        
        return paths
    except Exception as e:
        print(f"Error parsing SVG {svg_file_path}: {e}")
        return []

def draw_svg_paths_into_glyph(g, svg_paths, scale, lsb, ty, img_height):
    """Draw SVG paths into UFO glyph using proper curves"""
    pen = g.getPen()
    
    for path_commands in svg_paths:
        current_pos = None
        path_started = False
        
        for cmd, coords in path_commands:
            if cmd == 'M':  # Move to absolute
                if len(coords) >= 2:
                    x, y = coords[0], coords[1]
                    # Convert to UFO coordinate system (y-up)
                    ufo_x = x * scale + lsb  
                    ufo_y = (img_height - y) * scale + ty
                    pen.moveTo((ufo_x, ufo_y))
                    current_pos = (ufo_x, ufo_y)
                    path_started = True
            
            elif cmd == 'L':  # Line to absolute
                if len(coords) >= 2 and current_pos:
                    x, y = coords[0], coords[1]
                    ufo_x = x * scale + lsb
                    ufo_y = (img_height - y) * scale + ty
                    pen.lineTo((ufo_x, ufo_y))
                    current_pos = (ufo_x, ufo_y)
            
            elif cmd == 'C':  # Cubic Bezier curve absolute
                if len(coords) >= 6 and current_pos:
                    x1, y1, x2, y2, x3, y3 = coords[0:6]
                    # Convert control points and end point
                    cp1_x, cp1_y = x1 * scale + lsb, (img_height - y1) * scale + ty
                    cp2_x, cp2_y = x2 * scale + lsb, (img_height - y2) * scale + ty
                    end_x, end_y = x3 * scale + lsb, (img_height - y3) * scale + ty
                    pen.curveTo((cp1_x, cp1_y), (cp2_x, cp2_y), (end_x, end_y))
                    current_pos = (end_x, end_y)
            
            elif cmd == 'Q':  # Quadratic Bezier curve absolute
                if len(coords) >= 4 and current_pos:
                    x1, y1, x2, y2 = coords[0:4]
                    cp_x, cp_y = x1 * scale + lsb, (img_height - y1) * scale + ty
                    end_x, end_y = x2 * scale + lsb, (img_height - y2) * scale + ty
                    pen.qCurveTo((cp_x, cp_y), (end_x, end_y))
                    current_pos = (end_x, end_y)
            
            elif cmd == 'Z' or cmd == 'z':  # Close path
                if path_started:
                    pen.closePath()
                    path_started = False
        
        # Close path if not already closed
        if path_started:
            pen.closePath()

def find_svg_for_character(label, output_dir):
    """Find the SVG file for a character label"""
    svg_dir = os.path.join(output_dir, "characters", "svg")
    if not os.path.exists(svg_dir):
        return None
    
    # Try different naming patterns
    if label.isalpha():
        if label.isupper():
            svg_file = f"{label}_upper.svg"
        else:
            svg_file = f"{label}_lower.svg"
    else:
        # For non-alphabetic characters, try the character directly
        svg_file = f"{label}.svg"
    
    svg_path = os.path.join(svg_dir, svg_file)
    return svg_path if os.path.exists(svg_path) else None

def get_image_hash(image_path):
    """Get hash of image file for checkpoint identification"""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def get_checkpoint_path(image_path, params):
    """Get checkpoint file path based on image and parameters"""
    image_hash = get_image_hash(image_path)
    params_str = json.dumps(params, sort_keys=True)
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    
    checkpoint_dir = os.path.expanduser("~/.make_font_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_file = f"{image_hash}_{params_hash}.json"
    return os.path.join(checkpoint_dir, checkpoint_file)

def save_checkpoint(image_path, params, labels):
    """Save character labels to checkpoint file"""
    checkpoint_path = get_checkpoint_path(image_path, params)
    checkpoint_data = {
        "image_path": os.path.abspath(image_path),
        "params": params,
        "labels": labels,
        "timestamp": os.path.getmtime(image_path)
    }
    
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print(f"✓ Saved checkpoint to {checkpoint_path}")

def load_checkpoint(image_path, params):
    """Load character labels from checkpoint file if available"""
    checkpoint_path = get_checkpoint_path(image_path, params)
    
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Verify image hasn't changed
        current_timestamp = os.path.getmtime(image_path)
        if checkpoint_data.get("timestamp") != current_timestamp:
            print("⚠ Image file has been modified since checkpoint was saved")
            return None
        
        # Verify parameters match
        if checkpoint_data.get("params") != params:
            print("⚠ Detection parameters have changed since checkpoint was saved")
            return None
        
        return checkpoint_data.get("labels")
    
    except Exception as e:
        print(f"⚠ Could not load checkpoint: {e}")
        return None


def save_character_images(font, gray_img, boxes, labels, output_dir):
    """Save individual character images as PNG and SVG with letter filenames"""
    import os
    import subprocess
    import tempfile
    from PIL import Image
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for PNG and SVG
    png_dir = os.path.join(output_dir, "png")
    svg_dir = os.path.join(output_dir, "svg")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)
    
    saved_count = 0
    svg_count = 0
    
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        if i not in labels:
            continue
        
        label = labels[i]
        if not label or label == "":
            continue
        
        # Extract character image
        char_img = gray_img[y0:y1, x0:x1]
        pil_img = Image.fromarray(char_img)
        
        # Determine safe filename
        if label == " ":
            base_filename = "space"
        elif label.isalnum():
            # For letters and numbers, use as-is
            if label.isupper():
                base_filename = f"{label}_upper"
            elif label.islower():
                base_filename = f"{label}_lower"
            else:
                base_filename = f"{label}"
        else:
            # For special characters, use ASCII code
            base_filename = f"char_{ord(label)}"
        
        png_filename = base_filename + ".png"
        svg_filename = base_filename + ".svg"
        
        png_filepath = os.path.join(png_dir, png_filename)
        svg_filepath = os.path.join(svg_dir, svg_filename)
        
        # Handle duplicates by adding number suffix
        if os.path.exists(png_filepath):
            counter = 2
            while os.path.exists(os.path.join(png_dir, f"{base_filename}_{counter}.png")):
                counter += 1
            png_filepath = os.path.join(png_dir, f"{base_filename}_{counter}.png")
            svg_filepath = os.path.join(svg_dir, f"{base_filename}_{counter}.svg")
        
        # Save PNG
        pil_img.save(png_filepath)
        saved_count += 1
        
        # Convert to SVG using potrace
        try:
            # Create temporary PGM file
            with tempfile.NamedTemporaryFile(suffix='.pgm', delete=False) as tmp_pgm:
                tmp_pgm_path = tmp_pgm.name
                # Convert PNG to PGM with threshold
                subprocess.run(["magick", png_filepath, "-threshold", "50%", tmp_pgm_path], 
                             check=True, capture_output=True)
                # Trace to SVG
                subprocess.run(["potrace", tmp_pgm_path, "-s", "-o", svg_filepath], 
                             check=True, capture_output=True)
                svg_count += 1
                # Clean up temp file
                os.unlink(tmp_pgm_path)
        except FileNotFoundError as e:
            if svg_count == 0:  # Only warn once
                print(f"⚠ Warning: SVG conversion not available (magick/potrace not found)")
        except subprocess.CalledProcessError as e:
            print(f"⚠ Warning: Failed to convert {base_filename} to SVG: {e}")
    
    print(f"✓ Saved {saved_count} PNG images to {png_dir}/")
    if svg_count > 0:
        print(f"✓ Saved {svg_count} SVG images to {svg_dir}/")

def generate_preview_with_svgs(svg_dir, text, output_path, font_name):
    """Generate preview image using high-quality SVG rasterization"""
    from PIL import Image, ImageDraw, ImageFont
    import subprocess
    import tempfile
    import os
    
    try:
        # Create a high-res white background
        img_width = 1000
        img_height = 200
        img = Image.new('RGB', (img_width, img_height), color='white')
        
        # Position for each letter
        x_pos = 50
        y_pos = 20
        letter_height = 140
        
        for char in text:
            # Find the SVG file for this character
            svg_path = os.path.join(svg_dir, f"{char}.svg")
            if not os.path.exists(svg_path):
                # Try lowercase if uppercase not found
                svg_path = os.path.join(svg_dir, f"{char.lower()}.svg")
            if not os.path.exists(svg_path):
                print(f"⚠ SVG not found for '{char}'")
                x_pos += 80  # Skip space for missing character
                continue
            
            # Convert SVG to high-quality PNG using rsvg-convert or inkscape
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_png:
                tmp_png_path = tmp_png.name
            
            try:
                # Try rsvg-convert first (better quality)
                result = subprocess.run(
                    ["rsvg-convert", "-h", str(letter_height), svg_path, "-o", tmp_png_path],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    # Fallback to ImageMagick convert
                    subprocess.run(
                        ["magick", svg_path, "-resize", f"x{letter_height}", "-background", "white", "-alpha", "remove", tmp_png_path],
                        check=True, capture_output=True
                    )
            except:
                # Last resort: use inkscape
                subprocess.run(
                    ["inkscape", svg_path, "-h", str(letter_height), "-o", tmp_png_path],
                    capture_output=True
                )
            
            # Load the rasterized letter
            letter_img = Image.open(tmp_png_path)
            
            # Paste onto main image
            img.paste(letter_img, (x_pos, y_pos))
            x_pos += letter_img.width + 10  # Add spacing between letters
            
            # Clean up temp file
            os.unlink(tmp_png_path)
        
        # Add caption
        draw = ImageDraw.Draw(img)
        try:
            caption_font = ImageFont.load_default()
            caption = f"Font: {font_name}"
            draw.text((50, 170), caption, fill='gray', font=caption_font)
        except:
            pass
        
        # Save
        img.save(output_path)
        print(f"✓ Saved {output_path} (high-quality from SVGs)")
        return True
    except Exception as e:
        print(f"⚠ Could not generate preview from SVGs: {e}")
        return False

def generate_preview_with_ttf(ttf_path, text, output_path):
    """Generate preview image using the actual TTF font with high quality"""
    from PIL import Image, ImageDraw, ImageFont
    
    try:
        # Create high-resolution image for better anti-aliasing
        scale = 4  # 4x resolution for smoother rendering
        font_size = 120 * scale
        img_width = 1000 * scale
        img_height = 200 * scale
        
        # Load the TTF font at high resolution
        font = ImageFont.truetype(ttf_path, font_size)
        
        # Create high-res image
        img = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw the text at high resolution
        draw.text((50 * scale, 40 * scale), text, fill='black', font=font)
        
        # Add caption
        try:
            caption_font = ImageFont.load_default()
            # Scale caption font if possible
            if hasattr(caption_font, 'font_size'):
                caption_font = ImageFont.truetype(caption_font.path, caption_font.font_size * scale)
            caption = f"Font: {os.path.basename(ttf_path)}"
            draw.text((50 * scale, 160 * scale), caption, fill='gray', font=caption_font)
        except:
            pass
        
        # Downscale with high-quality resampling for smooth anti-aliasing
        img_final = img.resize((1000, 200), Image.LANCZOS)
        
        # Save
        img_final.save(output_path, quality=95)
        print(f"✓ Saved {output_path} (high-quality TTF render)")
        return True
    except Exception as e:
        print(f"⚠ Could not generate preview: {e}")
        return False

def compile_font(image_path:str, labels:Dict[int,str], params:dict, output_dir:str, family:str, style:str):
    gray = imread_gray(image_path)
    bw = binarize(gray, thresh=params["threshold"], close=params["close"], dilate=params["dilate"])
    boxes = find_boxes(bw, min_area=params["min_area"])
    boxes = merge_dots(boxes)  # Merge dots with their base characters
    boxes = sort_reading_order(boxes)  # Skip nms_merge like in interactive_labeling
    font = ufoLib2.Font()
    font.info.familyName = family
    font.info.styleName = style
    font.info.unitsPerEm = UPM
    font.info.ascender = ASCENT
    font.info.descender = -DESCENT
    font.info.versionMajor = 1; font.info.versionMinor = 0
    nd = font.newGlyph(".notdef"); nd.width = UPM//2

    top_pad, bot_pad = 40, 40
    usable_h = UPM - top_pad - bot_pad
    # scale by tile height at vectorization time (per glyph)
    for i,b in enumerate(boxes):
        if i not in labels: continue
        label = labels[i]
        if label == "": continue
        
        # Skip if glyph already exists (duplicate)
        if label in font:
            print(f"  Skipping duplicate glyph '{label}' at index {i}")
            continue
            
        x0,y0,x1,y1=b
        crop = gray[y0:y1, x0:x1]
        # make tight mask for vectorization
        sub_bw = binarize(crop, thresh=params["threshold"], close=params["close_vec"], dilate=0)
        ys, xs = np.where(sub_bw>0)
        if xs.size==0: continue
        bx0, bx1 = xs.min(), xs.max()+1
        by0, by1 = ys.min(), ys.max()+1
        sub = sub_bw[by0:by1, bx0:bx1]
        # Use SVG paths for smooth curves
        svg_path = find_svg_for_character(label, output_dir)
        if not svg_path or not os.path.exists(svg_path):
            raise FileNotFoundError(f"SVG file not found for character '{label}'. Expected: {svg_path}")
        
        svg_paths = svg_to_ufo_paths(svg_path, y1-y0)
        if not svg_paths:
            raise ValueError(f"Could not parse SVG paths from {svg_path}")
        
        g = font.newGlyph(label)
        g.unicode = ord(label) if len(label)==1 and label != " " else None
        
        H, W = y1-y0, x1-x0
        scale = (usable_h) / H
        ty = -DESCENT + bot_pad
        draw_svg_paths_into_glyph(g, svg_paths, scale, LSB, ty, H)
        adv = int(LSB + (W*scale) + RSB)
        g.width = max(adv, LSB+RSB+1)

    if " " not in font:
        sp = font.newGlyph("space"); sp.unicode=0x20; sp.width=int(UPM*0.33)

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Save UFO font
    ufo_path = os.path.join(output_dir, f"{family}-{style}.ufo")
    if os.path.exists(ufo_path):
        shutil.rmtree(ufo_path)
    font.save(ufo_path)
    print(f"✓ Saved font: {ufo_path}")
    
    # Save individual character images (PNG and SVG)
    print("\nSaving character images...")
    chars_dir = os.path.join(output_dir, "characters")
    save_character_images(font, gray, boxes, labels, chars_dir)
    
    # Generate TTF and OTF using fontforge first
    import subprocess
    font_base = os.path.splitext(os.path.basename(ufo_path))[0]
    ttf_path = os.path.join(output_dir, f"{font_base}.ttf")
    otf_path = os.path.join(output_dir, f"{font_base}.otf")
    
    print("\nGenerating TTF/OTF fonts...")
    font_generated = False
    try:
        # Generate TTF
        result = subprocess.run(["fontforge", "-lang=ff", "-c", 
                       f'Open("{ufo_path}"); Generate("{ttf_path}")'],
                      capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Generated {ttf_path}")
            font_generated = True
        else:
            raise subprocess.CalledProcessError(result.returncode, result.args, result.stderr)
        
        # Generate OTF
        result = subprocess.run(["fontforge", "-lang=ff", "-c", 
                       f'Open("{ufo_path}"); Generate("{otf_path}")'],
                      capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Generated {otf_path}")
        else:
            raise subprocess.CalledProcessError(result.returncode, result.args, result.stderr)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"⚠ Could not generate TTF/OTF: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"  Error: {e.stderr}")
        print(f"  You can generate them manually with:")
        print(f"  fontforge -lang=ff -c 'Open(\"{ufo_path}\"); Generate(\"{ttf_path}\")'")
        print(f"  fontforge -lang=ff -c 'Open(\"{ufo_path}\"); Generate(\"{otf_path}\")'")
    
    # Generate preview using high-quality SVGs first, TTF as fallback
    print("\nGenerating AUDIOGEN preview...")
    preview_path = os.path.join(output_dir, "AUDIOGEN_preview.png")
    svg_dir = os.path.join(output_dir, "characters", "svg")
    
    # Try SVG-based preview first for best quality
    if os.path.exists(svg_dir):
        success = generate_preview_with_svgs(svg_dir, "AUDIOGEN", preview_path, f"{family}-{style}")
        if not success and font_generated and os.path.exists(ttf_path):
            # Fallback to TTF if SVG method fails
            generate_preview_with_ttf(ttf_path, "AUDIOGEN", preview_path)
    elif font_generated and os.path.exists(ttf_path):
        # Use TTF if no SVGs available
        generate_preview_with_ttf(ttf_path, "AUDIOGEN", preview_path)
    
    # Create zip archive
    print("\nCreating zip archive...")
    zip_name = os.path.basename(output_dir) + ".zip"
    zip_path = os.path.join(output_dir, zip_name)
    
    # Create zip of the contents
    try:
        orig_dir = os.getcwd()
        os.chdir(output_dir)
        subprocess.run(["zip", "-r", zip_name, ".", "-x", zip_name], 
                       capture_output=True, check=True)
        os.chdir(orig_dir)
        print(f"✓ Created {zip_path}")
    except Exception as e:
        print(f"✗ Failed to create zip: {e}")

def interactive_labeling(image_path:str, params:dict, workdir:str)->Dict[int,str]:
    # Check for existing checkpoint
    existing_labels = load_checkpoint(image_path, params)
    if existing_labels:
        print(f"\n📁 Found saved checkpoint for this image with {len(existing_labels)} labeled characters")
        use_checkpoint = input("Use existing checkpoint? [Y/n] ").strip().lower()
        if use_checkpoint != 'n':
            print("✓ Using checkpoint labels")
            return {int(k): v for k, v in existing_labels.items()}
    
    gray = imread_gray(image_path)
    bw = binarize(gray, thresh=params["threshold"], close=params["close"], dilate=params["dilate"])
    boxes = find_boxes(bw, min_area=params["min_area"])
    print(f"Found {len(boxes)} raw components before merging")
    boxes = merge_dots(boxes)  # Merge dots with their base characters (i, j)
    print(f"Have {len(boxes)} components after merging dots")
    # Skip nms_merge to avoid merging adjacent characters
    boxes = sort_reading_order(boxes)
    preview_path = os.path.join(workdir, "_boxes.png")
    tiles_dir = os.path.join(workdir, "tiles")
    draw_preview(gray, boxes, preview_path)
    save_tiles(gray, boxes, tiles_dir)
    print("\nPreview saved:", preview_path)
    print("Tiles saved in:", tiles_dir)
    
    # Try to display the preview with various terminal image viewers
    import subprocess
    viewers = [["kitten", "icat"], ["icat"], ["kitty", "+kitten", "icat"], ["viu"], ["chafa", "--size=80x40"], ["img2txt", "-W", "80"], ["catimg"]]
    print("\nDetected characters preview:")
    displayed = False
    for viewer_cmd in viewers:
        try:
            subprocess.run(viewer_cmd + [preview_path], check=True, capture_output=False)
            displayed = True
            break
        except:
            continue
    if not displayed:
        print(f"(Could not display image - open {preview_path} manually)")
    
    print("\nOpen the preview image, then label in reading order.")
    print("Input a single character for each index. Enter to skip. Type 'space' for a space.")
    print("You can paste a whole string when prompted to speed things up.")
    labels={}
    # quick batch entry
    s = input("\nOptional: paste a continuous string of labels for all tiles (or leave blank): ").strip()
    if s:
        for i,ch in enumerate(s):
            labels[i]=(" " if ch==" " else ch)
    # fill gaps interactively
    viewers = [["kitten", "icat"], ["icat"], ["kitty", "+kitten", "icat"], ["viu"], ["chafa", "--size=20x10"], ["img2txt", "-W", "20"], ["catimg"]]
    for i in range(len(boxes)):
        if i in labels: continue
        tile_path = os.path.join(tiles_dir, f'{i:03d}.png')
        print(f"\nIndex {i:03d} → {tile_path}")
        # Try to display image with available viewer
        for viewer_cmd in viewers:
            try:
                subprocess.run(viewer_cmd + [tile_path], check=True, capture_output=False)
                break
            except:
                continue
        val = input(" label: ").strip()
        if val.lower()=="space": val=" "
        if len(val)>1:
            print("Take first char of your input."); val=val[0]
        labels[i]=val
    # confirm
    mapped = {i:ch for i,ch in labels.items() if ch}
    
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
                    except:
                        continue
                new_val = input(f"  Enter new label for index {idx} (or press Enter to keep '{ch}'): ").strip()
                if new_val:
                    if new_val.lower()=="space": new_val=" "
                    if len(new_val)>1:
                        print("Taking first char."); new_val=new_val[0]
                    mapped[idx] = new_val
    
    print("\nFinal Summary:", mapped)
    yn = input("Proceed to build font with these labels? [y/N] ").strip().lower()
    if yn!="y": print("Aborted."); sys.exit(0)
    
    # Save checkpoint for future use
    save_checkpoint(image_path, params, mapped)
    
    return mapped

def parse_args(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--image", required=True, help="black-on-white source image")
    p.add_argument("--name", required=True, help="font name (used for output folder and font family)")
    p.add_argument("--style", default="Regular")
    p.add_argument("--threshold", type=int, default=None, help="0-255; None=OTSU")
    p.add_argument("--close", type=int, default=0, help="morph close for box detection")
    p.add_argument("--dilate", type=int, default=0, help="dilate to connect parts")
    p.add_argument("--min-area", type=int, default=50, dest="min_area")
    p.add_argument("--merge-iou", type=float, default=0.0, dest="merge_iou", help="IOU threshold for merging - 0 disables merging")
    p.add_argument("--close-vec", type=int, default=1, dest="close_vec", help="morph close for vectorization")
    p.add_argument("--simplify", type=float, default=0.003, help="contour simplify ratio")
    return p.parse_args(argv)

def main():
    a = parse_args()
    params = dict(threshold=a.threshold, close=a.close, dilate=a.dilate,
                  min_area=a.min_area, merge_iou=a.merge_iou,
                  close_vec=a.close_vec, simplify=a.simplify)
    
    # Use the name argument for output directory and font family
    output_dir = a.name
    family = a.name
    
    # Use temporary directory for working files
    import tempfile
    with tempfile.TemporaryDirectory() as workdir:
        # Do interactive labeling
        labels = interactive_labeling(a.image, params, workdir)
        
        # Compile font and save all outputs
        compile_font(a.image, labels, params, output_dir, family, a.style)
    
    print(f"\n✅ All outputs saved in: {output_dir}/")
    print(f"   ├── {family}-{a.style}.ufo (font file)")
    print(f"   ├── characters/png/ (original images)")
    print(f"   ├── characters/svg/ (vector images)")
    print(f"   ├── AUDIOGEN_preview.png")
    print(f"   └── {output_dir}.zip (archive of everything)")

if __name__ == "__main__":
    main()
