import os
import cairosvg
from pathlib import Path
from PIL import Image, ImageOps
from io import BytesIO
import xml.etree.ElementTree as ET

ORIGINAL_IMGS_FOLDER = Path("../original")
RESIZED_IMGS_FOLDER = Path("../resized")
VALID_EXTENSIONS = ["svg", "png", "jpg"]
TARGET_SIZE = 672

def check_extension(img_directory: Path, valid_extensions: list[str]) -> bool:
    # Check if all files in the directory have valid extensions
    for file_name in os.listdir(img_directory):
        if not any(file_name.lower().endswith(f".{ext}") for ext in valid_extensions):
            print(f"Invalid file extension found: {file_name}")
            return False
    return True

def svg_to_png_bytes(svg_path: Path) -> BytesIO:
    """Convert SVG file to PNG in memory and return as BytesIO"""
    print(f"Converting SVG in memory: {svg_path.name}")
    
    # Read SVG content
    with open(svg_path, 'rb') as f:
        svg_content = f.read()
    
    # Parse SVG to get original dimensions
    tree = ET.fromstring(svg_content)
    
    # Get width and height from SVG attributes
    width = tree.get('width')
    height = tree.get('height')
    viewBox = tree.get('viewBox')
    
    # Helper function to convert various units to pixels
    def parse_dimension(value: str) -> float:
        """Convert SVG dimension string to pixels, returns None for percentages"""
        if not value:
            return None
        
        value = value.strip()
        
        # Skip percentages - we'll use viewBox instead
        if '%' in value:
            return None
        
        # Remove and convert units to pixels (96 DPI standard)
        unit_conversions = {
            'px': 1.0,
            'pt': 1.333333,  # 1pt = 1.333px
            'pc': 16.0,       # 1pc = 16px
            'mm': 3.7795275591,  # 1mm ≈ 3.78px
            'cm': 37.795275591,  # 1cm ≈ 37.8px
            'in': 96.0,       # 1in = 96px
        }
        
        for unit, conversion in unit_conversions.items():
            if value.endswith(unit):
                return float(value.replace(unit, '')) * conversion
        
        # If no unit, assume pixels
        return float(value)
    
    # Determine aspect ratio
    w = None
    h = None
    
    if width and height:
        w = parse_dimension(width)
        h = parse_dimension(height)
    
    # If we couldn't parse dimensions (e.g., percentages), use viewBox
    if (w is None or h is None) and viewBox:
        # viewBox format: "minX minY width height"
        parts = viewBox.split()
        if len(parts) == 4:
            _, _, w, h = map(float, parts)
    
    # Calculate target dimensions (longer side = TARGET_SIZE)
    if w > h:
        output_width = TARGET_SIZE
        output_height = int(h * (TARGET_SIZE / w))
    else:
        output_height = TARGET_SIZE
        output_width = int(w * (TARGET_SIZE / h))
    
    # Convert SVG to PNG at calculated dimensions
    png_bytes = cairosvg.svg2png(
        bytestring=svg_content,
        output_width=output_width,
        output_height=output_height,
        background_color='white',
        unsafe=True
    )
    return BytesIO(png_bytes)

def resize_and_save_images(src_folder: Path, dst_folder: Path) -> None:
    for img_path in src_folder.iterdir():
        ext = img_path.suffix.lower()
        
        # Load image (SVG must be converted first)
        if ext == ".svg":
            img_data = svg_to_png_bytes(img_path)
            img = Image.open(img_data).convert("RGB")
            output_name = img_path.stem + ".png"
        else:
            img = Image.open(img_path).convert("RGB")
            output_name = img_path.name
        
        print(f"Processing: {output_name}")
        
        w, h = img.size
        
        # Resize longest side to TARGET_SIZE (only for non-SVG images)
        if ext != ".svg":
            if w > h:
                new_w = TARGET_SIZE
                new_h = int(h * (TARGET_SIZE / w))
            else:
                new_h = TARGET_SIZE
                new_w = int(w * (TARGET_SIZE / h))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            w, h = new_w, new_h
        
        # Padding
        pad_w = TARGET_SIZE - w
        pad_h = TARGET_SIZE - h
        padding = (
            pad_w // 2,
            pad_h // 2,
            pad_w - pad_w // 2,
            pad_h - pad_h // 2
        )
        img = ImageOps.expand(img, padding, fill=(0, 0, 0))  # black padding
        
        # Save final processed image ONLY
        out_path = dst_folder / output_name
        img.save(out_path)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    RESIZED_IMGS_FOLDER.mkdir(exist_ok=True)
    
    if not check_extension(ORIGINAL_IMGS_FOLDER, VALID_EXTENSIONS):
        raise ValueError("Some files in the original images folder have invalid extensions.")
    
    resize_and_save_images(ORIGINAL_IMGS_FOLDER, RESIZED_IMGS_FOLDER)
    print("All images processed successfully!")