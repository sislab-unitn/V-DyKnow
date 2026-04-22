import json
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote

def load_json(file_path: Path):
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)

headers = {
    "User-Agent": "DyknowDownloader/1.0 (a@b.com) python-requests",
}

input_file = load_json(Path("../../annotations/final_refined_passages_with_images.json"))
output_dir = Path("../original/")
output_dir.mkdir(exist_ok=True)



# Check if all images have different names and the actual extensions 
images = set()
for key, data in input_file.items():
    if "image_url" in data:
        if data["image_url"] != None:
            img_url = data["image_url"]
            if img_url in images:
                print(f"Duplicate image URL found: {img_url}")
                continue
            images.add(img_url)
    else:
        raise ValueError(f"No image URL found for key: {key}")

# Download images
for url in images:
    print(f"Downloading {url}...")

    # if picture already saved, skip
    filename = unquote(url.split("/")[-1])
    save_path = output_dir / filename
    if save_path.exists():
        print(f" → File already exists: {save_path}")
        continue

    # Get the redirected URL from Special:FilePath
    r = requests.get(url, headers=headers, allow_redirects=True, timeout=15)
    r.raise_for_status()

    # Determine filename from final URL
    final_url = r.url
    filename = unquote(final_url.split("/")[-1])

    save_path = output_dir / filename

    if not filename.lower().endswith(".svg"):
        # Some redirects end in .svg?width=... — strip params
        filename = filename.split("?")[0]
    
    if filename.lower().endswith(".tif") or filename.lower().endswith(".tiff"):
        img = Image.open(BytesIO(r.content))
        # Replace .tif/.tiff with .png
        filename = filename.rsplit(".", 1)[0] + ".png"
        save_path = output_dir / filename
        img.save(save_path, format="PNG")
        print(f" → Converted TIFF → PNG and saved to {save_path}")
    else:
        # Save normally
        save_path.write_bytes(r.content)
        print(f" → Saved to {save_path}")
print("Done!")

