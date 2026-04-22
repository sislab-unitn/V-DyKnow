import json
import requests
from pathlib import Path
from urllib.parse import unquote

def load_json(file_path: Path):
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)

headers = {
    "User-Agent": "DyknowDownloader/1.0 (a@b.com) python-requests",
}

input_file = load_json(Path("../../annotations/wikidata_combined.json"))
output_dir = Path("../original/")
output_dir.mkdir(exist_ok=True)



# Check if all images have different names and the actual extensions 
images = set()
for category, data in input_file.items():
    for subject, info in data.items():
        if "images" in info:
            for img_name, img_url in info["images"].items():
                if img_url in images:
                    raise ValueError(f"Duplicate image URL found: {img_url}")
                images.add(img_url)
        else:
            raise ValueError(f"No images found for subject: {subject} in category: {category}")

# Download images
for url in images:
    print(f"Downloading {url}...")

    #Chck if the image is already downloaded
    if (output_dir / unquote(url.split("/")[-1])).exists():
        print(f" → Already exists, skipping.")
        continue

    # Get the redirected URL from Special:FilePath
    r = requests.get(url, headers=headers, allow_redirects=True, timeout=15)
    r.raise_for_status()

    # Determine filename from final URL
    final_url = r.url
    filename = unquote(final_url.split("/")[-1])
    if not filename.lower().endswith(".svg"):
        # Some redirects end in .svg?width=... — strip params
        filename = filename.split("?")[0]

    save_path = output_dir / filename

    save_path.write_bytes(r.content)
    print(f" → Saved to {save_path}")
print("Done!")

