from pathlib import Path
import pandas as pd

def build_image_manifest(image_dir: str, suffixes=(".jpg", ".jpeg", ".png", ".webp")):
    image_dir = Path(image_dir)
    rows = []
    for p in sorted(image_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in suffixes:
            rows.append({"image_id": p.stem, "file_path": str(p.resolve()), "suffix": p.suffix.lower()})
    return pd.DataFrame(rows)
