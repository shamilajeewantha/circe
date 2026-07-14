"""Normalize a folder of downloaded images for prediction.

- converts everything (webp/png/jpeg/...) to plain RGB .jpg
- strips alpha / palette / EXIF-rotation issues
- renames sequentially:  <PREFIX>_0001.jpg, _0002.jpg, ...
- writes into a fresh output folder (source left untouched)

Run:  python prep_images.py
"""
from pathlib import Path
from PIL import Image, ImageOps

# ---- settings (edit these) --------------------------------------
SRC    = "sample_downloaded_nuts"        # folder of raw downloads
DST    = "sample_downloaded_nuts_clean"  # new output folder
PREFIX = "nut"                           # output filename prefix
QUALITY = 92                             # jpeg quality
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
# -----------------------------------------------------------------

HERE = Path(__file__).parent


def main():
    src = HERE / SRC
    dst = HERE / DST
    dst.mkdir(exist_ok=True)

    files = sorted(p for p in src.iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXTS)

    ok, bad = 0, []
    for i, f in enumerate(files, 1):
        try:
            img = Image.open(f)
            img = ImageOps.exif_transpose(img)      # honor camera rotation
            img = img.convert("RGB")                # drop alpha/palette
            out = dst / f"{PREFIX}_{i:04d}.jpg"
            img.save(out, "JPEG", quality=QUALITY)
            ok += 1
        except Exception as e:
            bad.append(f"{f.name}: {e}")

    print(f"converted {ok}/{len(files)} -> {dst}")
    if bad:
        print(f"skipped {len(bad)} unreadable:")
        for b in bad:
            print("  " + b)


if __name__ == "__main__":
    main()
