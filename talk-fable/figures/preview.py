"""Stack a layered figure back into one picture, to check it without opening the slides.

    uv run --with pillow python talk-fable/figures/preview.py bounds /tmp/bounds.png
"""
import sys
from pathlib import Path
from PIL import Image

name, out = sys.argv[1], sys.argv[2]
layers = sorted(Path(__file__).parent.parent.glob(f"public/fig/{name}-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
picture = Image.new("RGBA", Image.open(layers[0]).size, "white")
for layer in layers:
    picture.alpha_composite(Image.open(layer).convert("RGBA"))
picture.convert("RGB").save(out)
