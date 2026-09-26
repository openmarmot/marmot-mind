"""Turn an uploaded or loaded picture into a smaller JPEG a vision model can read.

The long edge is capped. Smaller pictures are not enlarged. Every accepted
upload is re-encoded, so a heavy PNG or a phone photo lands at a similar size.
"""

import io

from PIL import Image, ImageOps, UnidentifiedImageError

# Plenty for a photo, still a modest number of vision tokens.
MAX_EDGE = 1024
JPEG_QUALITY = 85
# Pillow's own bomb guard. A normal photo is far under this.
Image.MAX_IMAGE_PIXELS = 80_000_000


def prepare_for_model(data: bytes) -> tuple[bytes, str]:
    """Return JPEG bytes and the mime type. Raises ValueError if it is not a picture."""
    if not data:
        raise ValueError("empty image")
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Image.DecompressionBombError as e:
        raise ValueError("image is too large to decode") from e
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("image must be png, jpeg, gif, webp, or bmp") from e

    im = ImageOps.exif_transpose(im) or im
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.getchannel("A"))
        im = flat
    else:
        im = im.convert("RGB")

    im.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue(), "image/jpeg"
