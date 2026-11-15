"""Small drawing helpers shared by the scripts that put crops or frames on a sheet."""

from PIL import ImageFont

# Tried in order before Pillow's built-in font. Helvetica is the Mac's; NeSI
# has none of these, so sheets drawn there use the built-in one.
FONT_CANDIDATES = ("/System/Library/Fonts/Helvetica.ttc",)


def load_font(size: int, candidates=FONT_CANDIDATES):
    """A PIL font at ``size``: the first of ``candidates`` that loads, else
    Pillow's built-in font. Never raises for a missing font file, so a sheet
    script runs on any machine. ``candidates=()`` gives the built-in font
    everywhere (the same look on the Mac and on NeSI)."""
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()
