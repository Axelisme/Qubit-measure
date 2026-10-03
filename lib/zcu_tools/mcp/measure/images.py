"""Validate GUI PNG bytes before session persistence or MCP image delivery."""

from io import BytesIO

from PIL import Image

from zcu_tools.mcp.core.reply import PngImage


def validated_png(png: bytes) -> PngImage:
    # Decoders can accept a truncated trailer; require the complete PNG end chunk.
    if not png.endswith(b"\x00\x00\x00\x00IEND\xaeB\x60\x82"):
        raise ValueError("Invalid PNG image: missing or truncated IEND chunk")
    try:
        with Image.open(BytesIO(png), formats=["PNG"]) as image:
            image.verify()
        # verify checks chunks and checksums, not whether the pixels can decode.
        with Image.open(BytesIO(png), formats=["PNG"]) as image:
            image.load()
    except (OSError, SyntaxError, ValueError) as exc:
        raise ValueError("Invalid PNG image") from exc
    return PngImage(png)
