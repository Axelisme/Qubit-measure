"""Historical Labber comment decoding through the migration codec."""

import pytest

from migrations.storage_migration.labber_comment import decode_labber_comment


@pytest.mark.parametrize(
    "text", ["operator note", "2024", "[1,2]", '"quoted"', '{"operator":"note"}', ""]
)
def test_comment_decoder_preserves_non_envelope_text(text: str) -> None:
    decoded = decode_labber_comment(text)
    assert decoded.cfg is None
    assert decoded.timestamp is None
    assert decoded.comment == text


@pytest.mark.parametrize(
    "text", ['{"cfg":[]}', '{"cfg":5}', '{"comment":false}', '{"timestamp":42}']
)
def test_comment_decoder_rejects_malformed_envelope_fields(text: str) -> None:
    with pytest.raises(ValueError, match="cfg|comment|timestamp"):
        decode_labber_comment(text)
