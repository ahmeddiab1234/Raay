"""Queue payload codec tests: plain text, JSON-with-id, and the skip rule."""

from raay.inference.batch_consumer import ReviewMessage, decode_message, encode_message


class TestDecodeEncode:
    def test_plain_text_is_used_as_message(self):
        msg = decode_message("الجودة ممتازة")
        assert msg is not None and msg.text == "الجودة ممتازة" and msg.id is None

    def test_json_payload_keeps_id(self):
        msg = decode_message('{"id": "abc", "text": "كلام عربي"}')
        assert msg is not None and msg.text == "كلام عربي" and msg.id == "abc"

    def test_encode_roundtrip_with_and_without_id(self):
        assert encode_message(ReviewMessage(text="نص")) == "نص"
        encoded = encode_message(ReviewMessage(text="نص", id="1"))
        msg = decode_message(encoded)
        assert msg is not None and msg.id == "1" and msg.text == "نص"

    def test_invalid_json_dict_without_text_skipped(self):
        assert decode_message('{"foo": 1}') is None
