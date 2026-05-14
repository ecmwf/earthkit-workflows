from cascade.ygg.protocol import Ack, Syn, parse_envelope, serialize_ack, serialize_syn


def test_protocol_roundtrip() -> None:
    syn = Syn(idx=42, ack_address="tcp://127.0.0.1:1234")
    ack = Ack(idx=42)
    assert parse_envelope(serialize_syn(syn)) == syn
    assert parse_envelope(serialize_ack(ack)) == ack


def test_protocol_non_envelope_frame_returns_none() -> None:
    assert parse_envelope(b"raw-payload") is None


def test_protocol_malformed_frame_returns_none() -> None:
    assert parse_envelope(b"YGG1A\x00") is None
    assert parse_envelope(b"YGG1S") is None
    assert parse_envelope(b"YGG1X\x00\x00\x00\x00\x00\x00\x00\x00") is None
