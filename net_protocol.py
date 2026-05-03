# net_protocol.py — shared UDP packet format for Pi ↔ Laptop communication.
#
# FramePacket  (Pi → Laptop)
#   Header: frame_id(u32) | width(u16) | height(u16)  →  8 bytes
#   Body:   JPEG bytes
#
# CoordPacket  (Laptop → Pi)  — 17 bytes, same wire size as before
#   frame_id(u32) | state(u8) | cx(u16) | cy(u16) | bbox_w(u16) | bbox_h(u16) | conf(f32)
#
#   bbox_w / bbox_h replace the old single `area` u32.  Sending exact
#   dimensions lets the Pi draw an accurate overlay box and compute area
#   precisely (w*h) rather than approximating from sqrt(area).

import struct

# ---- frame packet ----
_FRAME_HDR = struct.Struct("!IHH")   # frame_id, width, height
FRAME_HDR_SIZE = _FRAME_HDR.size     # 8

def encode_frame(frame_id: int, jpeg: bytes, width: int, height: int) -> bytes:
    return _FRAME_HDR.pack(frame_id, width, height) + jpeg

def decode_frame(data: bytes):
    """Returns (frame_id, width, height, jpeg_bytes)."""
    frame_id, w, h = _FRAME_HDR.unpack_from(data)
    return frame_id, w, h, data[FRAME_HDR_SIZE:]

# ---- coord packet ----
# frame_id(u32) | state(u8) | cx(u16) | cy(u16) | bbox_w(u16) | bbox_h(u16) | conf(f32)
_COORD = struct.Struct("!IBHHHHf")
COORD_SIZE = _COORD.size   # 17 bytes

STATE_SEARCHING = 0
STATE_TRACKING  = 1
STATE_LOST      = 2

_STATE_TO_STR = {
    STATE_SEARCHING: "searching",
    STATE_TRACKING:  "tracking",
    STATE_LOST:      "lost",
}
_STR_TO_STATE = {v: k for k, v in _STATE_TO_STR.items()}

def encode_coord(frame_id: int, state: int,
                 cx: int, cy: int,
                 bbox_w: int, bbox_h: int,
                 confidence: float) -> bytes:
    return _COORD.pack(frame_id, state, cx, cy, bbox_w, bbox_h, confidence)

def decode_coord(data: bytes):
    """Returns (frame_id, state_int, cx, cy, bbox_w, bbox_h, confidence)."""
    return _COORD.unpack_from(data)

def state_str(state_int: int) -> str:
    return _STATE_TO_STR.get(state_int, "searching")
