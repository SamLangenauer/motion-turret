# net_protocol.py — shared UDP packet format for Pi ↔ Laptop communication.
#
# Two packet types:
#
#   FramePacket  (Pi → Laptop)
#     Header: frame_id(u32) | width(u16) | height(u16)  →  8 bytes
#     Body:   JPEG-compressed frame bytes
#     Max size fits in a single UDP datagram (~65 KB).
#     At JPEG quality 40, 680×480 is typically 15–35 KB — well within limits.
#
#   CoordPacket  (Laptop → Pi)
#     18 bytes total — tiny.
#     frame_id(u32) | state(u8) | cx(u16) | cy(u16) | area(u32) | conf(f32)
#
# State constants match MotionTracker.STATE_* strings for easy bridging.

import struct

# ---- frame packet ----
_FRAME_HDR = struct.Struct("!IHH")   # frame_id, width, height
FRAME_HDR_SIZE = _FRAME_HDR.size     # 8

def encode_frame(frame_id: int, jpeg: bytes, width: int, height: int) -> bytes:
    return _FRAME_HDR.pack(frame_id, width, height) + jpeg

def decode_frame(data: bytes):
    """Returns (frame_id, width, height, jpeg_bytes) or raises struct.error."""
    frame_id, w, h = _FRAME_HDR.unpack_from(data)
    return frame_id, w, h, data[FRAME_HDR_SIZE:]

# ---- coord packet ----
_COORD = struct.Struct("!IBHHIf")    # frame_id, state, cx, cy, area, confidence
COORD_SIZE = _COORD.size             # 17  (padded to 18 by some impls — use SIZE)

# State byte values
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
                 cx: int, cy: int, area: int, confidence: float) -> bytes:
    return _COORD.pack(frame_id, state, cx, cy, area, confidence)

def decode_coord(data: bytes):
    """
    Returns (frame_id, state_int, cx, cy, area, confidence).
    state_int is one of STATE_SEARCHING / STATE_TRACKING / STATE_LOST.
    """
    return _COORD.unpack_from(data)

def state_str(state_int: int) -> str:
    return _STATE_TO_STR.get(state_int, "searching")
