"""
API route definitions.

Two input modes:
  POST /vsr/file      — upload an MP4; returns a full timestamped transcript.
  WS   /vsr/stream    — send JPEG frames over WebSocket; receive transcript segments in real time.

WebSocket protocol (client → server):
  - Text message  {"type": "config", "fps": 25}          — optional, before first frame
  - Binary message <JPEG bytes>                           — one frame per message
  - Text message  {"type": "end"}                         — signals no more frames

WebSocket protocol (server → client):
  - Text message  <StreamSegment JSON>                    — emitted per recognised window
  - Text message  <StreamSegment JSON, is_final=true>     — last message, signals completion
"""

import json
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from app.interface.schemas import StreamSegment, TranscriptResponse

router = APIRouter(prefix="/vsr", tags=["VSR"])


@router.post("/file", response_model=TranscriptResponse)
async def process_file(file: UploadFile = File(...)):
    """Accept an MP4 upload and return a timestamped transcript.

    Each TranscriptSegment carries the recognised text and the time range
    (start_ms / end_ms) within the original video that the segment covers.
    """
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=415, detail="Only video files are accepted.")

    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        segments = _run_file_pipeline(tmp_path)
    finally:
        os.unlink(tmp_path)

    total_ms = segments[-1].end_ms if segments else None
    return TranscriptResponse(segments=segments, total_duration_ms=total_ms)


@router.websocket("/stream")
async def stream_video(websocket: WebSocket):
    """Process a live video stream frame-by-frame.

    The client sends JPEG-encoded frames as binary WebSocket messages.
    The server emits StreamSegment JSON text messages for each recognised
    window.  A final message with is_final=True signals that processing is
    complete (sent after the client sends {"type": "end"}).
    """
    await websocket.accept()
    processor = _StreamProcessor()

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                segments = processor.push_frame(message["bytes"])
                for seg in segments:
                    await websocket.send_text(seg.model_dump_json())

            elif "text" in message and message["text"]:
                data = json.loads(message["text"])
                msg_type = data.get("type")

                if msg_type == "config":
                    processor.configure(fps=data.get("fps", 25.0))

                elif msg_type == "end":
                    final_segments = processor.flush()
                    for seg in final_segments:
                        await websocket.send_text(seg.model_dump_json())
                    sentinel = StreamSegment(text="", start_ms=0, end_ms=0, is_final=True)
                    await websocket.send_text(sentinel.model_dump_json())
                    break

    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# Stubs — replaced in the next phase when the processing pipeline is wired in
# ---------------------------------------------------------------------------

def _run_file_pipeline(video_path: str) -> list:
    """Process an MP4 file and return a list of TranscriptSegment objects.

    Stub: replace with real pipeline + decoder logic.
    """
    from app.interface.schemas import TranscriptSegment
    return [TranscriptSegment(text="[pipeline not yet connected]", start_ms=0.0, end_ms=0.0)]


class _StreamProcessor:
    """Stateful per-connection stream processor.

    Stub: replace with real frame buffering + pipeline + decoder logic.
    """

    def __init__(self):
        self._fps: float = 25.0
        self._frame_count: int = 0

    def configure(self, fps: float):
        self._fps = fps

    def push_frame(self, jpeg_bytes: bytes) -> list[StreamSegment]:
        self._frame_count += 1
        # TODO: decode JPEG, run through pipeline window, decode logits → text
        return []

    def flush(self) -> list[StreamSegment]:
        # TODO: flush remaining frames in the pipeline buffer
        return []
