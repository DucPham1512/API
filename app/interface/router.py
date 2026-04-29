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

import asyncio
import json
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from app.interface.schemas import StreamSegment, TranscriptResponse
from app.pipeline import VSRPipeline

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
        pipeline = VSRPipeline.get()
        segments = await asyncio.to_thread(pipeline.process_file, tmp_path)
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
    processor = VSRPipeline.get().make_stream_processor()

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                segments = await asyncio.to_thread(processor.push_frame, message["bytes"])
                for seg in segments:
                    await websocket.send_text(seg.model_dump_json())

            elif "text" in message and message["text"]:
                data = json.loads(message["text"])
                msg_type = data.get("type")

                if msg_type == "config":
                    processor.configure(fps=data.get("fps", 25.0))

                elif msg_type == "end":
                    final_segments = await asyncio.to_thread(processor.flush)
                    for seg in final_segments:
                        await websocket.send_text(seg.model_dump_json())
                    sentinel = StreamSegment(text="", start_ms=0, end_ms=0, is_final=True)
                    await websocket.send_text(sentinel.model_dump_json())
                    break

    except WebSocketDisconnect:
        pass
    finally:
        processor.close()
