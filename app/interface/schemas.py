from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    text: str
    start_ms: float
    end_ms: float


class TranscriptResponse(BaseModel):
    segments: list[TranscriptSegment]
    total_duration_ms: float | None = None


class StreamSegment(BaseModel):
    text: str
    start_ms: float
    end_ms: float
    is_final: bool = False
