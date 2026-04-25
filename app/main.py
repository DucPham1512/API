import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import GRAYSCALE, MODEL_PATH, STRIDE, TARGET_FPS, WINDOW_FRAMES
from app.inferencer import OnnxInferencer
from app.interface.schemas import PredictionResponse
from app.preprocessor.pipeline import VSRPreprocessingPipeline

app = FastAPI()

inferencer = OnnxInferencer(MODEL_PATH)


@app.post("/vsr/process", response_model=PredictionResponse)
async def process_video(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        pipeline = VSRPreprocessingPipeline(
            source=tmp_path,
            target_fps=TARGET_FPS,
            window_frames=WINDOW_FRAMES,
            stride=STRIDE,
            grayscale=GRAYSCALE,
        )
        tensors = pipeline.run()
    finally:
        os.unlink(tmp_path)

    if not tensors:
        raise HTTPException(status_code=422, detail="No lip windows detected in video.")

    predictions = [inferencer.predict(t).flatten().tolist() for t in tensors]
    return PredictionResponse(predictions=predictions)

@app.get("/vsr/transcript")
async def get_transcript():
    # Placeholder for transcript retrieval logic
    return {"message": "Transcript retrieval started"}