"""
main.py - FastAPI backend for Telugu-English code-mixed sentiment analysis.
"""
import os
import sys
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, field_validator

# Allow importing from src/
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import MAX_TEXT_LENGTH

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Model singleton ────────────────────────────────────────────────────────────
_model = None


def get_model():
    global _model
    if _model is None:
        from predict import SentimentModel
        _model = SentimentModel()
        log.info("Model loaded successfully.")
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_model()
    except Exception as e:
        log.error("Failed to load model at startup: %s", e)
    yield


app = FastAPI(
    title="CMTet Sentiment Analyzer",
    description="Telugu-English Code-Mixed Sentiment Analysis API",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("text must be non-empty")
        if len(v) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"text must be at most {MAX_TEXT_LENGTH} characters "
                f"(got {len(v)})"
            )
        return v


class PredictResponse(BaseModel):
    label:   str
    display: str
    probs:   dict
    cleaned: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        model = get_model()
        result = model.predict(req.text)
        return PredictResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.error("Prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ── Static files & frontend ────────────────────────────────────────────────────
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(str(static_dir / "index.html"))


# ── Dev runner ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
