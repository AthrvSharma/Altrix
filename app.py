# app.py
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from inference import MODEL_METADATA, run_inference

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional static serving for local demo so / and /ai-model.html work
try:
    app.mount("/assets", StaticFiles(directory="."), name="assets")
except Exception:
    pass


TelemetryPayload = Union[str, Dict[str, Any], List[Any]]

class TelemetryIn(BaseModel):
    telemetry: TelemetryPayload
    context: Optional[Dict[str, Any]] = None
    conditions: Optional[Dict[str, Any]] = None


@app.post("/api/infer")
def infer(t: TelemetryIn):
    merged_context: Dict[str, Any] = {}
    if t.context:
        merged_context.update(t.context)
    if t.conditions:
        existing_conditions = {}
        if isinstance(merged_context.get("conditions"), dict):
            existing_conditions = merged_context["conditions"]
        merged_context["conditions"] = {**existing_conditions, **t.conditions}
    return run_inference(t.telemetry, context=merged_context or None)


@app.get("/api/model-metadata")
def model_metadata():
    return MODEL_METADATA


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": bool(MODEL_METADATA.get("model_loaded", False)),
        "device": MODEL_METADATA.get("device"),
        "model_path": MODEL_METADATA.get("model_path"),
        "error": MODEL_METADATA.get("model_load_error"),
    }


def _file_or_404(path: str):
    try:
        return FileResponse(path)
    except Exception:
        raise HTTPException(status_code=404, detail=f"File not found: {path}")


@app.get("/")
def serve_root():
    try:
        return FileResponse("index.html")
    except Exception:
        # Fallback minimal index if file not present
        return HTMLResponse("""
            <html><body>
              <h3>Altrix API</h3>
              <ul>
                <li><a href='/index.html'>index.html</a></li>
                <li><a href='/ai-model.html'>ai-model.html</a></li>
                <li><a href='/api/health'>/api/health</a></li>
                <li><a href='/api/model-metadata'>/api/model-metadata</a></li>
              </ul>
            </body></html>
        """)


@app.get("/index.html")
def serve_index():
    return _file_or_404("index.html")


@app.get("/ai-model.html")
def serve_ai_model():
    return _file_or_404("ai-model.html")


@app.get("/Ferrari front.jpg")
def serve_image():
    return _file_or_404("Ferrari front.jpg")
