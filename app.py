# app.py
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
    return {"status": "ok", "model_loaded": True, "device": MODEL_METADATA.get("device")}
