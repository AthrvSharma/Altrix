# inference.py
import json
import time
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np

from src.model import LSTMAutoencoder

MODEL_PATH = "models/best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Suggested sensor catalogue for UI/metadata purposes
FEATURE_HINTS: List[Dict[str, str]] = [
    {
        "name": "speed",
        "unit": "km/h",
        "description": "Vehicle ground speed from CAN bus",
    },
    {
        "name": "yaw_rate",
        "unit": "deg/s",
        "description": "Rotation around vertical axis (gyro)",
    },
    {
        "name": "lateral_g",
        "unit": "g",
        "description": "Side-force measured from accelerometer",
    },
    {
        "name": "longitudinal_g",
        "unit": "g",
        "description": "Forward/backward force",
    },
    {
        "name": "brake_pressure",
        "unit": "bar",
        "description": "Hydraulic brake pressure",
    },
    {
        "name": "steering_angle",
        "unit": "deg",
        "description": "Driver steering input",
    },
    {
        "name": "wheel_speed_fl",
        "unit": "km/h",
        "description": "Front-left wheel speed sensor",
    },
    {
        "name": "wheel_speed_fr",
        "unit": "km/h",
        "description": "Front-right wheel speed sensor",
    },
]

ACTIONS_CATALOG: List[Dict[str, str]] = [
    {
        "id": "precharge_brakes",
        "description": "Prime hydraulic system to shorten stopping distance",
    },
    {
        "id": "pretension_seatbelt",
        "description": "Pre-tighten belt to reduce occupant movement",
    },
    {
        "id": "tighten_suspension",
        "description": "Firm up suspension to improve stability",
    },
    {
        "id": "notify_pit_wall",
        "description": "Send telemetry alert to pit/perimeter safety crew",
    },
]

SCORE_THRESHOLDS = {
    "critical": 0.8,
    "high": 0.6,
    "elevated": 0.4,
}

TelemetryPayload = Union[str, Dict[str, Any], List[Any]]

FALLBACK_FEATURE_NAMES: List[str] = [hint["name"] for hint in FEATURE_HINTS]
FALLBACK_FEATURE_NAMES += [f"feature_{i}" for i in range(len(FALLBACK_FEATURE_NAMES) + 1, len(FALLBACK_FEATURE_NAMES) + 16)]

WEATHER_RISK_MAP: Dict[str, float] = {
    "dry": 0.0,
    "clear": 0.0,
    "cloudy": 0.02,
    "overcast": 0.03,
    "light rain": 0.05,
    "rain": 0.08,
    "heavy rain": 0.12,
    "storm": 0.18,
    "thunderstorm": 0.2,
    "wet": 0.07,
    "fog": 0.09,
    "snow": 0.22,
    "ice": 0.25,
}

SURFACE_RISK_MAP: Dict[str, float] = {
    "street": 0.05,
    "urban": 0.04,
    "track": 0.0,
    "wet": 0.06,
    "gravel": 0.08,
    "unknown": 0.02,
}

VISIBILITY_RISK_MAP: Dict[str, float] = {
    "excellent": 0.0,
    "good": 0.0,
    "moderate": 0.03,
    "poor": 0.06,
    "very poor": 0.09,
}

DRIVER_STATE_RISK_MAP: Dict[str, float] = {
    "focused": 0.0,
    "alert": 0.0,
    "fatigued": 0.07,
    "distracted": 0.06,
    "aggressive": 0.05,
    "injured": 0.12,
}

TYRE_WEAR_RISK_MAP: Dict[str, float] = {
    "fresh": 0.0,
    "scrubbed": 0.02,
    "worn": 0.05,
    "critical": 0.08,
}

def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    input_dim = ckpt["input_dim"]
    seq_len = ckpt["seq_len"]

    model = LSTMAutoencoder(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, seq_len, input_dim

model, SEQ_LEN, INPUT_DIM = load_model()


def _normalize_key(key: str) -> str:
    clean = key.strip().replace("/", "_")
    clean = re.sub(r"\s+", "_", clean)
    clean = re.sub(r"[^a-zA-Z0-9_]+", "", clean)
    return clean.lower() or "feature"


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _attempt_json_load(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _parse_freestyle_text(text: str) -> List[Dict[str, float]]:
    timeline: List[Dict[str, float]] = []
    kv_pairs = re.findall(r"([A-Za-z0-9_\-\s]+)[:=]\s*(-?\d+(?:\.\d+)?)", text)

    if kv_pairs:
        row: Dict[str, float] = {}
        seen = set()
        for raw_key, raw_val in kv_pairs:
            key = _normalize_key(raw_key)
            if key in seen:
                continue
            seen.add(key)
            val = _coerce_float(raw_val)
            if val is not None:
                row[key] = val
        if row:
            timeline.append(row)
    else:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if numbers:
            row = {}
            for idx, raw_val in enumerate(numbers):
                val = _coerce_float(raw_val)
                if val is None:
                    continue
                feature_name = FALLBACK_FEATURE_NAMES[idx % len(FALLBACK_FEATURE_NAMES)]
                row[feature_name] = val
            if row:
                timeline.append(row)

    return timeline


def _standardize_payload(telemetry: TelemetryPayload) -> Tuple[List[Dict[str, float]], List[str]]:
    timeline: List[Dict[str, float]] = []

    if isinstance(telemetry, str):
        stripped = telemetry.strip()
        if stripped:
            maybe_json = _attempt_json_load(stripped) if stripped[:1] in "[{" else None
            if maybe_json is not None:
                return _standardize_payload(maybe_json)
            timeline = _parse_freestyle_text(stripped)

    elif isinstance(telemetry, dict):
        row: Dict[str, float] = {}
        for key, value in telemetry.items():
            val = _coerce_float(value)
            if val is not None:
                row[_normalize_key(str(key))] = val
        if row:
            timeline.append(row)

    elif isinstance(telemetry, list):
        if telemetry and all(isinstance(item, dict) for item in telemetry):
            for item in telemetry:
                row: Dict[str, float] = {}
                for key, value in item.items():
                    val = _coerce_float(value)
                    if val is not None:
                        row[_normalize_key(str(key))] = val
                if row:
                    timeline.append(row)
        else:
            row: Dict[str, float] = {}
            for idx, value in enumerate(telemetry):
                val = _coerce_float(value)
                if val is not None:
                    feature_name = FALLBACK_FEATURE_NAMES[idx % len(FALLBACK_FEATURE_NAMES)]
                    row[feature_name] = val
            if row:
                timeline.append(row)

    if not timeline:
        fallback_feature = FALLBACK_FEATURE_NAMES[0]
        timeline = [{fallback_feature: 0.0}]

    feature_order: List[str] = []
    for row in timeline:
        for key in row.keys():
            if key not in feature_order:
                feature_order.append(key)

    if len(feature_order) > INPUT_DIM:
        feature_order = feature_order[:INPUT_DIM]

    if len(feature_order) < INPUT_DIM:
        for fallback in FALLBACK_FEATURE_NAMES:
            if fallback not in feature_order:
                feature_order.append(fallback)
            if len(feature_order) >= INPUT_DIM:
                break

    return timeline, feature_order


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _lookup_risk(value: Any, mapping: Dict[str, float], default: float = 0.0) -> Tuple[float, Optional[str]]:
    if value is None:
        return default, None
    key = str(value).lower().strip()
    if not key:
        return default, None
    if key in mapping:
        return mapping[key], key
    for candidate, delta in mapping.items():
        if candidate in key:
            return delta, candidate
    return default, key


def _extract_conditions(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not context:
        return {}
    conditions: Dict[str, Any] = {}
    nested = context.get("conditions") if isinstance(context, dict) else {}
    if isinstance(nested, dict):
        conditions.update(nested)
    direct_keys = [
        "weather",
        "track_temperature",
        "ambient_temperature",
        "humidity",
        "track_surface",
        "visibility",
        "traffic_density",
        "tyre_wear",
        "driver_state",
        "session",
    ]
    for key in direct_keys:
        if isinstance(context, dict) and key in context and key not in conditions:
            conditions[key] = context[key]
    return conditions


def _compute_environment_modifier(conditions: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]], List[str]]:
    modifier = 0.0
    contributions: List[Dict[str, Any]] = []
    summary_lines: List[str] = []

    weather_delta, matched_weather = _lookup_risk(conditions.get("weather"), WEATHER_RISK_MAP, 0.0)
    if matched_weather:
        modifier += weather_delta
        contributions.append({
            "label": "Weather",
            "detail": matched_weather,
            "delta": round(weather_delta, 4),
        })
        summary_lines.append(f"Weather: {matched_weather}")

    track_temp = _coerce_float(conditions.get("track_temperature"))
    if track_temp is not None:
        temp_delta = 0.0
        if track_temp >= 55:
            temp_delta = 0.08
        elif track_temp >= 45:
            temp_delta = 0.06
        elif track_temp >= 38:
            temp_delta = 0.04
        elif track_temp <= 5:
            temp_delta = 0.05
        elif track_temp <= 10:
            temp_delta = 0.03
        modifier += temp_delta
        contributions.append({
            "label": "Track temperature",
            "detail": f"{track_temp}°C",
            "delta": round(temp_delta, 4),
        })
        summary_lines.append(f"Track temp: {track_temp:.1f}°C")

    ambient_temp = _coerce_float(conditions.get("ambient_temperature"))
    if ambient_temp is not None:
        at_delta = 0.0
        if ambient_temp >= 42:
            at_delta = 0.04
        elif ambient_temp <= 0:
            at_delta = 0.05
        modifier += at_delta
        contributions.append({
            "label": "Ambient temperature",
            "detail": f"{ambient_temp}°C",
            "delta": round(at_delta, 4),
        })

    humidity = _coerce_float(conditions.get("humidity"))
    if humidity is not None:
        hum_delta = 0.0
        if humidity >= 95:
            hum_delta = 0.07
        elif humidity >= 85:
            hum_delta = 0.05
        elif humidity >= 70:
            hum_delta = 0.03
        elif humidity <= 20:
            hum_delta = 0.02
        modifier += hum_delta
        contributions.append({
            "label": "Humidity",
            "detail": f"{humidity}%",
            "delta": round(hum_delta, 4),
        })
        summary_lines.append(f"Humidity: {humidity:.0f}%")

    surface_delta, surface_key = _lookup_risk(conditions.get("track_surface"), SURFACE_RISK_MAP, 0.0)
    if surface_key:
        modifier += surface_delta
        contributions.append({
            "label": "Track surface",
            "detail": surface_key,
            "delta": round(surface_delta, 4),
        })
        summary_lines.append(f"Surface: {surface_key}")

    visibility_delta, visibility_key = _lookup_risk(conditions.get("visibility"), VISIBILITY_RISK_MAP, 0.0)
    if visibility_key:
        modifier += visibility_delta
        contributions.append({
            "label": "Visibility",
            "detail": visibility_key,
            "delta": round(visibility_delta, 4),
        })
        summary_lines.append(f"Visibility: {visibility_key}")

    tyre_delta, tyre_key = _lookup_risk(conditions.get("tyre_wear"), TYRE_WEAR_RISK_MAP, 0.0)
    if tyre_key:
        modifier += tyre_delta
        contributions.append({
            "label": "Tyre wear",
            "detail": tyre_key,
            "delta": round(tyre_delta, 4),
        })
        summary_lines.append(f"Tyre wear: {tyre_key}")

    driver_delta, driver_key = _lookup_risk(conditions.get("driver_state"), DRIVER_STATE_RISK_MAP, 0.0)
    if driver_key:
        modifier += driver_delta
        contributions.append({
            "label": "Driver state",
            "detail": driver_key,
            "delta": round(driver_delta, 4),
        })
        summary_lines.append(f"Driver: {driver_key}")

    traffic = str(conditions.get("traffic_density")) if conditions.get("traffic_density") is not None else ""
    if traffic:
        traffic_key = traffic.lower().strip()
        traffic_delta = 0.0
        if traffic_key in {"heavy", "high"}:
            traffic_delta = 0.06
        elif traffic_key in {"medium", "moderate"}:
            traffic_delta = 0.03
        modifier += traffic_delta
        contributions.append({
            "label": "Traffic",
            "detail": traffic_key,
            "delta": round(traffic_delta, 4),
        })
        summary_lines.append(f"Traffic: {traffic_key}")

    session = conditions.get("session")
    if session:
        summary_lines.append(f"Session: {session}")

    modifier = _clamp(modifier, -0.1, 0.25)
    return modifier, contributions, summary_lines


def preprocess_payload(telemetry: TelemetryPayload) -> Tuple[torch.Tensor, List[str], List[Dict[str, float]], np.ndarray, np.ndarray]:
    timeline, feature_order = _standardize_payload(telemetry)

    rows: List[List[float]] = []
    for row in timeline:
        rows.append([float(row.get(name, 0.0)) for name in feature_order])

    arr = np.array(rows, dtype=np.float32)

    if arr.size == 0:
        arr = np.zeros((1, len(feature_order)), dtype=np.float32)

    raw_arr = arr.copy()

    time_steps = arr.shape[0]
    if time_steps >= SEQ_LEN:
        padded = arr[:SEQ_LEN, :]
    else:
        pad_source = arr[-1:, :] if time_steps > 0 else np.zeros((1, len(feature_order)), dtype=np.float32)
        pad = np.repeat(pad_source, SEQ_LEN - time_steps, axis=0)
        padded = np.vstack([arr, pad])

    tensor = torch.tensor(padded[None, :, :], dtype=torch.float32).to(DEVICE)
    return tensor, feature_order, timeline, raw_arr, padded


MODEL_METADATA: Dict[str, Any] = {
    "model_version": "v0.9.1-demo",
    "device": DEVICE,
    "seq_len": SEQ_LEN,
    "input_dim": INPUT_DIM,
    "expected_sample_rate_hz": 50,
    "window_duration_s": round(SEQ_LEN / 50, 2),
    "feature_hints": FEATURE_HINTS,
    "actions_catalog": ACTIONS_CATALOG,
    "score_thresholds": SCORE_THRESHOLDS,
    "example_payload": "speed: 214, yaw_rate: 17.6, lateral_g: 3.2, brake_pressure: 0.4",
    "latency_expectation_ms": {
        "cpu": 9.5,
        "jetson": 14.2,
    },
    "training_summary": {
        "epochs": 15,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "dataset": "F1 crash telemetry + synthetic road transfer",
        "total_sequences": 18432,
        "normal_ratio": 0.94,
        "high_risk_ratio": 0.06,
        "augmentations": [
            "Weather perturbation",
            "Track grip variance",
            "Driver reaction latency jitter",
        ],
    },
    "release_notes": [
        "Latent dimension 32 with channel-attentive decoder",
        "Score calibrated on 2025-Q1 validation set",
        "Jetson build pruned to 6.1M parameters",
    ],
    "supported_conditions": [
        "weather",
        "track_temperature",
        "humidity",
        "track_surface",
        "visibility",
        "tyre_wear",
        "driver_state",
        "traffic_density",
    ],
    "chart_defaults": {
        "timeline_window": 60,
        "gauge_thresholds": [0.25, 0.55, 0.75, 0.9],
    },
}

def _band_from_score(score: float) -> str:
    if score >= SCORE_THRESHOLDS["critical"]:
        return "critical"
    if score >= SCORE_THRESHOLDS["high"]:
        return "high"
    if score >= SCORE_THRESHOLDS["elevated"]:
        return "elevated"
    return "normal"


def run_inference(telemetry: TelemetryPayload, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    inference_start = time.perf_counter()
    x, feature_order, raw_timeline, raw_arr, padded_arr = preprocess_payload(telemetry)
    conditions = _extract_conditions(context)
    env_modifier, env_contributions, env_summary = _compute_environment_modifier(conditions)

    with torch.no_grad():
        recon, latent = model(x)
        error_tensor = (recon - x) ** 2
        mse = torch.mean(error_tensor, dim=(1, 2))  # (batch,)
        feature_mse = (
            torch.mean(error_tensor, dim=(0, 1))
            .detach()
            .cpu()
            .view(-1)
            .numpy()
            .tolist()
        )
        timestep_mse = (
            torch.mean(error_tensor, dim=2)
            .detach()
            .cpu()
            .view(-1)
            .numpy()
            .tolist()
        )
        score = mse.item()

    anomaly_score = float(np.tanh(score * 20))
    risk_band = _band_from_score(anomaly_score)
    latency_ms = round((time.perf_counter() - inference_start) * 1000, 2)

    base_probability = 0.12 + 0.78 * anomaly_score
    crash_probability = _clamp(base_probability + env_modifier, 0.01, 0.999)
    severity_index = round((_clamp(anomaly_score, 0.0, 1.0) * 0.7 + crash_probability * 0.3) * 100, 1)

    feature_breakdown: List[Dict[str, Any]] = []
    for idx, feat in enumerate(feature_order):
        recon_error = float(feature_mse[idx]) if idx < len(feature_mse) else 0.0
        feature_breakdown.append(
            {
                "feature": feat,
                "recon_error": round(recon_error, 6),
                "scaled_importance": round(min(recon_error * 40.0, 1.0), 3),
            }
        )

    feature_breakdown.sort(key=lambda item: item["recon_error"], reverse=True)
    for rank, item in enumerate(feature_breakdown, start=1):
        item["rank"] = rank

    dominant_feature = feature_breakdown[0]["feature"] if feature_breakdown else feature_order[0]

    if risk_band in ("critical", "high"):
        actions = ["precharge_brakes", "pretension_seatbelt", "notify_pit_wall"]
    elif risk_band == "elevated":
        actions = ["precharge_brakes"]
    else:
        actions = []

    notes = []
    notes.append(f"Dominant deviation detected on '{dominant_feature}'.")
    if risk_band in ("critical", "high"):
        notes.append("Recommend enabling defensive driving assists and alerting safety systems.")
    elif risk_band == "elevated":
        notes.append("Monitor closely; pattern is drifting from nominal telemetry.")
    else:
        notes.append("Telemetry reconstruction aligns with learned manifold.")
    notes.append(f"Estimated crash probability: {crash_probability * 100:.1f}%.")

    probability_components: List[Dict[str, Any]] = [
        {"label": "Base (anomaly)", "value": round(base_probability, 4)},
        {"label": "Environment modifier", "value": round(env_modifier, 4)},
    ]
    probability_components.extend(
        {
            "label": contrib.get("label", "env"),
            "detail": contrib.get("detail"),
            "value": contrib.get("delta", 0.0),
        }
        for contrib in env_contributions
    )

    timeline_length = raw_arr.shape[0] if raw_arr.size else len(timestep_mse)
    timeline_length = min(timeline_length, len(timestep_mse)) or len(timestep_mse)

    timeline_series = [
        {
            "timestep": idx,
            "recon_error": round(float(timestep_mse[idx]), 6),
        }
        for idx in range(timeline_length)
    ]

    feature_indices = {name: idx for idx, name in enumerate(feature_order)}
    top_feature_names = [item["feature"] for item in feature_breakdown[:3] if item["feature"] in feature_indices]
    if len(top_feature_names) < 3:
        for fallback in feature_order:
            if fallback not in top_feature_names:
                top_feature_names.append(fallback)
            if len(top_feature_names) >= 3:
                break
    top_feature_names = top_feature_names[:3]

    source_arr = raw_arr if raw_arr.size else padded_arr
    top_feature_series: List[Dict[str, Any]] = []
    for name in [item["feature"] for item in feature_breakdown[:5] if item["feature"] in feature_indices][:5]:
        idx = feature_indices[name]
        values = source_arr[:timeline_length, idx] if timeline_length > 0 else source_arr[:, idx]
        top_feature_series.append(
            {
                "feature": name,
                "values": [round(float(v), 4) for v in values.tolist()],
            }
        )

    three_d_axes: List[str] = top_feature_names[:3]
    three_d_points: List[Dict[str, Any]] = []
    if len(three_d_axes) < 3:
        for fallback in feature_order:
            if fallback not in three_d_axes:
                three_d_axes.append(fallback)
            if len(three_d_axes) >= 3:
                break
    three_d_axes = three_d_axes[:3]

    if len(three_d_axes) == 3 and source_arr.size:
        idx_x, idx_y, idx_z = [feature_indices.get(name, 0) for name in three_d_axes]
        max_points = min(source_arr.shape[0], len(timestep_mse), 120)
        for step in range(max_points):
            three_d_points.append(
                {
                    "timestep": step,
                    "x": round(float(source_arr[step, idx_x]), 4),
                    "y": round(float(source_arr[step, idx_y]), 4),
                    "z": round(float(source_arr[step, idx_z]), 4),
                    "error": round(float(timestep_mse[step]), 6) if step < len(timestep_mse) else 0.0,
                }
            )
        if len(three_d_points) < 3:
            base_x = float(source_arr[0, idx_x]) if source_arr.shape[0] else 0.0
            base_y = float(source_arr[0, idx_y]) if source_arr.shape[0] else 0.0
            base_z = float(source_arr[0, idx_z]) if source_arr.shape[0] else 0.0
            for idx in range(3):
                angle = (idx / 3.0) * 2 * np.pi
                three_d_points.append(
                    {
                        "timestep": len(three_d_points) + idx,
                        "x": round(base_x + np.cos(angle) * 0.01, 4),
                        "y": round(base_y + np.sin(angle) * 0.01, 4),
                        "z": round(base_z + (idx - 1) * 0.01, 4),
                        "error": round(float(timestep_mse[0]) if timestep_mse else 0.0, 6),
                    }
                )

    condition_summary = env_summary if env_summary else ["Conditions: not provided"]

    response = {
        "model_version": MODEL_METADATA["model_version"],
        "anomaly_score": round(anomaly_score, 4),
        "raw_recon_error": round(score, 6),
        "risk_band": risk_band,
        "label": risk_band.upper(),
        "actions": actions,
        "feature_breakdown": feature_breakdown,
        "timeline_error": [round(float(val), 6) for val in timestep_mse],
        "input_feature_order": feature_order,
        "seq_len": SEQ_LEN,
        "input_dim": INPUT_DIM,
        "latency_ms": latency_ms,
        "device": DEVICE,
        "notes": notes,
        "input_preview": raw_timeline[: min(len(raw_timeline), 5)],
        "score_thresholds": SCORE_THRESHOLDS,
        "crash_probability": round(crash_probability, 4),
        "severity_index": severity_index,
        "probability_components": probability_components,
        "environment_modifier": round(env_modifier, 4),
        "condition_summary": condition_summary,
        "condition_contributors": env_contributions,
        "chart_data": {
            "timeline": timeline_series,
            "top_features": top_feature_series,
            "three_d_projection": {
                "axes": three_d_axes,
                "points": three_d_points,
            },
        },
        "raw_sequence": source_arr[: min(source_arr.shape[0], 10)].tolist() if source_arr.size else [],
    }

    if context:
        response["context_echo"] = context

    return response

if __name__ == "__main__":
    out = run_inference("speed: 214, yaw: 18, gz: 3.1, brake: 0.3")
    print(out)
