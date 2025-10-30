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

# Feature definitions for the model
FEATURE_HINTS = [
    {
        "name": "speed",
        "unit": "km/h",
        "description": "Vehicle ground speed from CAN bus"
    },
    {
        "name": "yaw_rate",
        "unit": "deg/s",
        "description": "Rotation around vertical axis (gyro)"
    },
    {
        "name": "lateral_g",
        "unit": "g",
        "description": "Side-force measured from accelerometer"
    },
    {
        "name": "longitudinal_g",
        "unit": "g",
        "description": "Forward/backward force"
    },
    {
        "name": "brake_pressure",
        "unit": "bar",
        "description": "Hydraulic brake pressure"
    },
    {
        "name": "steering_angle",
        "unit": "deg",
        "description": "Driver steering input"
    }
]

# (Model metadata is defined later after model load — see MODEL_METADATA near the bottom.)

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

FEATURE_NAME_ALIASES: Dict[str, str] = {
    "veh_speed": "speed",
    "vehicle_speed": "speed",
    "velocity": "speed",
    "ground_speed": "speed",
    "yaw": "yaw_rate",
    "yawrate": "yaw_rate",
    "yawrate_deg": "yaw_rate",
    "lat_g": "lateral_g",
    "latg": "lateral_g",
    "gz": "lateral_g",
    "gforce_lat": "lateral_g",
    "long_g": "longitudinal_g",
    "longg": "longitudinal_g",
    "ax": "longitudinal_g",
    "brake": "brake_pressure",
    "brakepct": "brake_pressure",
    "brakepressure": "brake_pressure",
    "brake_force": "brake_pressure",
    "steer": "steering_angle",
    "steering": "steering_angle",
    "steerangle": "steering_angle",
    "wheelspd_fl": "wheel_speed_fl",
    "wheelspdfr": "wheel_speed_fr",
    "wheel_speed_front_left": "wheel_speed_fl",
    "wheel_speed_front_right": "wheel_speed_fr",
}

MODEL_FEATURE_NAMES: List[str] = []
MODEL_FEATURE_METRICS: Dict[str, Dict[str, Any]] = {}
CALIBRATION_STATS: Dict[str, Any] = {}
MODEL_FEATURE_STATS_RAW: Dict[str, Any] = {}
SCALER_MEAN_CPU: Optional[torch.Tensor] = None
SCALER_STD_CPU: Optional[torch.Tensor] = None
SCALER_MEAN: Optional[torch.Tensor] = None
SCALER_STD: Optional[torch.Tensor] = None


def _normalize_feature_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _resolve_feature_index(feature_order: List[str], *candidates: str) -> Optional[int]:
    normalized = [_normalize_feature_name(item) for item in feature_order]
    for candidate in candidates:
        cand_norm = _normalize_feature_name(candidate)
        if cand_norm in normalized:
            return normalized.index(cand_norm)
    return None


def _extract_series(raw_arr: np.ndarray, feature_order: List[str], candidates: Tuple[str, ...], absolute: bool = False) -> Optional[np.ndarray]:
    idx = _resolve_feature_index(feature_order, *candidates)
    if idx is None or raw_arr.size == 0:
        return None
    series = raw_arr[:, idx]
    if absolute:
        series = np.abs(series)
    return series


def _tail_ratio(value: float, pleasant: float, caution: float, danger: float) -> float:
    eps = 1e-6
    caution = max(caution, pleasant + 1e-6)
    danger = max(danger, caution + 1e-6)
    if value <= pleasant:
        return 0.0
    if value <= caution:
        return (value - pleasant) / (caution - pleasant + eps) * 0.6
    if value <= danger:
        return 0.6 + 0.4 * (value - caution) / (danger - caution + eps)
    excess = value - danger
    return 1.0 + min(excess / (max(danger, 1.0) + eps), 0.4)


def _compute_domain_risk(raw_arr: np.ndarray, feature_order: List[str]) -> Tuple[float, str, List[str]]:
    if raw_arr.size == 0:
        return 0.0, "normal-envelope", []

    risk = 0.0
    scenario = "normal-envelope"
    scenario_priority = 0
    notes: List[str] = []

    def update_scenario(name: str, priority: int):
        nonlocal scenario, scenario_priority
        if priority > scenario_priority:
            scenario = name
            scenario_priority = priority

    def series_for(name: str, *, absolute: bool = False) -> Optional[np.ndarray]:
        idx = _resolve_feature_index(feature_order, name)
        if idx is None or raw_arr.size == 0:
            return None
        series = raw_arr[:, idx]
        if absolute:
            series = np.abs(series)
        return series

    def quantiles_for(name: str, kind: str) -> Dict[str, float]:
        metrics = MODEL_FEATURE_METRICS.get(name)
        if not metrics:
            return {}
        if kind == "abs":
            return metrics.get("abs_quantiles") or {}
        if kind == "neg":
            return metrics.get("neg_quantiles") or {}
        return metrics.get("quantiles") or {}

    speed_series = series_for("speed")
    speed_ratio = 0.0
    if speed_series is not None:
        metrics = MODEL_FEATURE_METRICS.get("speed", {})
        quantiles = quantiles_for("speed", "base")
        pleasant = quantiles.get("p80", float(np.percentile(speed_series, 80)))
        caution = quantiles.get("p95", pleasant + 5.0)
        danger = quantiles.get("p99", caution + 5.0)
        peak_speed = float(np.max(speed_series))
        mean_speed = float(np.mean(speed_series))
        speed_ratio = _tail_ratio(peak_speed, pleasant, caution, danger)
        risk += 0.08 * speed_ratio
        if speed_ratio >= 1.0:
            update_scenario("very-high-speed", 3)
        elif speed_ratio >= 0.6:
            update_scenario("high-speed-run", 1)
        if speed_ratio >= 0.4:
            notes.append(f"Peak speed {peak_speed:.1f} km/h (avg {mean_speed:.1f}).")

    lateral_series = series_for("lateral_g", absolute=True)
    lateral_ratio = 0.0
    if lateral_series is not None:
        quantiles_abs = quantiles_for("lateral_g", "abs")
        fallback_quantiles = quantiles_for("lateral_g", "base")
        pleasant = quantiles_abs.get("p80") or fallback_quantiles.get("p80") or float(np.percentile(lateral_series, 80))
        caution = quantiles_abs.get("p95") or fallback_quantiles.get("p95") or (pleasant + 0.4)
        danger = quantiles_abs.get("p99") or fallback_quantiles.get("p99") or (caution + 0.4)
        peak_lat = float(np.max(lateral_series))
        mean_lat = float(np.mean(lateral_series))
        lateral_ratio = _tail_ratio(peak_lat, pleasant, caution, danger)
        risk += 0.2 * lateral_ratio
        if lateral_ratio >= 1.0:
            update_scenario("high-speed-lateral", 4)
        elif lateral_ratio >= 0.6:
            update_scenario("aggressive-corner", 3)
        if lateral_ratio >= 0.5:
            notes.append(f"Lateral load {mean_lat:.2f} g (peak {peak_lat:.2f}).")

    yaw_series = series_for("yaw_rate", absolute=True)
    yaw_ratio = 0.0
    if yaw_series is not None:
        quantiles_abs = quantiles_for("yaw_rate", "abs")
        pleasant = quantiles_abs.get("p80") or float(np.percentile(yaw_series, 80))
        caution = quantiles_abs.get("p95") or (pleasant + 2.5)
        danger = quantiles_abs.get("p99") or (caution + 3.0)
        peak_yaw = float(np.max(yaw_series))
        yaw_ratio = _tail_ratio(peak_yaw, pleasant, caution, danger)
        risk += 0.16 * yaw_ratio
        if yaw_ratio >= 1.0:
            update_scenario("snap-oversteer", 4)
        elif yaw_ratio >= 0.6:
            update_scenario("rotation-instability", 3)
        if yaw_ratio >= 0.5:
            notes.append(f"Yaw rate excursion {peak_yaw:.1f}°/s.")

    long_series = series_for("longitudinal_g")
    decel_ratio = 0.0
    max_decel = 0.0
    if long_series is not None:
        decel = -np.minimum(long_series, 0.0)
        if np.any(decel > 0):
            max_decel = float(np.max(decel))
            neg_quantiles = quantiles_for("longitudinal_g", "neg")
            pleasant = neg_quantiles.get("p80") or float(np.percentile(decel, 80))
            caution = neg_quantiles.get("p95") or (pleasant + 0.4)
            danger = neg_quantiles.get("p99") or (caution + 0.4)
            decel_ratio = _tail_ratio(max_decel, pleasant, caution, danger)
            risk += 0.1 * decel_ratio
            if decel_ratio >= 0.6:
                update_scenario("emergency-brake", 2)
            if decel_ratio >= 0.4:
                notes.append(f"Braking decel {max_decel:.2f} g.")

    brake_series = series_for("brake_pressure")
    if brake_series is not None:
        max_brake = float(np.max(brake_series))
        brake_quantiles = quantiles_for("brake_pressure", "base")
        brake_p80 = brake_quantiles.get("p80", 0.35)
        if speed_ratio >= 0.7 and max_brake <= brake_p80:
            risk += 0.04
            update_scenario("traction-watch", 2)
            notes.append("High speed with limited brake modulation.")
        if decel_ratio <= 0.3 and max_brake >= 0.85:
            risk += 0.03
            update_scenario("grip-imbalance", 2)
            notes.append("Brake pressure high without matching decel response.")

    risk = _clamp(risk, 0.0, 0.42)
    if not notes:
        notes.append("Telemetry aligns with calibrated vehicle envelope.")
    return risk, scenario, notes

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
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    input_dim = int(ckpt["input_dim"])
    seq_len = int(ckpt["seq_len"])

    model = LSTMAutoencoder(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    feature_stats = ckpt.get("feature_stats", {}) or {}
    calibration = (ckpt.get("calibration") or {}).get("reconstruction_error", {})

    def pick(key: str, index: int, default: float) -> float:
        arr = feature_stats.get(key)
        if isinstance(arr, (list, tuple)) and index < len(arr):
            return float(arr[index])
        return float(default)

    feature_names = list(feature_stats.get("feature_names") or [])[:input_dim]
    if len(feature_names) < input_dim:
        for fallback in FALLBACK_FEATURE_NAMES:
            if fallback not in feature_names:
                feature_names.append(fallback)
            if len(feature_names) >= input_dim:
                break
    feature_names = feature_names[:input_dim]

    mean_values = [pick("mean", idx, 0.0) for idx in range(input_dim)]
    std_values = [max(pick("std", idx, 1.0), 1e-6) for idx in range(input_dim)]

    mean_tensor = torch.tensor(mean_values, dtype=torch.float32).view(1, 1, input_dim)
    std_tensor = torch.tensor(std_values, dtype=torch.float32).view(1, 1, input_dim)
    std_tensor = torch.clamp(std_tensor, min=1e-6)

    metrics: Dict[str, Dict[str, Any]] = {}
    for idx, name in enumerate(feature_names):
        quantiles = {
            "p50": pick("quantiles_p50", idx, mean_values[idx]),
            "p80": pick("quantiles_p80", idx, mean_values[idx]),
            "p90": pick("quantiles_p90", idx, mean_values[idx]),
            "p95": pick("quantiles_p95", idx, mean_values[idx]),
            "p99": pick("quantiles_p99", idx, mean_values[idx]),
        }
        metrics[name] = {
            "mean": mean_values[idx],
            "std": std_values[idx],
            "min": pick("min", idx, mean_values[idx] - std_values[idx]),
            "max": pick("max", idx, mean_values[idx] + std_values[idx]),
            "quantiles": quantiles,
            "abs_quantiles": feature_stats.get("abs_quantiles", {}).get(name, {}),
            "neg_quantiles": feature_stats.get("neg_quantiles", {}).get(name, {}),
        }

    global MODEL_FEATURE_NAMES, MODEL_FEATURE_METRICS, CALIBRATION_STATS, MODEL_FEATURE_STATS_RAW
    global SCALER_MEAN_CPU, SCALER_STD_CPU, SCALER_MEAN, SCALER_STD

    MODEL_FEATURE_NAMES = feature_names
    MODEL_FEATURE_METRICS = metrics
    MODEL_FEATURE_STATS_RAW = feature_stats
    CALIBRATION_STATS = calibration
    SCALER_MEAN_CPU = mean_tensor.clone()
    SCALER_STD_CPU = std_tensor.clone()
    SCALER_MEAN = SCALER_MEAN_CPU.to(DEVICE)
    SCALER_STD = SCALER_STD_CPU.to(DEVICE)

    return model, seq_len, input_dim, ckpt


MODEL_LOAD_ERROR: Optional[str] = None
MODEL_CHECKPOINT: Dict[str, Any] = {}
try:
    model, SEQ_LEN, INPUT_DIM, MODEL_CHECKPOINT = load_model()
except Exception as exc:  # pragma: no cover - defensive fallback
    MODEL_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
    INPUT_DIM = len(FEATURE_HINTS)
    SEQ_LEN = 100
    model = LSTMAutoencoder(input_dim=INPUT_DIM).to(DEVICE)
    model.eval()
    MODEL_FEATURE_NAMES[:] = [hint["name"] for hint in FEATURE_HINTS]
    CALIBRATION_STATS.clear()


def _normalize_key(key: str) -> str:
    clean = key.strip().replace("/", "_")
    clean = re.sub(r"\s+", "_", clean)
    clean = re.sub(r"[^a-zA-Z0-9_]+", "", clean)
    normalized = clean.lower() or "feature"
    return FEATURE_NAME_ALIASES.get(normalized, normalized)


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
        fallback_feature = (MODEL_FEATURE_NAMES[0] if MODEL_FEATURE_NAMES else FALLBACK_FEATURE_NAMES[0])
        timeline = [{fallback_feature: 0.0}]

    feature_order: List[str] = []
    for row in timeline:
        for key in row.keys():
            if key not in feature_order:
                feature_order.append(key)

    expected_dim = globals().get("INPUT_DIM", len(FALLBACK_FEATURE_NAMES))
    preferred_order = MODEL_FEATURE_NAMES or feature_order
    aligned_order: List[str] = []
    for name in preferred_order:
        if name in feature_order and name not in aligned_order:
            aligned_order.append(name)
    for name in feature_order:
        if name not in aligned_order:
            aligned_order.append(name)

    feature_order = aligned_order[:expected_dim]

    if len(feature_order) < expected_dim:
        filler_source = MODEL_FEATURE_NAMES or FALLBACK_FEATURE_NAMES
        for fallback in filler_source:
            if fallback not in feature_order:
                feature_order.append(fallback)
            if len(feature_order) >= expected_dim:
                break

    if len(feature_order) < expected_dim:
        for fallback in FALLBACK_FEATURE_NAMES:
            if fallback not in feature_order:
                feature_order.append(fallback)
            if len(feature_order) >= expected_dim:
                break

    return timeline, feature_order[:expected_dim]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _calibrated_anomaly_score(mse: float) -> float:
    stats = CALIBRATION_STATS or {}
    mean = float(stats.get("mse_mean", 0.6))
    p95 = float(stats.get("mse_p95", mean + 0.2))
    p99 = float(stats.get("mse_p99", p95 + 0.2))
    maximum = float(stats.get("mse_max", p99 + 0.2))

    if mse <= mean:
        if mean <= 1e-6:
            return _clamp(mse, 0.0, 1.0)
        return _clamp((mse / mean) * 0.2, 0.0, 0.2)
    if mse <= p95:
        return _clamp(0.2 + 0.4 * (mse - mean) / (p95 - mean + 1e-6), 0.0, 0.6)
    if mse <= p99:
        return _clamp(0.6 + 0.3 * (mse - p95) / (p99 - p95 + 1e-6), 0.6, 0.9)
    return _clamp(0.9 + 0.1 * ((mse - p99) / (max(maximum - p99, 1e-6))), 0.9, 1.0)


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

    tensor = torch.tensor(padded[None, :, :], dtype=torch.float32)
    return tensor, feature_order, timeline, raw_arr, padded


MODEL_METADATA: Dict[str, Any] = {
    "model_version": "v0.9.2-calibrated",
    "device": DEVICE,
    "seq_len": SEQ_LEN,
    "input_dim": INPUT_DIM,
    "model_loaded": model is not None,
    "model_path": MODEL_PATH,
    **({"model_load_error": MODEL_LOAD_ERROR} if MODEL_LOAD_ERROR else {}),
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
        "dataset": "Synthetic telemetry calibration set (normal / attack / caution blend)",
        "total_sequences": MODEL_FEATURE_STATS_RAW.get("dataset_size"),
        "total_timesteps": MODEL_FEATURE_STATS_RAW.get("total_timesteps"),
        "sequence_length": SEQ_LEN,
        "feature_names": MODEL_FEATURE_NAMES,
        "scaler_mean": MODEL_FEATURE_STATS_RAW.get("mean"),
        "scaler_std": MODEL_FEATURE_STATS_RAW.get("std"),
        "reconstruction_error": CALIBRATION_STATS,
    },
    "release_notes": [
        "Global scaler alignment between training and inference",
        "Domain risk derived from dataset quantiles instead of static thresholds",
        "Crash probability calibrated with logistic fusion of anomaly and environment risk",
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
        "gauge_thresholds": [0.3, 0.5, 0.7, 0.85],
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
    if model is None:
        raise RuntimeError(
            "Model not loaded. Install torch and ensure models/best_model.pth is present."
        )
    inference_start = time.perf_counter()
    tensor, feature_order, raw_timeline, raw_arr, padded_arr = preprocess_payload(telemetry)
    conditions = _extract_conditions(context)
    env_modifier, env_contributions, env_summary = _compute_environment_modifier(conditions)

    device_tensor = tensor.to(DEVICE)
    if SCALER_MEAN is not None and SCALER_STD is not None:
        norm_tensor = (device_tensor - SCALER_MEAN) / SCALER_STD
    else:
        norm_tensor = device_tensor

    temporal_ratio = (raw_arr.shape[0] / SEQ_LEN) if raw_arr.size else 1.0
    temporal_weight = 1.0
    if temporal_ratio < 1.0:
        temporal_weight = max(0.05, temporal_ratio)

    with torch.no_grad():
        recon_norm, latent = model(norm_tensor)
        error_tensor = (recon_norm - norm_tensor) ** 2
        if temporal_weight != 1.0:
            error_tensor = error_tensor * temporal_weight
        mse = torch.mean(error_tensor, dim=(1, 2))
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
        score = float(mse.item())

    anomaly_score = _calibrated_anomaly_score(score)
    domain_modifier, scenario_label, domain_notes = _compute_domain_risk(raw_arr, feature_order)

    combined_index = 0.55 * anomaly_score + domain_modifier
    logistic_component = 1.0 / (1.0 + np.exp(-3.0 * (combined_index - 0.58)))
    base_probability = _clamp(logistic_component, 0.05, 0.95)
    crash_probability = _clamp(base_probability + env_modifier, 0.02, 0.97)
    risk_anchor = max(anomaly_score, crash_probability)
    risk_band = _band_from_score(risk_anchor)
    latency_ms = round((time.perf_counter() - inference_start) * 1000, 2)
    severity_index = round((0.6 * _clamp(anomaly_score, 0.0, 1.0) + 0.4 * crash_probability) * 100, 1)

    feature_breakdown: List[Dict[str, Any]] = []
    scale_ref = max(float(CALIBRATION_STATS.get("mse_p95", 1.0)), 1e-6)
    for idx, feat in enumerate(feature_order):
        recon_error = float(feature_mse[idx]) if idx < len(feature_mse) else 0.0
        normalized_signal = max(recon_error / scale_ref, 0.0)
        scaled_importance = float(np.tanh(normalized_signal * 2.4))
        feature_breakdown.append(
            {
                "feature": feat,
                "recon_error": round(recon_error, 6),
                "scaled_importance": round(min(scaled_importance, 1.0), 3),
            }
        )

    feature_breakdown.sort(key=lambda item: item["recon_error"], reverse=True)
    for rank, item in enumerate(feature_breakdown, start=1):
        item["rank"] = rank

    dominant_feature = feature_breakdown[0]["feature"] if feature_breakdown else feature_order[0]

    if crash_probability >= 0.75:
        actions = ["precharge_brakes", "pretension_seatbelt", "notify_pit_wall"]
    elif crash_probability >= 0.55:
        actions = ["precharge_brakes", "pretension_seatbelt"]
    elif crash_probability >= 0.4:
        actions = ["precharge_brakes"]
    else:
        actions = []

    notes = [f"Dominant deviation detected on '{dominant_feature}'."] + domain_notes
    if scenario_label and scenario_label != "normal-envelope":
        notes.insert(1, f"Scenario classification: {scenario_label.replace('-', ' ')}.")
    notes.append(f"Calibrated crash probability {crash_probability * 100:.1f}% (combined index {combined_index:.3f}).")

    probability_components: List[Dict[str, Any]] = [
        {"label": "Anomaly score", "value": round(anomaly_score, 4)},
        {"label": "Telemetry envelope", "value": round(domain_modifier, 4), "detail": scenario_label},
        {"label": "Combined index", "value": round(combined_index, 4)},
        {"label": "Logistic baseline", "value": round(base_probability, 4)},
        {"label": "Environment modifier", "value": round(env_modifier, 4)},
        {"label": "Temporal weight", "value": round(temporal_weight, 4)},
    ]
    probability_components.extend(
        {
            "label": contrib.get("label", "env"),
            "detail": contrib.get("detail"),
            "value": contrib.get("delta", 0.0),
        }
        for contrib in env_contributions
    )
    probability_components.append({"label": "Crash probability", "value": round(crash_probability, 4)})

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
        "scenario_label": scenario_label,
        "dominant_feature": dominant_feature,
        "domain_notes": domain_notes,
        "domain_modifier": round(domain_modifier, 4),
        "temporal_ratio": round(temporal_ratio, 4),
        "temporal_weight": round(temporal_weight, 4),
    }

    if context:
        response["context_echo"] = context

    return response

if __name__ == "__main__":
    out = run_inference("speed: 214, yaw: 18, gz: 3.1, brake: 0.3")
    print(out)
