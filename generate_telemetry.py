import json
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

SAMPLE_RATE = 50  # Hz
DT = 1.0 / SAMPLE_RATE
SEQ_DURATION = 2.0  # seconds
SEQ_LEN = int(SEQ_DURATION * SAMPLE_RATE)  # 100

OUTPUT_DIR = Path("./telemetry_synth")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = [
    "speed_kmh",
    "yaw_rate_dps",
    "lateral_g",
    "longitudinal_g",
    "brake_pressure",
    "steering_angle_deg",
    "wheel_speed_fl_kmh",
    "wheel_speed_fr_kmh",
]

# --------------------------- helpers ---------------------------


def smooth_series(x, strength=0.3):
    """Simple EMA smoothing to enforce time continuity."""
    out = np.zeros_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = strength * x[i] + (1 - strength) * out[i - 1]
    return out


def make_wheel_speeds(speed_kmh, slip_range=0.03):
    slip_fl = rng.uniform(-slip_range, slip_range, size=speed_kmh.shape)
    slip_fr = rng.uniform(-slip_range, slip_range, size=speed_kmh.shape)
    return speed_kmh * (1 + slip_fl), speed_kmh * (1 + slip_fr)


def make_baseline_sequence():
    base_speed = rng.uniform(120, 260)
    speed = base_speed + rng.normal(0, 1.5, size=SEQ_LEN)
    speed = smooth_series(speed, strength=0.25)
    speed = np.clip(speed, 60, 310)

    scenario = rng.choice(
        ["straight", "gentle_brake", "corner_left", "corner_right", "throttle_mod"],
        p=[0.25, 0.20, 0.18, 0.17, 0.20],
    )

    yaw = np.zeros(SEQ_LEN)
    lat_g = np.zeros(SEQ_LEN)
    long_g = np.zeros(SEQ_LEN)
    brake = np.zeros(SEQ_LEN)
    steer = np.zeros(SEQ_LEN)

    if scenario == "straight":
        lat_g += rng.normal(0, 0.03, size=SEQ_LEN)
        yaw += rng.normal(0, 0.15, size=SEQ_LEN)
        long_g += rng.normal(0, 0.02, size=SEQ_LEN)
        brake += np.maximum(0, rng.normal(0.02, 0.01, size=SEQ_LEN))
        steer += rng.normal(0, 0.5, size=SEQ_LEN)

    elif scenario == "gentle_brake":
        start = rng.integers(15, 40)
        end = start + rng.integers(20, 40)
        brake[start:end] = np.linspace(0.05, rng.uniform(0.25, 0.45), end - start)
        long_g[start:end] = -brake[start:end] * rng.uniform(2.8, 3.5) / 9.81
        steer += rng.normal(0, 0.8, size=SEQ_LEN)
        lat_g += rng.normal(0, 0.05, size=SEQ_LEN)
        yaw += lat_g * rng.uniform(18, 25)
        speed[start:end] -= np.linspace(0, rng.uniform(8, 20), end - start)

    elif scenario in ("corner_left", "corner_right"):
        sign = -1 if scenario == "corner_left" else 1
        start = rng.integers(10, 35)
        end = start + rng.integers(30, 55)
        corner_lat = rng.uniform(0.35, 1.45)
        lat_g[start:end] = np.linspace(0, sign * corner_lat, end - start)
        yaw[start:end] = lat_g[start:end] * rng.uniform(25, 40)
        steer[start:end] = (lat_g[start:end] / corner_lat) * sign * rng.uniform(8, 18)
        lat_g += rng.normal(0, 0.03, size=SEQ_LEN)
        yaw += rng.normal(0, 0.3, size=SEQ_LEN)
        long_g += rng.normal(0, 0.03, size=SEQ_LEN)

    elif scenario == "throttle_mod":
        long_g = rng.normal(0, 0.04, size=SEQ_LEN)
        for _ in range(2):
            start = rng.integers(5, 60)
            dur = rng.integers(10, 25)
            long_g[start:start + dur] += rng.uniform(0.03, 0.12)
            speed[start:start + dur] += np.linspace(0, rng.uniform(4, 10), dur)
        steer += rng.normal(0, 0.8, size=SEQ_LEN)
        lat_g += rng.normal(0, 0.05, size=SEQ_LEN)
        yaw += lat_g * rng.uniform(20, 30)

    speed = smooth_series(speed, strength=0.35)
    yaw = smooth_series(yaw, strength=0.45)
    lat_g = smooth_series(lat_g, strength=0.4)
    long_g = smooth_series(long_g, strength=0.4)
    brake = np.clip(smooth_series(brake, strength=0.4), 0, 1)
    steer = smooth_series(steer, strength=0.35)

    ws_fl, ws_fr = make_wheel_speeds(speed, slip_range=0.03)

    arr = np.vstack([speed, yaw, lat_g, long_g, brake, steer, ws_fl, ws_fr]).T
    meta = {"scenario": scenario}
    return arr, meta


def make_highrisk_sequence():
    base_speed = rng.uniform(140, 300)
    speed = base_speed + rng.normal(0, 1.5, size=SEQ_LEN)
    speed = smooth_series(speed, strength=0.25)
    speed = np.clip(speed, 80, 330)

    scenario = rng.choice(
        [
            "snap_oversteer",
            "traction_loss_highspeed",
            "heavy_brake_low_grip",
            "steering_oscillation",
            "tyre_blowout_response",
        ]
    )

    yaw = rng.normal(0, 0.2, size=SEQ_LEN)
    lat_g = rng.normal(0, 0.05, size=SEQ_LEN)
    long_g = rng.normal(0, 0.04, size=SEQ_LEN)
    brake = np.maximum(0, rng.normal(0.02, 0.015, size=SEQ_LEN))
    steer = rng.normal(0, 0.6, size=SEQ_LEN)

    lead_end = rng.integers(55, 75)
    severity = float(rng.uniform(0.35, 0.95))

    if scenario == "snap_oversteer":
        lat_g[:lead_end] += np.linspace(0, rng.uniform(0.6, 1.4), lead_end)
        yaw[:lead_end] += lat_g[:lead_end] * rng.uniform(20, 35)
        yaw[lead_end:] += rng.uniform(40, 120) * np.sign(rng.normal())
        lat_g[lead_end:] += np.sign(rng.normal()) * rng.uniform(0.2, 0.8)
        steer[lead_end:] += np.sign(rng.normal()) * rng.uniform(5, 18)

    elif scenario == "traction_loss_highspeed":
        lat_g[:lead_end] += np.linspace(0, rng.uniform(0.5, 1.0), lead_end)
        yaw[:lead_end] += lat_g[:lead_end] * rng.uniform(22, 30)
        speed[lead_end:] -= np.linspace(0, rng.uniform(5, 25), SEQ_LEN - lead_end)
        long_g[lead_end:] -= rng.uniform(0.15, 0.35)

    elif scenario == "heavy_brake_low_grip":
        brake[lead_end:] = np.linspace(0.2, 1.0, SEQ_LEN - lead_end)
        long_g[lead_end:] -= np.linspace(0.15, 0.5, SEQ_LEN - lead_end) * rng.uniform(0.4, 0.65)
        lat_g[lead_end:] += rng.normal(0, 0.08, size=SEQ_LEN - lead_end)

    elif scenario == "steering_oscillation":
        for i in range(lead_end, SEQ_LEN):
            steer[i] = 10 * np.sin(2 * np.pi * (i - lead_end) / rng.uniform(6, 12))
        yaw[lead_end:] += steer[lead_end:] * rng.uniform(1.3, 2.0)
        lat_g[lead_end:] += steer[lead_end:] / 10 * rng.uniform(0.4, 0.9)

    elif scenario == "tyre_blowout_response":
        lat_g[:lead_end] += np.linspace(0, rng.uniform(0.4, 0.8), lead_end)
        yaw[:lead_end] += lat_g[:lead_end] * rng.uniform(20, 30)
        speed[lead_end:] -= np.linspace(0, rng.uniform(10, 30), SEQ_LEN - lead_end)

    speed = smooth_series(speed, strength=0.35)
    yaw = smooth_series(yaw, strength=0.45)
    lat_g = smooth_series(lat_g, strength=0.4)
    long_g = smooth_series(long_g, strength=0.4)
    brake = np.clip(smooth_series(brake, strength=0.4), 0, 1)
    steer = smooth_series(steer, strength=0.35)

    ws_fl, ws_fr = make_wheel_speeds(speed, slip_range=0.03)
    if scenario in ("traction_loss_highspeed", "tyre_blowout_response"):
        ws_fl[lead_end:] *= rng.uniform(0.90, 0.98)
        ws_fr[lead_end:] *= rng.uniform(1.02, 1.12)

    arr = np.vstack([speed, yaw, lat_g, long_g, brake, steer, ws_fl, ws_fr]).T
    meta = {"scenario_label": scenario, "severity": severity}
    return arr, meta


def random_env_meta():
    weather = rng.choice(["sunny", "overcast", "light_rain", "rain", "wet"])
    track_surface = rng.choice(["green", "rubbered", "wet", "mixed"])
    humidity = float(rng.uniform(30, 95))
    tyre_wear = float(rng.uniform(0, 1))
    driver_state = rng.choice(["fresh", "fatigued", "pushing", "cautious"])
    return {
        "weather": weather,
        "track_surface": track_surface,
        "humidity": humidity,
        "tyre_wear": tyre_wear,
        "driver_state": driver_state,
    }


def apply_stress(arr):
    mask = np.ones_like(arr, dtype=bool)
    noise_scales = np.array([0.5, 0.6, 0.02, 0.02, 0.01, 0.5, 0.5, 0.5])
    noise = rng.normal(0, noise_scales, size=arr.shape)
    arr = arr + noise

    if rng.random() < 0.45:
        start = rng.integers(5, arr.shape[0] - 10)
        length = rng.integers(2, 6)
        arr[start:start+length, :] = np.nan
        mask[start:start+length, :] = False

    return arr, mask


def maybe_resample_to_50hz_from_80hz(arr):
    if rng.random() < 0.35:
        return arr
    T = arr.shape[0]
    t80 = np.linspace(0, SEQ_DURATION, 160, endpoint=False)
    t50 = np.linspace(0, SEQ_DURATION, T, endpoint=False)
    arr80 = np.zeros((len(t80), arr.shape[1]))
    for c in range(arr.shape[1]):
        arr80[:, c] = np.interp(t80, t50, arr[:, c])
    jitter = rng.normal(0, 0.0008, size=t50.shape)
    t50_j = np.clip(t50 + jitter, 0, SEQ_DURATION)
    arr50 = np.zeros_like(arr)
    for c in range(arr.shape[1]):
        arr50[:, c] = np.interp(t50_j, t80, arr80[:, c])
    return arr50


# --------------------------- generate bundles ---------------------------


def generate_bundle_1(n_sequences=2000):
    records = []
    for idx in range(n_sequences):
        arr, meta = make_baseline_sequence()
        for t in range(SEQ_LEN):
            rec = {
                "seq_id": idx,
                "t": t * DT,
            }
            for ci, ch in enumerate(CHANNELS):
                rec[ch] = float(arr[t, ci])
            rec.update(meta)
            records.append(rec)
    df = pd.DataFrame(records)
    return df


def generate_bundle_2(n_sequences=800):
    records = []
    for idx in range(n_sequences):
        arr, meta = make_highrisk_sequence()
        for t in range(SEQ_LEN):
            rec = {
                "seq_id": idx,
                "t": t * DT,
            }
            for ci, ch in enumerate(CHANNELS):
                rec[ch] = float(arr[t, ci])
            rec.update(meta)
            records.append(rec)
    df = pd.DataFrame(records)
    return df


def generate_bundle_3(n_sequences=400):
    records = []
    for idx in range(n_sequences):
        if rng.random() < 0.7:
            arr, _ = make_baseline_sequence()
        else:
            arr, _ = make_highrisk_sequence()
        arr = maybe_resample_to_50hz_from_80hz(arr)
        arr, mask = apply_stress(arr)
        env = random_env_meta()
        base_risk = np.clip(np.nanmax(np.abs(arr[:, 2])) * 0.4, 0, 1)
        base_risk += env["tyre_wear"] * 0.35
        if env["driver_state"] == "pushing":
            base_risk += 0.15
        if env["track_surface"] in ("wet", "mixed"):
            base_risk += 0.12
        base_risk = float(np.clip(base_risk, 0, 1))
        for t in range(SEQ_LEN):
            rec = {
                "seq_id": idx,
                "t": t * DT,
            }
            for ci, ch in enumerate(CHANNELS):
                val = arr[t, ci]
                rec[ch] = None if np.isnan(val) else float(val)
            rec["missing_mask"] = json.dumps(mask[t, :].tolist())
            rec["env_weather"] = env["weather"]
            rec["env_track_surface"] = env["track_surface"]
            rec["env_humidity"] = env["humidity"]
            rec["env_tyre_wear"] = env["tyre_wear"]
            rec["env_driver_state"] = env["driver_state"]
            rec["gt_crash_probability"] = base_risk
            records.append(rec)
    df = pd.DataFrame(records)
    return df


def save_df(df, name):
    csv_path = OUTPUT_DIR / f"{name}.csv"
    jsonl_path = OUTPUT_DIR / f"{name}.jsonl"
    df.to_csv(csv_path, index=False)
    with jsonl_path.open("w") as f:
        for _, row in df.iterrows():
            f.write(row.to_json() + "\n")
    print(f"saved {name} → {csv_path} , {jsonl_path}")


def main():
    b1 = generate_bundle_1()
    b2 = generate_bundle_2()
    b3 = generate_bundle_3()

    save_df(b1, "bundle_1_baseline")
    save_df(b2, "bundle_2_highrisk")
    save_df(b3, "bundle_3_stress")

    summary = {}
    for ch in CHANNELS:
        series = b1[ch]
        summary[ch] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "p80": float(series.quantile(0.80)),
            "p95": float(series.quantile(0.95)),
            "p99": float(series.quantile(0.99)),
        }
    (OUTPUT_DIR / "bundle_1_summary.json").write_text(json.dumps(summary, indent=2))
    print("bundle_1 summary saved.")


if __name__ == "__main__":
    main()
