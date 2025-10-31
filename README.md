# VISIONALTRIX - F1-to-Road Safety AI

VISIONALTRIX is a demo stack that blends a FastAPI backend, an LSTM autoencoder, and a cinematic landing experience to illustrate how Formula 1 crash telemetry can transfer to consumer Advanced Driver Assistance Systems (ADAS). The repository ships with synthetic telemetry generators, trained model weights, and rich front-end storytelling (hero page, model training console, live visual simulator, and impact dashboard).

## Contents

- [Highlights](#highlights)
- [Project Layout](#project-layout)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
  - [Launching the API](#launching-the-api)
  - [Available Pages](#available-pages)
  - [Core API Endpoints](#core-api-endpoints)
- [Synthetic Telemetry + Training](#synthetic-telemetry--training)
- [Testing](#testing)
- [3D Assets](#3d-assets)
- [Troubleshooting](#troubleshooting)

## Highlights

- **Immersive hero experience** - `index1.html` combines a gradient-lit hero, animated stats, and a GLB Formula 1 car rendered with `<model-viewer>` for the instant "wow".
- **Interactive AI model console** - `ai-model.html` exposes inference controls, feature insights, and live output cards driven directly by the FastAPI `/api/infer` endpoint.
- **WebGL scenario playback** - `visual.html` uses Three.js to re-run wet-lane overtakes and show before/after ADAS behaviour.
- **Results dashboard** - `result.html` summarises risk deltas, braking improvements, and scenario coverage in an executive-friendly board.
- **Machine learning backbone** - `inference.py`, `src/model.py`, and `train_anomaly.py` implement an LSTM autoencoder and calibration pipeline for crash-risk scoring.
- **Synthetic data tooling** - `generate_telemetry.py` and the `telemetry_synth/` bundles can be used to quickly regenerate or augment telemetry sequences for experimentation.

## Project Layout

```
project-root/
|-- app.py                     # FastAPI application serving HTML + inference endpoints
|-- index1.html                # Hero landing page (source of truth for nav styling)
|-- ai-model.html              # AI training & inference microsite
|-- visual.html                # Three.js live scenario demo
|-- result.html                # Executive impact summary page
|-- inference.py               # Inference orchestration and risk post-processing
|-- src/
|   |-- dataset.py             # PyTorch dataset + dataloader helpers
|   `-- model.py               # LSTM autoencoder definition
|-- train_anomaly.py           # Training script for the autoencoder
|-- generate_telemetry.py      # Synthetic telemetry generator (CSV + JSONL bundles)
|-- telemetry_synth/           # Pre-generated telemetry bundles
|-- models/                    # Persisted weights (expects best_model.pth)
|-- tests/test_demo_payload.py # Smoke test for inference calibration
|-- f1_2023_mercedes_*.glb     # Blender-exported W14 model rendered on the hero
`-- data*, data_generated/     # Optional raw + processed training data
```

## Prerequisites

- Python **3.10+** recommended.
- Python packages (install via `pip install ...`):
  - `fastapi`, `uvicorn[standard]`
  - `pydantic`
  - `torch`, `numpy`, `pandas`
  - `scikit-learn` (for metrics inside training scripts)
  - `python-dotenv` (optional, if you later externalise config)
- Node.js is **not** required; the front-end pages are static assets served by the API.

If you are running on CPU-only hardware, ensure you install the CPU build of PyTorch (`pip install torch --index-url https://download.pytorch.org/whl/cpu`).

## Quick Start

### Launching the API

From a PowerShell prompt:

```powershell
cd C:\Users\Urvansh\OneDrive\Desktop\Altrix-1
pip install fastapi "uvicorn[standard]" torch numpy pandas scikit-learn pydantic
uvicorn app:app --reload
```

The server now listens on `http://127.0.0.1:8000`.

### Available Pages

Once the server is running, open the following routes in a browser:

- `http://127.0.0.1:8000/index1.html` - hero landing page (root `/` also falls back here).
- `http://127.0.0.1:8000/ai-model.html` - AI model training & inference walkthrough.
- `http://127.0.0.1:8000/visual.html` - Live scenario simulator.
- `http://127.0.0.1:8000/result.html` - Impact dashboard.

All navigation bars now reference `index1.html`, so intra-site links resolve without 404s.

### Core API Endpoints

| Method | Route                  | Description                                                 |
|--------|------------------------|-------------------------------------------------------------|
| GET    | `/api/health`          | Simple readiness probe (status, device, model path).       |
| GET    | `/api/model-metadata`  | Returns cached calibration metadata and feature hints.      |
| POST   | `/api/infer`           | Runs inference on telemetry payloads and returns risk.      |

Example inference request:

```bash
curl -X POST http://127.0.0.1:8000/api/infer \
  -H "Content-Type: application/json" \
  -d '{"telemetry": "speed: 214, yaw_rate: 17.6, lateral_g: 3.2, brake_pressure: 0.4"}'
```

The response contains `crash_probability`, `risk_band`, recommended safety actions, and feature attributions derived from `inference.py`.

## Synthetic Telemetry & Training

- Generate fresh telemetry samples:
  ```bash
  python generate_telemetry.py
  ```
  Outputs land in `telemetry_synth/` and mirror the structure expected by the training pipeline.

- Train (or re-train) the LSTM autoencoder:
  ```bash
  python train_anomaly.py
  ```
  The script reads sequences from `data_generated/train/`, computes feature statistics, trains for the configured epochs, and writes updated weights + metadata into `models/`. Ensure the resulting `best_model.pth` is present so `inference.py` can load it.

- Model code lives in `src/model.py`; dataset and batching helpers are in `src/dataset.py`.

## Testing

Run the bundled smoke test to make sure inference outputs stay calibrated:

```bash
python -m unittest tests/test_demo_payload.py
```

The test shields against significant probability drift for the published demo payload. Extend this module with additional regression or integration tests as needed.

## 3D Assets

- `f1_2023_mercedes_amg_w14_e_performance_s1.glb` is loaded via `<model-viewer>` on the landing page. Keep it in the repository root so the FastAPI route `/f1_2023_mercedes_amg_w14_e_performance_s1.glb` can serve it without extra configuration.
- The hero model enters with a GSAP bounce animation defined in `index1.html`. Adjust `camera-orbit`, `camera-target`, or the `.car-model` transform to tweak the presentation.

## Troubleshooting

- **404 for static pages/models** - ensure you start the FastAPI app from the repo root so relative paths resolve. The app mounts the current directory under `/assets` and exposes explicit routes for each HTML page and the GLB file.
- **PyTorch GPU warnings** - the code automatically falls back to CPU (`torch.cuda.is_available()` guard). Install the appropriate torch build for your hardware.
- **Large telemetry bundles** - the synthetic datasets are sizeable (tens of MB each). If disk space is a concern, regenerate smaller bundles via `generate_telemetry.py` with custom parameters.
- **Navigation bar overlap on mobile** - the nav wraps below 720 px width. If you add items, mirror the adjustments in each page's `@media (max-width: 720px)` rule.

Enjoy exploring how F1 telemetry can harden road-car safety systems!
