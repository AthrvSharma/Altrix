import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import run_inference


class DemoPayloadCalibrationTest(unittest.TestCase):
    """
    Guard the calibrated crash probability for our published demo payload.
    If this starts drifting by >5 percentage points, investigate the calibration pipeline.
    """

    def test_demo_payload_probability(self):
        payload = "speed: 214, yaw_rate: 17.6, lateral_g: 3.2, brake_pressure: 0.4"
        result = run_inference(payload)

        probability = result["crash_probability"]
        risk_band = result["risk_band"]

        self.assertIn(
            risk_band,
            {"normal", "elevated", "high", "critical"},
            "Risk band should be a recognised value.",
        )
        self.assertTrue(math.isfinite(probability), "Crash probability should be a finite float.")
        self.assertLessEqual(
            abs(probability - 0.42),
            0.05,
            f"Demo payload probability drifted to {probability:.3f}; check calibration thresholds or model weights.",
        )
        self.assertEqual("elevated", risk_band, f"Unexpected risk band for demo payload: {risk_band}")


if __name__ == "__main__":
    unittest.main()
