from pathlib import Path
import sys
import unittest

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo.demo_logic import (  # noqa: E402
    PRESETS,
    build_patient_row,
    predict,
    risk_band,
    score_clinical_baselines,
    what_if_deltas,
)


class DemoLogicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = joblib.load(ROOT / "demo" / "artifacts" / "readmission_demo.joblib")

    def test_presets_are_ordered_by_risk(self):
        risks = []
        for preset in PRESETS.values():
            row = build_patient_row(self.bundle["defaults"], preset)
            self.assertEqual(row.columns.tolist(), list(self.bundle["defaults"]))
            risks.append(predict(self.bundle, row))
        self.assertLess(risks[0], risks[1])
        self.assertLess(risks[1], risks[2])
        self.assertLess(risks[0], self.bundle["prevalence"])
        self.assertGreater(risks[2], self.bundle["threshold_top20"])

    def test_high_risk_profile_is_prioritized(self):
        row = build_patient_row(self.bundle["defaults"], PRESETS["Alto riesgo"])
        risk = predict(self.bundle, row)
        self.assertEqual(
            risk_band(risk, self.bundle["threshold_top20"], self.bundle["prevalence"]),
            "Prioridad de seguimiento",
        )
        self.assertEqual(len(what_if_deltas(self.bundle, row)), 9)

    def test_clinical_scores_return_probabilities(self):
        result = score_clinical_baselines(self.bundle, PRESETS["Riesgo intermedio"])
        self.assertTrue(0 < result["LACE"] < 1)
        self.assertTrue(0 < result["HOSPITAL"] < 1)
        self.assertIsInstance(result["LACE_points"], int)
        self.assertIsInstance(result["HOSPITAL_points"], int)


if __name__ == "__main__":
    unittest.main()
