from pathlib import Path
import unittest


class TestRunReportWrapper(unittest.TestCase):
    def test_wrapper_exposes_intelligence_flags(self):
        script = Path(__file__).resolve().parents[1] / "run_report.sh"
        content = script.read_text(encoding="utf-8")

        self.assertIn("--intelligence", content)
        self.assertIn("--no-intelligence", content)
        self.assertIn("--intelligence-model", content)
        self.assertIn("--intelligence-fallback-models", content)
        self.assertIn("--intelligence-sections", content)

    def test_wrapper_forwards_flags_to_python_args(self):
        script = Path(__file__).resolve().parents[1] / "run_report.sh"
        content = script.read_text(encoding="utf-8")

        self.assertIn('PYTHON_ARGS+=("$INTELLIGENCE_TOGGLE")', content)
        self.assertIn('PYTHON_ARGS+=("--intelligence-model" "$INTELLIGENCE_MODEL")', content)
        self.assertIn('PYTHON_ARGS+=("--intelligence-fallback-models" "$INTELLIGENCE_FALLBACK_MODELS")', content)
        self.assertIn('PYTHON_ARGS+=("--intelligence-sections" "$INTELLIGENCE_SECTIONS")', content)


if __name__ == "__main__":
    unittest.main()
