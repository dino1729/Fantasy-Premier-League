import sys
from pathlib import Path
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import generate_fpl_report as gfr


class TestGenerateFplReportCli(unittest.TestCase):
    def test_intelligence_cli_flags_parse(self):
        argv = [
            "generate_fpl_report.py",
            "--team", "847569",
            "--intelligence",
            "--intelligence-model", "gpt-5.2",
            "--intelligence-fallback-models", "gemini-3.1-pro-preview,model-router",
            "--intelligence-sections", "transfer_strategy,chip_usage_strategy",
        ]
        with patch.object(sys, "argv", argv):
            args = gfr.parse_args()

        self.assertTrue(args.intelligence)
        self.assertFalse(args.no_intelligence)
        self.assertEqual(args.intelligence_model, "gpt-5.2")
        self.assertEqual(args.intelligence_fallback_models, "gemini-3.1-pro-preview,model-router")
        self.assertEqual(args.intelligence_sections, "transfer_strategy,chip_usage_strategy")

    def test_no_intelligence_cli_flag_parse(self):
        argv = ["generate_fpl_report.py", "--team", "847569", "--no-intelligence"]
        with patch.object(sys, "argv", argv):
            args = gfr.parse_args()

        self.assertFalse(args.intelligence)
        self.assertTrue(args.no_intelligence)

    def test_parse_csv_list_helper(self):
        self.assertEqual(gfr._parse_csv_list(None), [])
        self.assertEqual(
            gfr._parse_csv_list(" transfer_strategy , chip_usage_strategy ,, "),
            ["transfer_strategy", "chip_usage_strategy"],
        )


if __name__ == "__main__":
    unittest.main()
