import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from reports.fpl_report.dgw_bgw_fetcher import fetch_dgw_bgw_intelligence, merge_bgw_dgw_data


class _DummyMessage:
    def __init__(self, content: str):
        self.content = content


class _DummyChoice:
    def __init__(self, content: str):
        self.message = _DummyMessage(content)


class _DummyResponse:
    def __init__(self, content: str):
        self.choices = [_DummyChoice(content)]


class _DummyCompletions:
    def __init__(self, handler):
        self._handler = handler

    def create(self, **kwargs):
        return self._handler(kwargs)


class _DummyChat:
    def __init__(self, handler):
        self.completions = _DummyCompletions(handler)


class _DummyClient:
    def __init__(self, handler):
        self.chat = _DummyChat(handler)


def _predicted_payload() -> str:
    return '''
{
    "bgw": [
        {
            "gw": 29,
            "teams_missing": 4,
            "confidence": "high",
            "reason": "FA Cup quarter-finals"
        }
    ],
    "dgw": [
        {
            "gw": 34,
            "teams_doubled": 6,
            "confidence": "medium",
            "reason": "Rescheduled fixtures"
        }
    ]
}
'''


class TestDgwBgwFetcher(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch('reports.fpl_report.dgw_bgw_fetcher.INTELLIGENCE', {"dgw_bgw_search": {"enabled": True}})
    @patch('reports.fpl_report.dgw_bgw_fetcher.SEASON', "2025-26")
    def test_fetch_dgw_bgw_intelligence_success(self):
        def handler(kwargs):
            return _DummyResponse(_predicted_payload())

        # We must patch the OpenAI client inside fetch_dgw_bgw_intelligence
        with patch.dict(os.environ, {"LITELLM_API_BASE": "http://test", "LITELLM_API_KEY": "k"}, clear=True):
            with patch('openai.OpenAI', return_value=_DummyClient(handler)):
                result = fetch_dgw_bgw_intelligence(current_gw=20)
                
        self.assertIn("bgw", result)
        self.assertIn("dgw", result)
        self.assertEqual(len(result["bgw"]), 1)
        self.assertEqual(result["bgw"][0]["gw"], 29)
        self.assertEqual(len(result["dgw"]), 1)
        self.assertEqual(result["dgw"][0]["gw"], 34)

    @patch('reports.fpl_report.dgw_bgw_fetcher.INTELLIGENCE', {"dgw_bgw_search": {"enabled": True}})
    @patch('reports.fpl_report.dgw_bgw_fetcher.SEASON', "2025-26")
    def test_fetch_dgw_bgw_intelligence_disabled_or_missing_env(self):
        with patch.dict(os.environ, {}, clear=True):
            result = fetch_dgw_bgw_intelligence(current_gw=20)
            self.assertEqual(result, {"bgw": [], "dgw": []})

    def test_merge_bgw_dgw_data(self):
        confirmed = {
            "bgw": [{"gw": 28, "teams_missing": 2, "team_ids": [1, 2]}],
            "dgw": [{"gw": 34, "teams_doubled": 2, "team_ids": [3, 4]}],
            "normal": [1, 2, 3]
        }
        
        predicted = {
            "bgw": [
                {"gw": 28, "teams_missing": 4, "confidence": "low", "reason": "test"}, # Should be ignored because 28 is confirmed
                {"gw": 32, "teams_missing": 6, "confidence": "high", "reason": "FA Cup"} # Should be added
            ],
            "dgw": [
                {"gw": 34, "teams_doubled": 4, "confidence": "medium", "reason": "test"}, # Should be ignored
                {"gw": 37, "teams_doubled": 8, "confidence": "high", "reason": "Rescheduled"} # Should be added
            ]
        }
        
        merged = merge_bgw_dgw_data(confirmed, predicted)
        
        self.assertEqual(len(merged["bgw"]), 2)
        # BGW 28 is confirmed, so predicted should be False
        bgw_28 = next(b for b in merged["bgw"] if b["gw"] == 28)
        self.assertFalse(bgw_28["predicted"])
        self.assertEqual(bgw_28["teams_missing"], 2)
        
        # BGW 32 is predicted
        bgw_32 = next(b for b in merged["bgw"] if b["gw"] == 32)
        self.assertTrue(bgw_32["predicted"])
        self.assertEqual(bgw_32["teams_missing"], 6)
        
        self.assertEqual(len(merged["dgw"]), 2)
        # DGW 34 is confirmed
        dgw_34 = next(d for d in merged["dgw"] if d["gw"] == 34)
        self.assertFalse(dgw_34["predicted"])
        self.assertEqual(dgw_34["teams_doubled"], 2)
        
        # DGW 37 is predicted
        dgw_37 = next(d for d in merged["dgw"] if d["gw"] == 37)
        self.assertTrue(dgw_37["predicted"])
        self.assertEqual(dgw_37["teams_doubled"], 8)
