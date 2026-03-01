import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


from reports.fpl_report.intelligence_layer import (
    IntelligenceLayer,
    SECTION_TRANSFER_STRATEGY,
    SECTION_SEASON_INSIGHTS,
)


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


def _narrative_payload() -> str:
    return (
        '{"headline":"Headline","tactical_summary":"Summary","metric_highlights":["+6.5 xP"],'
        '"actions":["Action 1"],"risks":["Risk 1"]}'
    )


def _chip_payload() -> str:
    return (
        '{"headline":"Chip Headline","tactical_summary":"Chip summary","metric_highlights":["3 chips"],'
        '"actions":["Action 1"],"risks":["Risk 1"],'
        '"chip_recommendations":{"wildcard":{"recommendation":"WC","trigger":"T1"},'
        '"freehit":{"recommendation":"FH","trigger":"T2"},'
        '"bboost":{"recommendation":"BB","trigger":"T3"},'
        '"3xc":{"recommendation":"TC","trigger":"T4"}}}'
    )


class TestIntelligenceLayer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name)
        self.base_settings = {
            "enabled": True,
            "model": "gpt-5.2",
            "fallback_models": ["gemini-3.1-pro-preview"],
            "retries": 2,
            "timeout_seconds": 30,
            "cache_ttl_seconds": 3600,
            "sections": {
                "transfer_strategy": True,
                "wildcard_draft": False,
                "free_hit_draft": False,
                "chip_usage_strategy": False,
                "season_insights": False,
            },
            "max_tokens": {"transfer_strategy": 600, "season_insights": 600},
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def _minimal_context(self):
        return {
            "multi_week_strategy": {
                "current_gameweek": 17,
                "planning_horizon": 5,
                "expected_value": {
                    "current_squad": 51.2,
                    "optimized_squad": 57.7,
                    "potential_gain": 6.5,
                },
                "immediate_recommendations": [],
                "model_confidence": "medium",
                "model_metrics": {"mae": 2.5, "r2": 0.2},
                "mip_recommendation": None,
            },
            "squad_analysis": [
                {"name": "Alpha", "raw_stats": {"total_points": 120}, "form_analysis": {"average": 6.0, "trend": "rising"}}
            ],
            "gw_history": [{"points": 60}, {"points": 55}],
            "chip_analysis": {
                "current_gw": 17,
                "half": "first",
                "chips_remaining_display": "3/4",
                "deadline_warning": {"message": "warning"},
                "squad_issues": {"summary": "issue summary"},
                "triggers": ["trigger 1"],
                "bgws": [{"gw": 29, "teams_missing": 4, "predicted": True}],
                "dgws": [{"gw": 34, "teams_doubled": 6, "predicted": False}],
                "synergies": [{"strategy": "WC then BB"}],
                "chips": {
                    "wildcard": {"urgency": "high", "recommendation": "WC now", "target_gw": 17},
                    "freehit": {"urgency": "low", "recommendation": "Save FH", "target_gw": None},
                    "bboost": {
                        "urgency": "low",
                        "recommendation": "Save BB",
                        "target_gw": None,
                        "best_dgw": {"gw": 34, "total_projected": 25.5}
                    },
                    "3xc": {"urgency": "low", "recommendation": "Save TC", "target_gw": None}
                }
            }
        }

    def test_env_missing_uses_deterministic_fallback(self):
        layer = IntelligenceLayer(settings=self.base_settings, cache_dir=self.cache_dir)
        with patch.dict(os.environ, {}, clear=True):
            result = layer.run(self._minimal_context())
        self.assertEqual(result.meta[SECTION_TRANSFER_STRATEGY]["source"], "deterministic")
        self.assertEqual(result.meta[SECTION_TRANSFER_STRATEGY]["error"], "credentials_missing")

    def test_primary_success_path(self):
        def handler(kwargs):
            return _DummyResponse(_narrative_payload())

        layer = IntelligenceLayer(
            settings=self.base_settings,
            cache_dir=self.cache_dir,
            client_factory=lambda _base, _key: _DummyClient(handler),
        )
        with patch.dict(os.environ, {"LITELLM_API_BASE": "http://test", "LITELLM_API_KEY": "k"}, clear=True):
            result = layer.run(self._minimal_context())
        self.assertIn(SECTION_TRANSFER_STRATEGY, result.payload)
        self.assertEqual(result.meta[SECTION_TRANSFER_STRATEGY]["source"], "ai")
        self.assertEqual(result.meta[SECTION_TRANSFER_STRATEGY]["model"], "gpt-5.2")

    def test_retry_and_backoff_then_success(self):
        calls = {"count": 0}
        sleeps = []

        def handler(kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise ValueError("transient")
            return _DummyResponse(_narrative_payload())

        layer = IntelligenceLayer(
            settings=self.base_settings,
            cache_dir=self.cache_dir,
            sleep_func=lambda s: sleeps.append(s),
            client_factory=lambda _base, _key: _DummyClient(handler),
        )
        with patch.dict(os.environ, {"LITELLM_API_BASE": "http://test", "LITELLM_API_KEY": "k"}, clear=True):
            result = layer.run(self._minimal_context())
        self.assertEqual(result.meta[SECTION_TRANSFER_STRATEGY]["source"], "ai")
        self.assertGreaterEqual(calls["count"], 2)
        self.assertEqual(len(sleeps), 1)

    def test_fallback_model_order(self):
        def handler(kwargs):
            if kwargs["model"] == "gpt-5.2":
                raise ValueError("primary failed")
            return _DummyResponse(_narrative_payload())

        layer = IntelligenceLayer(
            settings=self.base_settings,
            cache_dir=self.cache_dir,
            client_factory=lambda _base, _key: _DummyClient(handler),
        )
        with patch.dict(os.environ, {"LITELLM_API_BASE": "http://test", "LITELLM_API_KEY": "k"}, clear=True):
            result = layer.run(self._minimal_context())
        self.assertEqual(result.meta[SECTION_TRANSFER_STRATEGY]["source"], "ai")
        self.assertEqual(result.meta[SECTION_TRANSFER_STRATEGY]["model"], "gemini-3.1-pro-preview")
        self.assertTrue(result.meta[SECTION_TRANSFER_STRATEGY]["fallback_used"])

    def test_strict_schema_failure_falls_back(self):
        def handler(kwargs):
            return _DummyResponse('{"headline":"h","tactical_summary":"s","metric_highlights":["m"]}')

        layer = IntelligenceLayer(
            settings=self.base_settings,
            cache_dir=self.cache_dir,
            client_factory=lambda _base, _key: _DummyClient(handler),
        )
        with patch.dict(os.environ, {"LITELLM_API_BASE": "http://test", "LITELLM_API_KEY": "k"}, clear=True):
            result = layer.run(self._minimal_context())
        self.assertNotIn(SECTION_TRANSFER_STRATEGY, result.payload)
        self.assertEqual(result.meta[SECTION_TRANSFER_STRATEGY]["source"], "deterministic")

    def test_per_section_fallback_isolation(self):
        settings = dict(self.base_settings)
        settings["sections"] = dict(self.base_settings["sections"])
        settings["sections"]["season_insights"] = True
        settings["max_tokens"] = {"transfer_strategy": 600, "season_insights": 600}

        def handler(kwargs):
            user_prompt = kwargs["messages"][1]["content"]
            if "Section: season_insights" in user_prompt:
                return _DummyResponse('{"bad":"payload"}')
            return _DummyResponse(_narrative_payload())

        layer = IntelligenceLayer(
            settings=settings,
            cache_dir=self.cache_dir,
            client_factory=lambda _base, _key: _DummyClient(handler),
        )
        with patch.dict(os.environ, {"LITELLM_API_BASE": "http://test", "LITELLM_API_KEY": "k"}, clear=True):
            result = layer.run(self._minimal_context())

        self.assertEqual(result.meta[SECTION_TRANSFER_STRATEGY]["source"], "ai")
        self.assertEqual(result.meta[SECTION_SEASON_INSIGHTS]["source"], "deterministic")

    def test_cache_hit_avoids_network_call(self):
        calls = {"count": 0}

        def handler(kwargs):
            calls["count"] += 1
            return _DummyResponse(_narrative_payload())

        layer = IntelligenceLayer(
            settings=self.base_settings,
            cache_dir=self.cache_dir,
            client_factory=lambda _base, _key: _DummyClient(handler),
        )
        env = {"LITELLM_API_BASE": "http://test", "LITELLM_API_KEY": "k"}
        with patch.dict(os.environ, env, clear=True):
            first = layer.run(self._minimal_context())
            second = layer.run(self._minimal_context())

        self.assertEqual(first.meta[SECTION_TRANSFER_STRATEGY]["source"], "ai")
        self.assertEqual(second.meta[SECTION_TRANSFER_STRATEGY]["source"], "ai")
        self.assertTrue(second.meta[SECTION_TRANSFER_STRATEGY]["from_cache"])
        self.assertEqual(calls["count"], 1)

    def test_chip_schema_validation(self):
        settings = dict(self.base_settings)
        settings["sections"] = {k: False for k in self.base_settings["sections"]}
        settings["sections"]["chip_usage_strategy"] = True
        settings["max_tokens"] = {"chip_usage_strategy": 700}

        def handler(kwargs):
            return _DummyResponse(_chip_payload())

        layer = IntelligenceLayer(
            settings=settings,
            cache_dir=self.cache_dir,
            client_factory=lambda _base, _key: _DummyClient(handler),
        )
        env = {"LITELLM_API_BASE": "http://test", "LITELLM_API_KEY": "k"}
        with patch.dict(os.environ, env, clear=True):
            result = layer.run(self._minimal_context())

        chip_payload = result.payload.get("chip_usage_strategy", {})
        self.assertIn("chip_recommendations", chip_payload)
        self.assertEqual(chip_payload["chip_recommendations"]["wildcard"]["recommendation"], "WC")


if __name__ == "__main__":
    unittest.main()
