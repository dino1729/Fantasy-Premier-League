"""Optional intelligence layer for narrative report sections.

This module generates structured narrative content via a LiteLLM gateway
using an OpenAI-compatible API. It is designed to be fully optional and to
fail closed: any error returns deterministic fallback for that section.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


SECTION_TRANSFER_STRATEGY = "transfer_strategy"
SECTION_WILDCARD_DRAFT = "wildcard_draft"
SECTION_FREE_HIT_DRAFT = "free_hit_draft"
SECTION_CHIP_USAGE_STRATEGY = "chip_usage_strategy"
SECTION_SEASON_INSIGHTS = "season_insights"

SECTION_KEYS = [
    SECTION_TRANSFER_STRATEGY,
    SECTION_WILDCARD_DRAFT,
    SECTION_FREE_HIT_DRAFT,
    SECTION_CHIP_USAGE_STRATEGY,
    SECTION_SEASON_INSIGHTS,
]

PROMPT_VERSION = "1"
SCHEMA_VERSION = "1"


@dataclass
class IntelligenceResult:
    payload: Dict[str, Any]
    meta: Dict[str, Any]


class IntelligenceLayer:
    """Generates section-specific narrative intelligence via LLM gateway."""

    def __init__(
        self,
        settings: Dict[str, Any],
        cache_dir: Optional[Path] = None,
        logger: Optional[Callable[[str], None]] = None,
        sleep_func: Callable[[float], None] = time.sleep,
        client_factory: Optional[Callable[[str, str], Any]] = None,
    ):
        self.settings = settings or {}
        self.logger = logger
        self.sleep_func = sleep_func
        self.client_factory = client_factory
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / "cache"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / "intelligence_cache.json"
        self._cache = self._load_cache()

    def run(self, context: Dict[str, Any]) -> IntelligenceResult:
        """Generate intelligence payloads for enabled sections."""
        payload: Dict[str, Any] = {}
        meta: Dict[str, Any] = {}

        if not self.settings.get("enabled", False):
            for section in SECTION_KEYS:
                meta[section] = self._deterministic_meta("global_disabled")
            return IntelligenceResult(payload=payload, meta=meta)

        base_url = os.getenv("LITELLM_API_BASE", "").strip()
        api_key = os.getenv("LITELLM_API_KEY", "").strip()
        if not base_url or not api_key:
            self._log("Intelligence disabled at runtime: missing LITELLM_API_BASE or LITELLM_API_KEY")
            for section in SECTION_KEYS:
                meta[section] = self._deterministic_meta("credentials_missing")
            return IntelligenceResult(payload=payload, meta=meta)

        sections_cfg = self.settings.get("sections", {})
        for section in SECTION_KEYS:
            enabled = bool(sections_cfg.get(section, True))
            if not enabled:
                meta[section] = self._deterministic_meta("section_disabled")
                continue

            facts = self._build_facts(section, context)
            if not facts:
                meta[section] = self._deterministic_meta("section_facts_unavailable")
                continue

            section_payload, section_meta = self._generate_section(
                section=section,
                facts=facts,
                base_url=base_url,
                api_key=api_key,
            )
            if section_payload is not None:
                payload[section] = section_payload
            meta[section] = section_meta

        self._save_cache()
        return IntelligenceResult(payload=payload, meta=meta)

    def _generate_section(
        self,
        section: str,
        facts: Dict[str, Any],
        base_url: str,
        api_key: str,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        max_tokens = int((self.settings.get("max_tokens") or {}).get(section, 1000))
        retries = int(self.settings.get("retries", 2))
        timeout_seconds = int(self.settings.get("timeout_seconds", 120))
        primary_model = str(self.settings.get("model", "gpt-5.2")).strip()
        fallback_models = list(self.settings.get("fallback_models", []))
        models = [primary_model] + [m for m in fallback_models if m and m != primary_model]

        facts_hash = self._hash_json(facts)
        last_error = "unknown_error"
        fallback_used = False

        for m_idx, model in enumerate(models):
            if m_idx > 0:
                fallback_used = True
                self._log(f"Intelligence[{section}] switching to fallback model={model}")

            cache_key = self._cache_key(section=section, model=model, facts_hash=facts_hash)
            cached = self._cache_get(cache_key)
            if cached is not None:
                self._log(f"Intelligence[{section}] cache hit model={model}")
                return cached, {
                    "source": "ai",
                    "model": model,
                    "fallback_used": fallback_used,
                    "from_cache": True,
                    "label": f"Intelligence layer enabled - model: {model}",
                }

            for attempt in range(retries + 1):
                start = time.time()
                try:
                    raw = self._call_gateway(
                        model=model,
                        base_url=base_url,
                        api_key=api_key,
                        section=section,
                        facts=facts,
                        max_tokens=max_tokens,
                        timeout_seconds=timeout_seconds,
                    )
                    parsed = self._parse_json(raw)
                    validated = self._validate_schema(section, parsed)
                    if validated is None:
                        raise ValueError("schema_validation_failed")

                    latency = time.time() - start
                    self._log(f"Intelligence[{section}] model={model} latency={latency:.2f}s")
                    self._cache_set(cache_key, validated)
                    return validated, {
                        "source": "ai",
                        "model": model,
                        "fallback_used": fallback_used,
                        "from_cache": False,
                        "label": f"Intelligence layer enabled - model: {model}",
                    }
                except Exception as exc:  # noqa: BLE001
                    latency = time.time() - start
                    last_error = str(exc)
                    self._log(
                        f"Intelligence[{section}] model={model} attempt={attempt + 1}/{retries + 1} "
                        f"latency={latency:.2f}s error={last_error}"
                    )
                    if attempt < retries:
                        sleep_seconds = (2 ** attempt) + (0.1 * (attempt + 1))
                        self.sleep_func(sleep_seconds)

        return None, self._deterministic_meta(last_error)

    def _call_gateway(
        self,
        model: str,
        base_url: str,
        api_key: str,
        section: str,
        facts: Dict[str, Any],
        max_tokens: int,
        timeout_seconds: int,
    ) -> str:
        client = self._build_client(base_url=base_url, api_key=api_key)
        system_prompt, user_prompt = self._build_prompts(section=section, facts=facts)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            timeout=timeout_seconds,
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ValueError("empty_response_choices")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if not content:
            raise ValueError("empty_response_content")
        return content

    def _build_client(self, base_url: str, api_key: str) -> Any:
        if self.client_factory is not None:
            return self.client_factory(base_url, api_key)
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"openai_sdk_unavailable: {exc}") from exc
        return OpenAI(base_url=base_url, api_key=api_key)

    def _build_prompts(self, section: str, facts: Dict[str, Any]) -> Tuple[str, str]:
        schema = self._schema_template(section)
        system_prompt = (
            "You are an FPL strategy analyst. Return ONLY valid JSON, no markdown. "
            "Use concise analyst-direct tone. Every key recommendation must reference "
            "specific metrics from provided facts."
        )
        user_prompt = (
            f"Section: {section}\n"
            f"Schema version: {SCHEMA_VERSION}\n"
            "Return JSON matching this schema shape exactly:\n"
            f"{json.dumps(schema, ensure_ascii=True)}\n\n"
            "Facts:\n"
            f"{json.dumps(facts, ensure_ascii=True)}"
        )
        return system_prompt, user_prompt

    def _schema_template(self, section: str) -> Dict[str, Any]:
        base = {
            "headline": "string",
            "tactical_summary": "string",
            "metric_highlights": ["string"],
            "actions": ["string"],
            "risks": ["string"],
        }
        if section == SECTION_CHIP_USAGE_STRATEGY:
            base["chip_recommendations"] = {
                "wildcard": {"recommendation": "string", "trigger": "string"},
                "freehit": {"recommendation": "string", "trigger": "string"},
                "bboost": {"recommendation": "string", "trigger": "string"},
                "3xc": {"recommendation": "string", "trigger": "string"},
            }
        return base

    def _validate_schema(self, section: str, payload: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return None
        required_keys = ["headline", "tactical_summary", "metric_highlights", "actions", "risks"]
        for key in required_keys:
            if key not in payload:
                return None
        if not isinstance(payload.get("headline"), str) or not payload["headline"].strip():
            return None
        if not isinstance(payload.get("tactical_summary"), str) or not payload["tactical_summary"].strip():
            return None
        for list_key in ["metric_highlights", "actions", "risks"]:
            value = payload.get(list_key)
            if not isinstance(value, list) or not value:
                return None
            if not all(isinstance(item, str) and item.strip() for item in value):
                return None

        normalized: Dict[str, Any] = {
            "headline": payload["headline"].strip(),
            "tactical_summary": payload["tactical_summary"].strip(),
            "metric_highlights": [str(x).strip() for x in payload["metric_highlights"] if str(x).strip()],
            "actions": [str(x).strip() for x in payload["actions"] if str(x).strip()],
            "risks": [str(x).strip() for x in payload["risks"] if str(x).strip()],
        }
        if not normalized["metric_highlights"] or not normalized["actions"] or not normalized["risks"]:
            return None

        if section == SECTION_CHIP_USAGE_STRATEGY:
            chips = payload.get("chip_recommendations")
            if not isinstance(chips, dict):
                return None
            normalized_chips: Dict[str, Dict[str, str]] = {}
            for chip_key in ["wildcard", "freehit", "bboost", "3xc"]:
                chip_payload = chips.get(chip_key)
                if not isinstance(chip_payload, dict):
                    return None
                rec = chip_payload.get("recommendation")
                trig = chip_payload.get("trigger")
                if not isinstance(rec, str) or not rec.strip():
                    return None
                if not isinstance(trig, str) or not trig.strip():
                    return None
                normalized_chips[chip_key] = {
                    "recommendation": rec.strip(),
                    "trigger": trig.strip(),
                }
            normalized["chip_recommendations"] = normalized_chips

        return normalized

    def _parse_json(self, raw_content: str) -> Dict[str, Any]:
        text = (raw_content or "").strip()
        if text.startswith("```"):
            # Strip fenced markdown if provider adds it despite instruction.
            text = text.strip("`")
            if "\n" in text:
                text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Best-effort recovery: locate outermost object.
        first = text.find("{")
        last = text.rfind("}")
        if first >= 0 and last > first:
            parsed = json.loads(text[first : last + 1])
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("invalid_json_payload")

    def _build_facts(self, section: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if section == SECTION_TRANSFER_STRATEGY:
            return self._facts_transfer_strategy(context)
        if section == SECTION_WILDCARD_DRAFT:
            return self._facts_wildcard_draft(context)
        if section == SECTION_FREE_HIT_DRAFT:
            return self._facts_free_hit_draft(context)
        if section == SECTION_CHIP_USAGE_STRATEGY:
            return self._facts_chip_usage(context)
        if section == SECTION_SEASON_INSIGHTS:
            return self._facts_season_insights(context)
        return {}

    def _facts_transfer_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        data = context.get("multi_week_strategy") or {}
        ev = data.get("expected_value") or {}
        recs = data.get("immediate_recommendations") or []
        rec_names = []
        for r in recs[:3]:
            out_name = (r.get("out_player") or {}).get("name")
            in_options = r.get("in_options") or []
            in_name = in_options[0].get("name") if in_options else None
            if out_name and in_name:
                rec_names.append(f"{out_name}->{in_name}")
        return {
            "current_gw": data.get("current_gameweek"),
            "horizon": data.get("planning_horizon"),
            "current_xp": ev.get("current_squad"),
            "optimized_xp": ev.get("optimized_squad"),
            "potential_gain": ev.get("potential_gain"),
            "model_confidence": data.get("model_confidence"),
            "model_metrics": data.get("model_metrics") or {},
            "mip_status": (data.get("mip_recommendation") or {}).get("status"),
            "immediate_recommendations": rec_names,
        }

    def _facts_wildcard_draft(self, context: Dict[str, Any]) -> Dict[str, Any]:
        wc = context.get("wildcard_team") or {}
        budget = wc.get("budget") or {}
        ev = wc.get("ev_analysis") or {}
        xi = wc.get("starting_xi") or []
        top_xi = [p.get("name") for p in sorted(xi, key=lambda p: p.get("xp_5gw", 0), reverse=True)[:5] if p.get("name")]
        return {
            "formation": wc.get("formation"),
            "budget_total": budget.get("total"),
            "budget_spent": budget.get("spent"),
            "budget_itb": budget.get("remaining"),
            "captain": (wc.get("captain") or {}).get("name"),
            "vice_captain": (wc.get("vice_captain") or {}).get("name"),
            "current_xp": ev.get("current_squad_xp"),
            "optimized_xp": ev.get("optimized_xp"),
            "potential_gain": ev.get("potential_gain"),
            "top_xi_by_xp": top_xi,
        }

    def _facts_free_hit_draft(self, context: Dict[str, Any]) -> Dict[str, Any]:
        fh = context.get("free_hit_team") or {}
        budget = fh.get("budget") or {}
        ev = fh.get("ev_analysis") or {}
        league = fh.get("league_analysis") or {}
        differentials = [d.get("name") for d in (league.get("differentials") or [])[:5] if d.get("name")]
        return {
            "target_gw": fh.get("target_gw"),
            "strategy": fh.get("strategy"),
            "formation": fh.get("formation"),
            "budget_total": budget.get("total"),
            "budget_spent": budget.get("spent"),
            "budget_itb": budget.get("remaining"),
            "captain": (fh.get("captain") or {}).get("name"),
            "vice_captain": (fh.get("vice_captain") or {}).get("name"),
            "current_xp": ev.get("current_squad_xp"),
            "optimized_xp": ev.get("optimized_xp"),
            "potential_gain": ev.get("potential_gain"),
            "league_sample_size": league.get("sample_size"),
            "differentials": differentials,
        }

    def _facts_chip_usage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        analysis = context.get("chip_analysis") or {}
        chips = analysis.get("chips") or {}
        
        # Get simplified BB projections
        bb_proj = None
        if "bboost" in chips and "best_dgw" in chips["bboost"]:
            best = chips["bboost"]["best_dgw"]
            if best:
                bb_proj = {
                    "dgw": best.get("gw"),
                    "total_projected": best.get("total_projected")
                }
                
        base = {
            "current_gw": analysis.get("current_gw"),
            "half": analysis.get("half"),
            "chips_remaining_display": analysis.get("chips_remaining_display"),
            "deadline_warning": (analysis.get("deadline_warning") or {}).get("message"),
            "squad_issue_summary": (analysis.get("squad_issues") or {}).get("summary"),
            "chip_recommendations": {},
            "triggers": analysis.get("triggers") or [],
            "bgw_calendar": [{"gw": b.get("gw"), "teams_missing": b.get("teams_missing"), "predicted": b.get("predicted", False)} for b in analysis.get("bgws") or []],
            "dgw_calendar": [{"gw": d.get("gw"), "teams_doubled": d.get("teams_doubled"), "predicted": d.get("predicted", False)} for d in analysis.get("dgws") or []],
            "synergies": [s.get("strategy") for s in analysis.get("synergies") or []],
            "bb_projections": bb_proj,
        }
        for chip_key in ["wildcard", "freehit", "bboost", "3xc"]:
            c = chips.get(chip_key) or {}
            base["chip_recommendations"][chip_key] = {
                "urgency": c.get("urgency", "low"),
                "recommendation": c.get("recommendation"),
                "target_gw": c.get("target_gw"),
            }
        return base

    def _facts_season_insights(self, context: Dict[str, Any]) -> Dict[str, Any]:
        squad_analysis = context.get("squad_analysis") or []
        gw_history = context.get("gw_history") or []
        total_pts = sum(gw.get("points", 0) for gw in gw_history)
        gws_played = len(gw_history)
        avg = (total_pts / gws_played) if gws_played else 0.0
        projected_total = total_pts + (avg * max(0, 38 - gws_played))

        top_name = None
        top_points = None
        if squad_analysis:
            sorted_points = sorted(
                squad_analysis,
                key=lambda x: (x.get("raw_stats") or {}).get("total_points", 0),
                reverse=True,
            )
            if sorted_points:
                top_name = sorted_points[0].get("name")
                top_points = (sorted_points[0].get("raw_stats") or {}).get("total_points", 0)

        return {
            "gameweeks_played": gws_played,
            "total_points": total_pts,
            "avg_points_per_gw": round(avg, 2),
            "projected_final_points": round(projected_total, 1),
            "top_performer": top_name,
            "top_performer_points": top_points,
        }

    def _cache_key(self, section: str, model: str, facts_hash: str) -> str:
        material = {
            "section": section,
            "prompt_version": PROMPT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "model": model,
            "facts_hash": facts_hash,
        }
        return self._hash_json(material)

    def _hash_json(self, value: Any) -> str:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load_cache(self) -> Dict[str, Any]:
        if not self.cache_path.exists():
            return {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}

    def _save_cache(self) -> None:
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=True)
        except Exception as exc:  # noqa: BLE001
            self._log(f"Intelligence cache save error: {exc}")

    def _cache_get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        entry = self._cache.get(cache_key)
        if not isinstance(entry, dict):
            return None
        created_at = entry.get("created_at")
        ttl = int(self.settings.get("cache_ttl_seconds", 86400))
        if not isinstance(created_at, str):
            return None
        try:
            created_dt = datetime.fromisoformat(created_at)
        except Exception:
            return None
        if datetime.now() - created_dt > timedelta(seconds=ttl):
            self._cache.pop(cache_key, None)
            return None
        value = entry.get("value")
        return value if isinstance(value, dict) else None

    def _cache_set(self, cache_key: str, value: Dict[str, Any]) -> None:
        self._cache[cache_key] = {
            "created_at": datetime.now().isoformat(),
            "value": value,
        }

    def _deterministic_meta(self, reason: str) -> Dict[str, Any]:
        return {
            "source": "deterministic",
            "model": None,
            "fallback_used": False,
            "from_cache": False,
            "error": reason,
        }

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)

