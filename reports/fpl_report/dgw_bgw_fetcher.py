"""
Module for fetching unannounced Double Gameweek (DGW) and Blank Gameweek (BGW)
predictions using Perplexity Sonar via LiteLLM.
"""

from typing import Dict, List, Any, Optional
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from utils.config import INTELLIGENCE, SEASON

SCHEMA_VERSION = "1"

def fetch_dgw_bgw_intelligence(current_gw: int, session_cache=None) -> Dict[str, List[Dict]]:
    """
    Fetch DGW/BGW predictions from FPL community sources via Perplexity Sonar.
    
    Returns:
        Dict with 'bgw' and 'dgw' containing lists of predicted events.
    """
    settings = INTELLIGENCE.get("dgw_bgw_search", {})
    if not settings.get("enabled", False):
        return {"bgw": [], "dgw": []}
        
    base_url = os.getenv("LITELLM_API_BASE", "").strip()
    api_key = os.getenv("LITELLM_API_KEY", "").strip()
    
    if not base_url or not api_key:
        return {"bgw": [], "dgw": []}

    # Try cache first
    cache_ttl = settings.get("cache_ttl_seconds", 21600)
    cache_key = f"dgw_bgw_predictions_{SEASON}_{current_gw}"
    
    if session_cache:
        cached = session_cache.get(cache_key)
        if cached:
            # Check TTL (if session cache doesn't handle TTL for this specific key)
            return cached
            
    # Alternatively, use a local file cache for this specific prediction
    cache_dir = Path(__file__).parent.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "dgw_bgw_predictions.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
                cached_time = datetime.fromisoformat(data.get("timestamp", "2000-01-01T00:00:00"))
                if datetime.now() - cached_time < timedelta(seconds=cache_ttl):
                    if data.get("current_gw") == current_gw and data.get("season") == SEASON:
                        return data.get("predictions", {"bgw": [], "dgw": []})
        except Exception:
            pass

    # Fetch from API
    model = settings.get("model", "perplexity/sonar")
    
    prompt = f"""
You are an FPL expert for the {SEASON} season.
Identify upcoming Blank Gameweeks (BGW) and Double Gameweeks (DGW) that are expected but might not be officially confirmed yet.
Look for community predictions (e.g., from Ben Crellin, Fantasy Football Scout, etc.).
Only include gameweeks strictly greater than {current_gw}.
Respond ONLY with a JSON object matching this schema exactly, no markdown formatting or extra text:

{{
    "bgw": [
        {{
            "gw": 28,
            "teams_missing": 2,
            "confidence": "high",
            "reason": "FA Cup clash"
        }}
    ],
    "dgw": [
        {{
            "gw": 34,
            "teams_doubled": 4,
            "confidence": "medium",
            "reason": "Rescheduled fixtures"
        }}
    ]
}}
    """
    
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful FPL assistant that returns strictly valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            timeout=30
        )
        
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if "\n" in content:
                content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
        predictions = json.loads(content)
        
        # Validate shape
        if not isinstance(predictions, dict):
            predictions = {"bgw": [], "dgw": []}
            
        bgws = predictions.get("bgw", [])
        dgws = predictions.get("dgw", [])
        
        valid_predictions = {
            "bgw": [b for b in bgws if isinstance(b, dict) and isinstance(b.get("gw"), int)],
            "dgw": [d for d in dgws if isinstance(d, dict) and isinstance(d.get("gw"), int)]
        }
        
        # Save to cache
        try:
            with open(cache_file, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "current_gw": current_gw,
                    "season": SEASON,
                    "predictions": valid_predictions
                }, f)
        except Exception:
            pass
            
        return valid_predictions
        
    except Exception as e:
        print(f"[WARN] Failed to fetch DGW/BGW predictions: {e}")
        return {"bgw": [], "dgw": []}


def merge_bgw_dgw_data(confirmed: Dict, predicted: Dict) -> Dict:
    """
    Merges confirmed BGW/DGW data with predicted data.
    Confirmed data always takes precedence for a given gameweek.
    Entries are tagged with a 'predicted' boolean flag.
    """
    result = {
        "bgw": [],
        "dgw": [],
        "normal": confirmed.get("normal", [])
    }
    
    # Process BGWs
    confirmed_bgw_gws = set()
    for b in confirmed.get("bgw", []):
        b_copy = dict(b)
        b_copy["predicted"] = False
        result["bgw"].append(b_copy)
        confirmed_bgw_gws.add(b.get("gw"))
        
    for b in predicted.get("bgw", []):
        gw = b.get("gw")
        if gw and gw not in confirmed_bgw_gws:
            result["bgw"].append({
                "gw": gw,
                "teams_missing": b.get("teams_missing", 0),
                "predicted": True,
                "confidence": b.get("confidence", "unknown"),
                "reason": b.get("reason", "")
            })
            
    # Process DGWs
    confirmed_dgw_gws = set()
    for d in confirmed.get("dgw", []):
        d_copy = dict(d)
        d_copy["predicted"] = False
        result["dgw"].append(d_copy)
        confirmed_dgw_gws.add(d.get("gw"))
        
    for d in predicted.get("dgw", []):
        gw = d.get("gw")
        if gw and gw not in confirmed_dgw_gws:
            result["dgw"].append({
                "gw": gw,
                "teams_doubled": d.get("teams_doubled", 0),
                "predicted": True,
                "confidence": d.get("confidence", "unknown"),
                "reason": d.get("reason", "")
            })
            
    # Sort by gameweek
    result["bgw"].sort(key=lambda x: x.get("gw", 0))
    result["dgw"].sort(key=lambda x: x.get("gw", 0))
    
    return result
