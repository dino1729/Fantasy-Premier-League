#!/usr/bin/env python3
"""FPL Report Generator CLI

Generates comprehensive LaTeX/PDF reports for Fantasy Premier League teams.

Usage:
    python generate_fpl_report.py --team 847569
    python generate_fpl_report.py --team 847569 --gw 17
    python generate_fpl_report.py --team 847569 --no-pdf
    python generate_fpl_report.py --team 847569 --compare 21023 6696002 223259
"""

import argparse
import subprocess
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime
import copy
from typing import Dict, List, Optional

if 'LOKY_MAX_CPU_COUNT' not in os.environ:
    os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)

from reports.fpl_report.data_fetcher import (
    FPLDataFetcher, 
    build_competitive_dataset,
    get_league_entry_ids,
    compute_league_ownership,
    get_top_global_teams
)
from reports.fpl_report.player_analyzer import PlayerAnalyzer
from reports.fpl_report.transfer_recommender import TransferRecommender
from reports.fpl_report.latex_generator import LaTeXReportGenerator
from reports.fpl_report.plot_generator import PlotGenerator
from reports.fpl_report.transfer_strategy import TransferStrategyPlanner, WildcardOptimizer, FreeHitOptimizer
from reports.fpl_report.cache_manager import CacheManager
from reports.fpl_report.session_cache import SessionCacheManager
from reports.fpl_report.intelligence_layer import IntelligenceLayer
from etl.fetchers import FPLCoreInsightsFetcher

# Import centralized configuration
from utils.config import (
    TEAM_ID as DEFAULT_TEAM_ID,
    COMPETITORS as DEFAULT_COMPETITORS,
    GAMEWEEK as DEFAULT_GAMEWEEK,
    SEASON as DEFAULT_SEASON,
    FREE_HIT_TARGET_GW,
    FREE_HIT_STRATEGY,
    WILDCARD_STRATEGY,
    TRANSFER_HORIZON,
    FREE_TRANSFERS_OVERRIDE,
    MIP_ENABLED,
    MIP_TIME_LIMIT,
    MIP_CANDIDATE_POOL,
    LEAGUE_ID,
    LEAGUE_SAMPLE_SIZE,
    NO_COMPETITIVE,
    VERBOSE,
    CACHE,
    TOP_GLOBAL_COUNT,
    INTELLIGENCE,
)


def parse_args():
    """Parse command line arguments.
    
    Most defaults come from config.yml. CLI args override config values.
    """
    parser = argparse.ArgumentParser(
        description='Generate FPL team analysis report (config: reports/config.yml)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script reads settings from reports/config.yml by default.
Command-line arguments override config file values.

Run via: ./reports/run_report.sh
        """
    )
    parser.add_argument('--team', '-t', type=int, default=DEFAULT_TEAM_ID,
                        help=f'FPL team ID (config: {DEFAULT_TEAM_ID})')
    parser.add_argument('--gw', '-g', type=int, default=DEFAULT_GAMEWEEK,
                        help='Gameweek to analyze (config or auto-detect)')
    parser.add_argument('--season', '-s', type=str, default=DEFAULT_SEASON,
                        help=f'Season folder name (config: {DEFAULT_SEASON})')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output filename (default: report_{team_id}.tex)')
    parser.add_argument('--no-pdf', action='store_true',
                        help='Skip PDF compilation')
    parser.add_argument('--verbose', '-v', action='store_true', default=VERBOSE,
                        help='Verbose output')
    parser.add_argument('--compare', '-c', type=int, nargs='*', default=None,
                        help='Entry IDs for competitive analysis (config: competitors)')
    parser.add_argument('--no-competitive', action='store_true', default=NO_COMPETITIVE,
                        help='Skip competitive analysis section')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching (fetch fresh data)')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear all cached data before running')
    parser.add_argument('--cache-stats', action='store_true',
                        help='Show cache statistics and exit')
    # Free Hit options
    parser.add_argument('--league-id', '-l', type=int, default=LEAGUE_ID,
                        help='Classic league ID for league-aware Free Hit draft')
    parser.add_argument('--league-sample', type=int, default=LEAGUE_SAMPLE_SIZE,
                        help=f'Number of teams to sample from league (config: {LEAGUE_SAMPLE_SIZE})')
    parser.add_argument('--free-hit-gw', type=int, default=FREE_HIT_TARGET_GW,
                        help='Target gameweek for Free Hit draft')
    parser.add_argument('--free-hit-strategy', type=str, default=FREE_HIT_STRATEGY,
                        choices=['safe', 'balanced', 'aggressive'],
                        help=f'Free Hit differential strategy (config: {FREE_HIT_STRATEGY})')
    # MIP Solver options
    parser.add_argument('--no-mip', action='store_true', default=not MIP_ENABLED,
                        help='Disable MIP solver for transfer optimization')
    parser.add_argument('--mip-time-limit', type=float, default=MIP_TIME_LIMIT,
                        help=f'MIP solver time limit in seconds (config: {MIP_TIME_LIMIT})')
    parser.add_argument('--mip-pool-size', type=int, default=MIP_CANDIDATE_POOL,
                        help=f'Number of candidates per position (config: {MIP_CANDIDATE_POOL})')
    parser.add_argument('--free-transfers', type=int, default=FREE_TRANSFERS_OVERRIDE,
                        help='Number of free transfers available (config or auto-calc)')
    intelligence_group = parser.add_mutually_exclusive_group()
    intelligence_group.add_argument(
        '--intelligence',
        action='store_true',
        help='Enable intelligence layer for narrative report sections'
    )
    intelligence_group.add_argument(
        '--no-intelligence',
        action='store_true',
        help='Disable intelligence layer for narrative report sections'
    )
    parser.add_argument(
        '--intelligence-model',
        type=str,
        default=None,
        help='Primary model for intelligence layer (LiteLLM gateway model id)'
    )
    parser.add_argument(
        '--intelligence-fallback-models',
        type=str,
        default=None,
        help='Comma-separated fallback models for intelligence layer'
    )
    parser.add_argument(
        '--intelligence-sections',
        type=str,
        default=None,
        help='Comma-separated section keys to enable when intelligence is on '
             '(transfer_strategy,wildcard_draft,free_hit_draft,chip_usage_strategy,season_insights)'
    )
    return parser.parse_args()


def log(message: str, verbose: bool = True):
    """Print log message if verbose mode is enabled."""
    if verbose:
        print(f"[INFO] {message}")


def _parse_csv_list(value: Optional[str]) -> List[str]:
    """Parse comma-separated CLI values into a cleaned list."""
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def get_top_global_comparison_ids(
    team_id: int,
    use_cache: bool = True,
    session_cache: Optional[SessionCacheManager] = None,
) -> List[int]:
    """Get entry IDs for user + top N global managers (based on config)."""
    top_teams = get_top_global_teams(n=TOP_GLOBAL_COUNT, use_cache=use_cache, session_cache=session_cache)
    top_entry_ids = [t.get('entry_id') for t in top_teams if t.get('entry_id') is not None]

    # De-duplicate while preserving order
    unique_top_ids: List[int] = []
    for entry_id in top_entry_ids:
        if entry_id not in unique_top_ids:
            unique_top_ids.append(entry_id)

    # Put user's team first for consistent report ordering
    if team_id in unique_top_ids:
        return [team_id] + [eid for eid in unique_top_ids if eid != team_id]

    return [team_id] + unique_top_ids


def get_latest_available_fplcore_gw(
    all_gw_data: Dict[int, Dict],
    requested_gw: int,
    dataset_name: str = "player_gameweek_stats",
    require_finished_fixtures: bool = False,
) -> Optional[int]:
    """Return the latest GW <= requested_gw with non-empty FPL Core dataset."""
    if not all_gw_data or requested_gw is None:
        return None

    available_gws: List[int] = []
    for gw, gw_data in all_gw_data.items():
        try:
            gw_int = int(gw)
        except (TypeError, ValueError):
            continue

        if gw_int > requested_gw or not isinstance(gw_data, dict):
            continue

        dataset = gw_data.get(dataset_name)
        if dataset is None:
            continue
        if hasattr(dataset, "empty") and dataset.empty:
            continue

        if require_finished_fixtures:
            fixtures = gw_data.get("fixtures")
            if fixtures is None or not hasattr(fixtures, "empty") or fixtures.empty:
                continue

            if "finished" in fixtures.columns:
                if not fixtures["finished"].fillna(False).all():
                    continue

        available_gws.append(gw_int)

    return max(available_gws) if available_gws else None


class PhaseTimer:
    """Simple phase-level timer for report generation."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._started_at: Dict[str, float] = {}
        self.measurements: List[Dict[str, float]] = []

    def start(self, phase_name: str):
        self._started_at[phase_name] = time.perf_counter()

    def stop(self, phase_name: str):
        started = self._started_at.pop(phase_name, None)
        if started is None:
            return
        elapsed = time.perf_counter() - started
        self.measurements.append({"phase": phase_name, "seconds": elapsed})
        if self.verbose:
            print(f"[TIME] {phase_name}: {elapsed:.2f}s")

    def print_summary(self):
        if not self.measurements:
            return
        total = sum(item["seconds"] for item in self.measurements)
        print(f"\n{'='*60}")
        print("  Timing Summary")
        print(f"{'='*60}")
        for item in self.measurements:
            phase = item["phase"]
            seconds = item["seconds"]
            pct = (seconds / total * 100.0) if total else 0.0
            print(f"  {phase:<40} {seconds:>7.2f}s ({pct:>5.1f}%)")
        print(f"  {'TOTAL':<40} {total:>7.2f}s")
        print(f"{'='*60}")


def _analyze_squad_player(
    player: Dict,
    analyzer: PlayerAnalyzer,
    fetcher: FPLDataFetcher,
) -> Dict:
    """Analyze one squad player and attach GW history payload for charts."""
    player_id = player['id']
    position = player['position']
    analysis = analyzer.generate_player_summary(player_id, position)
    analysis['position_in_squad'] = player.get('position_in_squad', 0)
    analysis['is_captain'] = player.get('is_captain', False)
    analysis['is_vice_captain'] = player.get('is_vice_captain', False)

    player_history = fetcher.get_player_history(player_id)
    if not player_history.empty:
        analysis['gw_history'] = player_history.to_dict('records')
    else:
        analysis['gw_history'] = []
    return analysis


def _execute_plot_task(plot_gen: PlotGenerator, task: Dict) -> Optional[object]:
    """Execute one plot generation task."""
    method_name = task["method"]
    method = getattr(plot_gen, method_name)
    args = task.get("args", ())
    kwargs = task.get("kwargs", {})
    return method(*args, **kwargs)


def _run_plot_tasks(
    plot_gen: PlotGenerator,
    tasks: List[Dict],
    verbose: bool,
    max_workers: int = 6,
) -> Dict[str, Optional[object]]:
    """Run plot tasks sequentially.

    matplotlib's pyplot state machine is not thread-safe, so plots must be
    generated one at a time to avoid corrupted figures (wrong dimensions,
    elements from different charts merging, etc.).
    """
    if not tasks:
        return {}

    results: Dict[str, Optional[object]] = {}
    for task in tasks:
        task_name = task.get("name", task.get("method", "plot"))
        try:
            results[task_name] = _execute_plot_task(plot_gen, task)
        except Exception as exc:
            log(f"  [WARN] Plot task '{task_name}' failed: {exc}", verbose)
            results[task_name] = None
    return results


def main():
    """Main entry point."""
    args = parse_args()

    # Determine if caching is enabled
    use_cache = not args.no_cache
    
    # Get gameweek early for session cache initialization (will be refined later)
    # For now, use a placeholder - we'll update it after fetching team data
    preliminary_gameweek = args.gw
    
    # Handle cache stats command (show legacy cache manager stats)
    if args.cache_stats:
        # Show session cache stats
        cache_dir = Path(__file__).parent / "reports" / "cache"
        session_files = list(cache_dir.glob("session_*.pkl"))
        
        print(f"\n{'='*60}")
        print(f"  Session Cache Statistics")
        print(f"{'='*60}")
        print(f"  Cache Directory: {cache_dir}")
        print(f"  Total Sessions: {len(session_files)}")
        
        if session_files:
            print(f"\n  Active Sessions:")
            for sf in sorted(session_files, key=lambda x: x.stat().st_mtime, reverse=True):
                size_mb = sf.stat().st_size / (1024 * 1024)
                mtime = datetime.fromtimestamp(sf.stat().st_mtime)
                age = datetime.now() - mtime
                age_str = f"{int(age.total_seconds() / 60)}m ago" if age.total_seconds() < 3600 else f"{int(age.total_seconds() / 3600)}h ago"
                print(f"    - {sf.name}")
                print(f"      Size: {size_mb:.2f} MB | Age: {age_str}")
        
        print(f"{'='*60}\n")
        return
    
    # Handle clear cache command
    if args.clear_cache:
        print("[INFO] Clearing all cache files...")
        cache_dir = Path(__file__).parent / "reports" / "cache"
        deleted = 0

        # Delete all pickle cache files (legacy md5 cache + session cache)
        for pkl_file in cache_dir.glob("*.pkl"):
            try:
                pkl_file.unlink()
                deleted += 1
            except Exception as e:
                print(f"[WARN] Failed to delete {pkl_file.name}: {e}")

        # Delete legacy cache metadata files if present
        for meta_name in ("cache_metadata.json", "session_metadata.json"):
            meta_path = cache_dir / meta_name
            if meta_path.exists():
                try:
                    meta_path.unlink()
                    deleted += 1
                except Exception as e:
                    print(f"[WARN] Failed to delete {meta_path.name}: {e}")

        print(f"[INFO] Cleared {deleted} cache file(s)")
        return
    
    # Validate team_id is provided for report generation
    if not args.team:
        print("[ERROR] --team/-t argument is required for report generation")
        print("Use --cache-stats to view cache statistics without generating a report")
        sys.exit(1)
    
    team_id = args.team
    season = args.season
    verbose = args.verbose
    phase_timer = PhaseTimer(verbose=verbose)

    # Resolve intelligence settings (config defaults + CLI overrides)
    intelligence_config = copy.deepcopy(INTELLIGENCE) if isinstance(INTELLIGENCE, dict) else {}
    intelligence_config.setdefault('enabled', False)
    intelligence_config.setdefault('model', 'gpt-5.2')
    intelligence_config.setdefault('fallback_models', ['gemini-3.1-pro-preview'])
    intelligence_config.setdefault('retries', 2)
    intelligence_config.setdefault('timeout_seconds', 120)
    intelligence_config.setdefault('cache_ttl_seconds', 86400)
    intelligence_config.setdefault('sections', {})
    intelligence_config.setdefault('max_tokens', {})

    if args.intelligence:
        intelligence_config['enabled'] = True
    elif args.no_intelligence:
        intelligence_config['enabled'] = False

    if args.intelligence_model:
        intelligence_config['model'] = args.intelligence_model.strip()

    if args.intelligence_fallback_models is not None:
        intelligence_config['fallback_models'] = _parse_csv_list(args.intelligence_fallback_models)

    valid_intelligence_sections = {
        'transfer_strategy',
        'wildcard_draft',
        'free_hit_draft',
        'chip_usage_strategy',
        'season_insights',
    }
    if args.intelligence_sections is not None:
        requested_sections = _parse_csv_list(args.intelligence_sections)
        normalized_sections = set(requested_sections)
        unknown_sections = sorted(normalized_sections - valid_intelligence_sections)
        if unknown_sections:
            print(f"[WARN] Ignoring unknown intelligence sections: {', '.join(unknown_sections)}")
        enabled_sections = {key: False for key in valid_intelligence_sections}
        for key in sorted(normalized_sections & valid_intelligence_sections):
            enabled_sections[key] = True
        intelligence_config['sections'] = enabled_sections

    print(f"\n{'='*60}")
    print(f"  FPL Report Generator")
    print(f"  Team ID: {team_id} | Season: {season}")
    if use_cache:
        print(f"  Cache: Enabled (Session-based)")
    else:
        print(f"  Cache: Disabled")
    print(
        f"  Intelligence: {'Enabled' if intelligence_config.get('enabled') else 'Disabled'}"
    )
    print(f"{'='*60}\n")

    # Initialize temporary fetcher to get current gameweek if not specified
    phase_timer.start("Gameweek auto-detection")
    if preliminary_gameweek is None:
        temp_fetcher = FPLDataFetcher(team_id, season, use_cache=False)
        preliminary_gameweek = temp_fetcher.get_current_gameweek()
        log(f"Auto-detected gameweek: {preliminary_gameweek}", verbose)
    phase_timer.stop("Gameweek auto-detection")
    
    # Initialize session cache manager
    phase_timer.start("Session cache init")
    session_cache = None
    if use_cache:
        # Get cache settings from centralized config
        session_ttl = CACHE.get('session_ttl', 3600)
        max_sessions = CACHE.get('max_sessions', 10)
        
        session_cache = SessionCacheManager(
            team_id=team_id,
            gameweek=preliminary_gameweek,
            ttl=session_ttl,
            max_sessions=max_sessions,
            enabled=True,
            single_file=True
        )
        log(f"Session cache initialized: {session_cache.session_id}", verbose)
    phase_timer.stop("Session cache init")
    
    # Initialize components
    log("Initializing data fetcher...", verbose)
    fetcher = FPLDataFetcher(team_id, season, use_cache=use_cache, session_cache=session_cache)
    
    # Get current gameweek if not specified
    gameweek = args.gw or fetcher.get_current_gameweek()
    log(f"Analyzing Gameweek {gameweek}", verbose)
    
    # Fetch FPL Core Insights data (enhanced datasets with match stats + Elo)
    phase_timer.start("FPL Core Insights fetch")
    log("Fetching FPL Core Insights data...", verbose)
    # Convert season format from "2025-26" to "2025-2026"
    fpl_core_season = season.replace("-", "-20") if len(season.split("-")[1]) == 2 else season
    fpl_core_fetcher = FPLCoreInsightsFetcher(season=fpl_core_season)
    
    # Extract season start year (e.g., 2025 from "2025-2026")
    current_season_start_year = int(fpl_core_season.split("-")[0])
    
    # Fetch season-level data
    log("  Fetching season-level data...", verbose)
    fpl_core_season_data = fpl_core_fetcher.fetch_all()
    
    # Fetch ALL gameweek data from GW1 to current GW (for historical analysis)
    log(f"  Fetching all gameweek data (GW1-GW{gameweek})...", verbose)
    all_gw_data = fpl_core_fetcher.fetch_all_gameweeks(up_to_gw=gameweek)
    # Prediction context (may include next-GW fixtures), keep historical data immutable
    all_gw_data_pred = copy.deepcopy(all_gw_data)
    
    # Check for pre-trained models (trained with cross-season data)
    # If available, skip fetching previous season data (faster startup)
    prev_season_start_year = current_season_start_year - 1
    prev_fpl_core_season = f"{prev_season_start_year}-{prev_season_start_year + 1}"
    prev_all_gw_data = None
    prev_fpl_core_season_data = None
    
    # Check if pre-trained models exist
    from pathlib import Path
    pretrained_models_dir = Path(__file__).parent / 'models' / 'artifacts' / 'fpl_core'
    pretrained_available = (pretrained_models_dir / 'latest.json').exists()
    
    if pretrained_available:
        log(f"  Pre-trained cross-season models found - skipping previous season data fetch", verbose)
        log(f"    Models will be loaded from: {pretrained_models_dir}", verbose)
    else:
        log(f"  Fetching previous season data ({prev_fpl_core_season}) for training...", verbose)
        try:
            prev_fpl_core_fetcher = FPLCoreInsightsFetcher(season=prev_fpl_core_season)
            prev_fpl_core_season_data = prev_fpl_core_fetcher.fetch_all()
            prev_all_gw_data = prev_fpl_core_fetcher.fetch_all_gameweeks(up_to_gw=38)
            
            if prev_fpl_core_season_data and prev_all_gw_data:
                prev_gw_count = len([gw for gw in prev_all_gw_data.values() 
                                   if gw.get('playermatchstats') is not None])
                log(f"    Previous season: {prev_gw_count} GWs with match stats available", verbose)
            else:
                log(f"    Previous season data not fully available", verbose)
                prev_all_gw_data = None
                prev_fpl_core_season_data = None
        except Exception as e:
            log(f"    Could not fetch previous season data: {e}", verbose)
            prev_all_gw_data = None
            prev_fpl_core_season_data = None
    
    # Fetch next GW fixtures/teams for prediction context (if available)
    next_gw = gameweek + 1
    log(f"  Fetching next GW ({next_gw}) fixture data for predictions...", verbose)
    try:
        next_gw_fixtures = fpl_core_fetcher.get_fixtures(gameweek=next_gw)
        next_gw_teams = fpl_core_fetcher.get_teams(gameweek=next_gw)
        
        if next_gw_fixtures is not None or next_gw_teams is not None:
            all_gw_data_pred[next_gw] = {
                'fixtures': next_gw_fixtures,
                'teams': next_gw_teams,
            }
            log(f"    Next GW fixtures available: {len(next_gw_fixtures) if next_gw_fixtures is not None else 0} rows", verbose)
        else:
            log(f"    Next GW fixtures not yet available", verbose)
    except Exception as e:
        log(f"    Could not fetch next GW fixtures: {e}", verbose)
    
    # Use latest completed GW for per-GW FPL Core charts (current GW may be empty/in-progress)
    fpl_core_effective_gw = get_latest_available_fplcore_gw(
        all_gw_data,
        gameweek,
        require_finished_fixtures=True,
    )
    if fpl_core_effective_gw is None:
        fpl_core_effective_gw = gameweek
        fpl_core_gw_data = {}
    else:
        fpl_core_gw_data = all_gw_data.get(fpl_core_effective_gw, {})
    
    # Log summary
    if verbose:
        print(f"  Season-level data:")
        cache_info_season = fpl_core_fetcher.get_cache_info()
        available_season = sum(1 for info in cache_info_season.values() if info['exists'])
        print(f"    {available_season}/{len(cache_info_season)} datasets cached")
        
        print(f"  Gameweek data (GW1-GW{gameweek}):")
        total_datasets = 0
        cached_datasets = 0
        for gw in range(1, gameweek + 1):
            cache_info_gw = fpl_core_fetcher.get_cache_info(gameweek=gw)
            total_datasets += len(cache_info_gw)
            cached_datasets += sum(1 for info in cache_info_gw.values() if info['exists'])
        print(f"    {cached_datasets}/{total_datasets} datasets cached across {gameweek} gameweeks")
    phase_timer.stop("FPL Core Insights fetch")

    # Fetch team info
    phase_timer.start("Team and history fetch")
    log("Fetching team information...", verbose)
    try:
        team_info = fetcher.get_team_info(gameweek=gameweek)
        print(f"  Team: {team_info['team_name']}")
        print(f"  Manager: {team_info['manager_name']}")
        print(f"  Points: {team_info['overall_points']} | Rank: {team_info['overall_rank']:,}")
    except Exception as e:
        print(f"[ERROR] Failed to fetch team info: {e}")
        sys.exit(1)

    # Fetch GW history
    log("Fetching gameweek history...", verbose)
    gw_history = fetcher.get_gw_history()
    # Filter history if a specific gameweek limit is set
    if gameweek:
        gw_history = [gw for gw in gw_history if gw['event'] <= gameweek]
    print(f"  Gameweeks played: {len(gw_history)}")

    # Fetch full season history for plots
    log("Fetching full season history...", verbose)
    season_history = fetcher.get_season_history()
    if gameweek:
        season_history = [gw for gw in season_history if gw['gameweek'] <= gameweek]
    
    # Fetch current squad
    log("Fetching current squad...", verbose)
    try:
        squad = fetcher.get_current_squad(gameweek)
        print(f"  Squad size: {len(squad)} players")
    except Exception as e:
        print(f"[ERROR] Failed to fetch squad: {e}")
        sys.exit(1)
    phase_timer.stop("Team and history fetch")

    # Initialize analyzer
    log("Initializing player analyzer...", verbose)
    analyzer = PlayerAnalyzer(fetcher)

    # Analyze each player
    phase_timer.start("Squad analysis")
    log("Analyzing squad players...", verbose)
    squad_workers = max(1, min(5, len(squad)))
    log(f"  Running squad analysis with {squad_workers} workers...", verbose)
    with ThreadPoolExecutor(max_workers=squad_workers) as executor:
        futures = [
            executor.submit(_analyze_squad_player, player, analyzer, fetcher)
            for player in squad
        ]
        squad_analysis = [future.result() for future in futures]
    phase_timer.stop("Squad analysis")

    # Add FPL API fixtures as fallback for future GW predictions
    # This allows the FPL Core predictor to use fixture data even when
    # Core Insights hasn't updated with future GW fixtures yet
    fpl_core_season_data['fpl_api_fixtures'] = fetcher.fixtures_df
    
    # Initialize recommender with FPL Core data for enhanced predictions
    # Pass previous season data for cross-season training (if available)
    phase_timer.start("Transfer recommendation engine")
    log("Generating transfer recommendations...", verbose)
    recommender = TransferRecommender(
        fetcher, analyzer,
        use_new_models=True,
        use_fpl_core_predictor=True,
        all_gw_data=all_gw_data_pred,
        fpl_core_season_data=fpl_core_season_data,
        current_gw=gameweek,
        # Cross-season training data (optional)
        prev_all_gw_data=prev_all_gw_data,
        prev_fpl_core_season_data=prev_fpl_core_season_data,
        prev_season_start_year=prev_season_start_year,
        current_season_start_year=current_season_start_year,
    )

    # Get underperformers and recommendations (legacy)
    underperformers = recommender.identify_underperformers(squad_analysis)
    if underperformers:
        print(f"  Found {len(underperformers)} underperforming players")
        recommendations = recommender.get_recommendations(underperformers)
    else:
        print("  No underperformers identified!")
        recommendations = []

    # Get captain picks
    captain_picks = recommender.get_best_captain_picks(squad_analysis)

    # Get chips used and transfers
    chips_used = fetcher.get_chips_used()
    transfers = fetcher.get_transfers()
    
    # Filter for data leakage prevention if generating past report
    if gameweek:
        transfers = [t for t in transfers if int(t.get('event') or 0) <= gameweek]
        chips_used = [c for c in chips_used if int(c.get('event') or 0) <= gameweek]
        
    log(f"  Transfers made: {len(transfers)}", verbose)
    
    # Determine free transfers
    if args.free_transfers is not None:
        free_transfers = args.free_transfers
        print(f"  Using specified Free Transfers: {free_transfers}")
    else:
        try:
            free_transfers = fetcher.calculate_free_transfers(gameweek)
            print(f"  Calculated Free Transfers: {free_transfers}")
        except Exception as e:
            print(f"  [WARN] Failed to calculate FTs: {e}. Defaulting to 1.")
            free_transfers = 1
    phase_timer.stop("Transfer recommendation engine")

    # Generate multi-week transfer strategy
    phase_timer.start("Multi-week transfer strategy")
    log("Generating multi-week transfer strategy...", verbose)
    strategy_planner = TransferStrategyPlanner(fetcher, analyzer, recommender)
    use_mip = not args.no_mip
    try:
        from utils.config import TRANSFER_HORIZON
        multi_week_strategy = strategy_planner.generate_strategy(
            squad_analysis,
            num_weeks=TRANSFER_HORIZON,
            use_mip=use_mip,
            mip_time_limit=args.mip_time_limit,
            mip_candidate_pool=args.mip_pool_size,
            free_transfers=free_transfers,
            current_gw=gameweek
        )
        ev_data = multi_week_strategy.get('expected_value', {})
        print(f"  Current squad 5-GW xP: {ev_data.get('current_squad', 0):.1f}")
        print(f"  Potential gain: +{ev_data.get('potential_gain', 0):.1f} pts")
        print(f"  Model confidence: {multi_week_strategy.get('model_confidence', 'unknown')}")
        
        # Print MIP solver result if available
        mip_rec = multi_week_strategy.get('mip_recommendation')
        if mip_rec:
            mip_status = mip_rec.get('status', 'unknown')
            if mip_status == 'optimal':
                print(f"  MIP Solver: Optimal solution found in {mip_rec.get('solver_time', 0):.2f}s")
                print(f"  Recommended transfers: {mip_rec.get('num_transfers', 0)}")
                print(f"  Expected points (5-GW): {mip_rec.get('expected_points', 0):.1f}")
            elif mip_status == 'unavailable':
                print(f"  MIP Solver: Not available (install sasoptpy + highspy)")
            else:
                print(f"  MIP Solver: {mip_status} - {mip_rec.get('message', '')}")
                
    except Exception as e:
        print(f"[WARNING] Multi-week strategy generation failed: {e}")
        multi_week_strategy = None
    phase_timer.stop("Multi-week transfer strategy")

    # Generate chip analysis for personalized recommendations
    phase_timer.start("Chip opportunities analysis")
    log("Analyzing chip opportunities...", verbose)
    chip_analysis = None
    try:
        chip_analysis = strategy_planner.analyze_chip_opportunities(
            squad_analysis=squad_analysis,
            chips_used=chips_used,
            gw_history=gw_history,
            ml_position=None  # TODO: Add ML position data in Phase 2
        )
        issues = chip_analysis.get('squad_issues', {})
        print(f"  Squad issues detected: {issues.get('summary', 'None')}")
        print(f"  Chips remaining (this half): {chip_analysis.get('chips_remaining_display', '?')}")
        if chip_analysis.get('deadline_warning'):
            print(f"  ⚠ DEADLINE: {chip_analysis['deadline_warning']['message']}")
    except Exception as e:
        print(f"  [WARN] Chip analysis failed: {e}")
        chip_analysis = None
    phase_timer.stop("Chip opportunities analysis")

    # Generate Phase 2 chip projections (BB/TC/FH/WC optimizer-based)
    phase_timer.start("Phase 2 chip projections")
    log("Generating Phase 2 chip projections (optimizer-based)...", verbose)
    try:
        phase2_analysis = strategy_planner.get_phase2_chip_analysis(
            squad_analysis=squad_analysis,
            chips_used=chips_used,
            gw_history=gw_history
        )
        # Merge Phase 2 into chip_analysis
        if chip_analysis is None:
            chip_analysis = {}
        chip_analysis['phase2'] = phase2_analysis

        # Report what was generated
        if phase2_analysis.get('bb_projections'):
            bb = phase2_analysis['bb_projections']
            print(f"  BB Projections: {bb.get('recommendation', 'Available')}")
        if phase2_analysis.get('tc_rankings'):
            tc = phase2_analysis['tc_rankings']
            print(f"  TC Rankings: {tc.get('recommendation', 'Available')}")
        if phase2_analysis.get('fh_squad'):
            fh = phase2_analysis['fh_squad']
            print(f"  FH Squad: {fh.get('recommendation', 'Generated')}")
        if phase2_analysis.get('wc_squad'):
            wc = phase2_analysis['wc_squad']
            print(f"  WC Squad: {wc.get('recommendation', 'Generated')}")
    except Exception as e:
        print(f"  [WARN] Phase 2 chip analysis failed: {e}")
        if chip_analysis is None:
            chip_analysis = {}
        chip_analysis['phase2'] = None
    phase_timer.stop("Phase 2 chip projections")

    # Generate 5-GW predictions for ALL top players (for Wildcard/Free Hit drafts)
    phase_timer.start("Draft candidate predictions")
    log("Generating 5-GW predictions for draft candidates...", verbose)
    all_player_predictions = {}
    try:
        # Get predictor from strategy planner (it's already trained)
        predictor = strategy_planner.predictor
        if predictor and predictor.is_trained:
            # Get top 200 players by total_points as draft candidates
            top_players = fetcher.players_df.nlargest(200, 'total_points')['id'].tolist()
            # Use the wrapper method to handle both predictor types
            all_player_predictions = strategy_planner._predict_multiple_gws(top_players, num_gws=5)
            print(f"  Generated predictions for {len(all_player_predictions)} players")
        else:
            print("  Predictor not trained, skipping draft predictions")
    except Exception as e:
        print(f"  [WARN] Draft predictions failed: {e}")
        all_player_predictions = {}
    phase_timer.stop("Draft candidate predictions")

    # Generate Plots
    phase_timer.start("Plot generation")
    log("Generating visualizations...", verbose)
    output_dir = Path(__file__).parent / "reports" / "plots"
    plot_gen = PlotGenerator(output_dir)

    # Generate core plots in parallel
    base_plot_tasks = [
        {
            "name": "points_per_gw",
            "method": "generate_points_per_gw",
            "args": (gw_history, chips_used),
        },
        {
            "name": "contribution_heatmap",
            "method": "generate_contribution_heatmap",
            "args": (season_history,),
        },
        {
            "name": "treemap",
            "method": "generate_treemap",
            "args": (season_history,),
        },
        {
            "name": "transfer_matrix",
            "method": "generate_transfer_matrix",
            "args": (transfers,),
        },
    ]
    _run_plot_tasks(plot_gen, base_plot_tasks, verbose=verbose)

    # Generate advanced finishing & creativity plots from FPL Core Insights data
    log("Generating advanced finishing & creativity analysis...", verbose)
    squad_ids = [p['id'] for p in squad]
    if fpl_core_effective_gw != gameweek:
        print(
            f"  [INFO] Advanced finishing/creativity charts using latest completed GW{fpl_core_effective_gw} "
            f"(GW{gameweek} data not complete yet)"
        )
    try:
        advanced_plot_tasks = [
            {
                "name": "clinical_wasteful",
                "method": "generate_clinical_wasteful_chart",
                "args": (fpl_core_season_data, fpl_core_gw_data, squad_ids, fpl_core_effective_gw),
            },
            {
                "name": "clinical_wasteful_squad",
                "method": "generate_clinical_wasteful_chart_squad_only",
                "args": (fpl_core_season_data, fpl_core_gw_data, squad_ids, fpl_core_effective_gw),
            },
            {
                "name": "clutch_frustrated",
                "method": "generate_clutch_frustrated_chart",
                "args": (fpl_core_season_data, fpl_core_gw_data, squad_ids, fpl_core_effective_gw),
            },
            {
                "name": "clutch_frustrated_squad",
                "method": "generate_clutch_frustrated_chart_squad_only",
                "args": (fpl_core_season_data, fpl_core_gw_data, squad_ids, fpl_core_effective_gw),
            },
            {
                "name": "usage_output_scatter",
                "method": "generate_usage_output_scatter",
                "args": (all_gw_data, fpl_core_season_data, squad_ids),
            },
            {
                "name": "usage_output_scatter_recent",
                "method": "generate_usage_output_scatter_recent",
                "args": (all_gw_data, fpl_core_season_data, squad_ids),
                "kwargs": {"last_n_gw": 5},
            },
            {
                "name": "usage_output_scatter_squad",
                "method": "generate_usage_output_scatter_squad_only",
                "args": (all_gw_data, fpl_core_season_data, squad_ids, gameweek),
            },
            {
                "name": "usage_output_scatter_squad_recent",
                "method": "generate_usage_output_scatter_squad_recent",
                "args": (all_gw_data, fpl_core_season_data, squad_ids, gameweek),
                "kwargs": {"last_n_gw": 5},
            },
            {
                "name": "defensive_value_scatter",
                "method": "generate_defensive_value_scatter",
                "args": (all_gw_data, fpl_core_season_data, squad_ids),
            },
            {
                "name": "defensive_value_scatter_recent",
                "method": "generate_defensive_value_scatter_recent",
                "args": (all_gw_data, fpl_core_season_data, squad_ids),
                "kwargs": {"last_n_gw": 5},
            },
            {
                "name": "goalkeeper_value_scatter",
                "method": "generate_goalkeeper_value_scatter",
                "args": (all_gw_data, fpl_core_season_data, squad_ids),
            },
            {
                "name": "goalkeeper_value_scatter_recent",
                "method": "generate_goalkeeper_value_scatter_recent",
                "args": (all_gw_data, fpl_core_season_data, squad_ids),
                "kwargs": {"last_n_gw": 5},
            },
        ]
        advanced_results = _run_plot_tasks(plot_gen, advanced_plot_tasks, verbose=verbose)
        if any(value is not None for value in advanced_results.values()):
            log("  Advanced analysis plots generated successfully (league + squad + defensive)", verbose)
        else:
            log("  Warning: Advanced analysis plots were not generated", verbose)
    except Exception as e:
        log(f"  Warning: Could not generate advanced analysis plots: {e}", verbose)

    # Generate hindsight fixture analysis (parallelized task wrapper for consistency)
    log("Generating hindsight fixture analysis...", verbose)
    teams_data = fetcher.bootstrap_data.get('teams', [])
    hindsight_tasks = [
        {
            "name": "hindsight_fixture_analysis",
            "method": "generate_hindsight_fixture_analysis",
            "args": (
                season_history,
                fetcher.fixtures_df,
                teams_data,
            ),
            "kwargs": {
                "start_gw": max(1, gameweek - 10),
                "end_gw": gameweek,
            },
        }
    ]
    _run_plot_tasks(plot_gen, hindsight_tasks, verbose=verbose)
    phase_timer.stop("Plot generation")

    # Determine competitor IDs (used for both competitive analysis and Free Hit ownership)
    if args.compare is not None:
        # User specified --compare (even if empty list means use defaults)
        competitor_ids = args.compare if args.compare else DEFAULT_COMPETITORS
    else:
        # Use default competitors
        competitor_ids = DEFAULT_COMPETITORS

    # Include the main team in the comparison if not already present
    if team_id not in competitor_ids:
        competitor_ids = [team_id] + competitor_ids

    # Competitive Analysis
    phase_timer.start("Competitive analysis")
    competitive_data = None
    if not args.no_competitive:
        log(f"Building competitive analysis for {len(competitor_ids)} teams...", verbose)
        try:
            competitive_data = build_competitive_dataset(
                entry_ids=competitor_ids,
                season=season,
                gameweek=gameweek,
                use_cache=use_cache,
                session_cache=session_cache
            )
            print(f"  Comparing {len(competitive_data)} teams")

            # Generate competitive plots
            log("Generating competitive plots...", verbose)
            competitive_plot_tasks = [
                {
                    "name": "competitive_points_per_gw",
                    "method": "generate_competitive_points_per_gw",
                    "args": (competitive_data,),
                },
                {
                    "name": "competitive_points_progression",
                    "method": "generate_competitive_points_progression",
                    "args": (competitive_data,),
                },
                {
                    "name": "competitive_rank_progression",
                    "method": "generate_competitive_rank_progression",
                    "args": (competitive_data,),
                },
                {
                    "name": "competitive_treemaps",
                    "method": "generate_competitive_treemaps",
                    "args": (competitive_data,),
                },
            ]
            competitive_plot_results = _run_plot_tasks(plot_gen, competitive_plot_tasks, verbose=verbose)
            log("Generating competitor treemaps...", verbose)
            treemap_files = competitive_plot_results.get("competitive_treemaps") or []
            print(f"  Generated {len(treemap_files)} competitor treemaps")
        except Exception as e:
            print(f"[WARNING] Competitive analysis failed: {e}")
            competitive_data = None
    phase_timer.stop("Competitive analysis")

    # Top Global Managers Competitive Analysis
    phase_timer.start("Top global comparison")
    top_global_data = None
    if not args.no_competitive:
        log(f"Building competitive analysis vs Top {TOP_GLOBAL_COUNT} Global Managers...", verbose)
        try:
            global_comparison_ids = get_top_global_comparison_ids(
                team_id=team_id,
                use_cache=use_cache,
                session_cache=session_cache
            )
            top_entry_ids = [eid for eid in global_comparison_ids if eid != team_id]

            if top_entry_ids:
                
                print(f"  Comparing user vs top {len(top_entry_ids)} global managers")
                
                # Build competitive dataset for global comparison
                top_global_data = build_competitive_dataset(
                    entry_ids=global_comparison_ids,
                    season=season,
                    gameweek=gameweek,
                    use_cache=use_cache,
                    session_cache=session_cache
                )
                
                # Generate plots for global comparison (with different filenames)
                log("Generating top global manager plots...", verbose)
                global_plot_tasks = [
                    {
                        "name": "global_points_per_gw",
                        "method": "generate_competitive_points_per_gw",
                        "args": (top_global_data,),
                        "kwargs": {"filename": "global_top_points_per_gw.png"},
                    },
                    {
                        "name": "global_points_progression",
                        "method": "generate_competitive_points_progression",
                        "args": (top_global_data,),
                        "kwargs": {"filename": "global_top_points_progression.png"},
                    },
                    {
                        "name": "global_rank_progression",
                        "method": "generate_competitive_rank_progression",
                        "args": (top_global_data,),
                        "kwargs": {"filename": "global_top_rank_progression.png"},
                    },
                    {
                        "name": "global_treemaps",
                        "method": "generate_competitive_treemaps",
                        "args": (top_global_data,),
                        "kwargs": {"prefix": "global_"},
                    },
                ]
                global_plot_results = _run_plot_tasks(plot_gen, global_plot_tasks, verbose=verbose)

                # Generate treemaps for global top teams
                log("Generating top global manager treemaps...", verbose)
                global_treemap_files = global_plot_results.get("global_treemaps") or []
                print(f"  Generated {len(global_treemap_files)} global top treemaps")
            else:
                print("  [WARN] Could not fetch top global teams")
        except Exception as e:
            print(f"[WARNING] Top global analysis failed: {e}")
            top_global_data = None
    phase_timer.stop("Top global comparison")

    # Compute differential ownership: players owned by top managers but not by user
    differential_targets = set()
    try:
        user_player_ids = {p.get('id', p.get('element', 0)) for p in squad} if squad else set()
        rival_sources = []
        if top_global_data:
            rival_sources.extend(top_global_data)
        if competitive_data:
            rival_sources.extend([t for t in competitive_data if t.get('team_id') != team_id])

        if rival_sources:
            # Count how many rivals own each player
            rival_player_counts = {}
            for rival in rival_sources:
                for p in rival.get('squad', []):
                    pid = p.get('element', p.get('id', 0))
                    if pid and pid not in user_player_ids:
                        rival_player_counts[pid] = rival_player_counts.get(pid, 0) + 1

            # Players owned by 2+ rivals are template convergence targets
            differential_targets = {pid for pid, count in rival_player_counts.items() if count >= 2}
            if differential_targets:
                print(f"  Template convergence targets: {len(differential_targets)} players owned by 2+ rivals")
    except Exception as e:
        print(f"  [WARN] Differential analysis failed: {e}")

    # Generate Wildcard draft squad
    phase_timer.start("Wildcard optimization")
    log("Building Wildcard draft squad...", verbose)
    wildcard_team = None
    try:
        # Calculate total budget: squad value + bank
        team_value = fetcher.get_team_value(gameweek=gameweek)
        bank = fetcher.get_bank(gameweek=gameweek)
        total_budget = team_value + bank
        print(f"  Wildcard budget: {team_value:.1f}m (squad) + {bank:.1f}m (bank) = {total_budget:.1f}m")
        
        # Use all player predictions for draft (includes all top 200 players)
        predictions = all_player_predictions if all_player_predictions else {}
        
        # Get teams and fixtures data for FDR display
        teams_data = fetcher.bootstrap_data.get('teams', [])
        fixtures_list = fetcher.fixtures_df.to_dict('records') if not fetcher.fixtures_df.empty else []
        
        # Build optimal Wildcard squad
        optimizer = WildcardOptimizer(
            players_df=fetcher.players_df,
            total_budget=total_budget,
            predictions=predictions,
            teams_data=teams_data,
            fixtures_data=fixtures_list,
            current_gw=gameweek
        )
        wildcard_team = optimizer.build_squad()
        
        # Calculate current squad 5-GW xP for comparison
        current_squad_5gw_xp = 0.0
        if multi_week_strategy:
            current_squad_5gw_xp = multi_week_strategy.get('expected_value', {}).get('current_squad', 0.0)
        
        # Calculate optimized (wildcard) 5-GW xP from starting XI
        wildcard_5gw_xp = sum(p.get('xp_5gw', 0) for p in wildcard_team.get('starting_xi', []))
        
        # Add EV analysis to wildcard_team
        wildcard_team['ev_analysis'] = {
            'current_squad_xp': round(current_squad_5gw_xp, 1),
            'optimized_xp': round(wildcard_5gw_xp, 1),
            'potential_gain': round(wildcard_5gw_xp - current_squad_5gw_xp, 1),
            'horizon': '5 GWs'
        }
        
        budget_info = wildcard_team.get('budget', {})
        print(f"  Wildcard squad built: {budget_info.get('spent', 0):.1f}m spent, {budget_info.get('remaining', 0):.1f}m ITB")
        print(f"  Formation: {wildcard_team.get('formation', 'N/A')}")
        print(f"  Captain: {wildcard_team.get('captain', {}).get('name', 'N/A')}")
        print(f"  5-GW xP: {current_squad_5gw_xp:.1f} -> {wildcard_5gw_xp:.1f} (+{wildcard_5gw_xp - current_squad_5gw_xp:.1f})")
    except Exception as e:
        print(f"[WARNING] Wildcard squad generation failed: {e}")
        wildcard_team = None
    phase_timer.stop("Wildcard optimization")

    # Generate Free Hit draft squad (league-aware)
    phase_timer.start("Free Hit optimization")
    log("Building Free Hit draft squad...", verbose)
    free_hit_team = None
    league_ownership_data = None
    try:
        # Determine target gameweek for Free Hit (next deadline by default)
        free_hit_gw = args.free_hit_gw
        if free_hit_gw is None:
            # Default to next deadline GW (current + 1, or current if current is not finished)
            events = fetcher.bootstrap_data.get('events', [])
            for event in events:
                if event.get('is_next', False):
                    free_hit_gw = event['id']
                    break
            if free_hit_gw is None:
                free_hit_gw = gameweek + 1
        
        print(f"  Free Hit target: GW{free_hit_gw}")
        
        # Fetch league ownership - from league_id OR competitor IDs
        league_entry_ids = []
        
        if args.league_id:
            # Option 1: Use classic league standings
            log(f"Fetching league ownership from league {args.league_id}...", verbose)
            try:
                league_entry_ids = get_league_entry_ids(
                    league_id=args.league_id,
                    sample_n=args.league_sample,
                    focus_entry_id=team_id,
                    use_cache=use_cache,
                    session_cache=session_cache
                )
                if league_entry_ids:
                    print(f"  Sampled {len(league_entry_ids)} teams from league {args.league_id}")
            except Exception as e:
                print(f"  [WARN] League fetch failed: {e}")
                league_entry_ids = []
        
        # Option 2: Fall back to competitor IDs if no league-id or league fetch failed
        if not league_entry_ids:
            # Use competitor IDs for Free Hit ownership analysis
            if competitor_ids and len(competitor_ids) > 1:
                league_entry_ids = competitor_ids
                print(f"  Using {len(league_entry_ids)} competitor teams for ownership analysis")
            else:
                print("  No competitor teams available, using pure xPts optimization")
        
        # Compute league ownership if we have entry IDs
        # Use current GW for ownership (what squads people have NOW, not the future target GW)
        if league_entry_ids:
            try:
                ownership_gw = gameweek  # Use current GW, not target Free Hit GW
                league_ownership_data = compute_league_ownership(
                    entry_ids=league_entry_ids,
                    gw=ownership_gw,
                    use_cache=use_cache,
                    session_cache=session_cache
                )
                sample_size = league_ownership_data.get('sample_size', 0)
                print(f"  League ownership computed from {sample_size} teams (based on GW{ownership_gw} squads)")
            except Exception as e:
                print(f"  [WARN] League ownership computation failed: {e}")
                league_ownership_data = None
        
        # Calculate total budget (same as wildcard)
        if total_budget is None or total_budget <= 0:
            team_value = fetcher.get_team_value(gameweek=gameweek)
            bank = fetcher.get_bank(gameweek=gameweek)
            total_budget = team_value + bank
        
        print(f"  Free Hit budget: {total_budget:.1f}m")
        print(f"  Primary strategy: {args.free_hit_strategy}")
        
        # Get teams and fixtures data for FDR display
        teams_data = fetcher.bootstrap_data.get('teams', [])
        fixtures_list = fetcher.fixtures_df.to_dict('records') if not fetcher.fixtures_df.empty else []
        
        # Get current squad STARTING XI player IDs (needed for all strategies)
        current_xi_ids = []
        for p in squad_analysis:
            pos_in_squad = p.get('position_in_squad', 0)
            if pos_in_squad >= 1 and pos_in_squad <= 11:  # Starting XI only
                current_xi_ids.append(p.get('player_id'))
        
        # Calculate current squad xP for the target GW (same for all strategies)
        target_gw_offset = free_hit_gw - gameweek
        target_gw_current_xp = 0.0
        predictor = strategy_planner.predictor if strategy_planner else None
        
        if predictor and predictor.is_trained and all_player_predictions:
            for pid in current_xi_ids:
                if pid in all_player_predictions:
                    preds = all_player_predictions[pid].get('predictions', [])
                    if target_gw_offset <= len(preds) and target_gw_offset > 0:
                        target_gw_current_xp += preds[target_gw_offset - 1]
        
        # Build 3 FREE HIT SQUAD PERMUTATIONS with different strategies
        strategies = ['safe', 'balanced', 'aggressive']
        strategy_descriptions = {
            'safe': 'Template Squad - High ownership, minimize risk',
            'balanced': 'Balanced Squad - Mix of template & differentials',
            'aggressive': 'Differential Squad - Low ownership, chase upside'
        }
        
        free_hit_permutations = []
        
        for strat in strategies:
            fh_optimizer = FreeHitOptimizer(
                players_df=fetcher.players_df,
                total_budget=total_budget,
                league_ownership=league_ownership_data,
                strategy=strat,
                teams_data=teams_data,
                fixtures_data=fixtures_list,
                current_gw=gameweek,
                target_gw=free_hit_gw,
                predictions=all_player_predictions
            )
            squad_result = fh_optimizer.build_squad()
            squad_result['target_gw'] = free_hit_gw
            squad_result['strategy'] = strat
            squad_result['strategy_description'] = strategy_descriptions[strat]
            
            # Calculate xP for this permutation's XI
            fh_xi_ids = [p.get('id') for p in squad_result.get('starting_xi', [])]
            target_gw_fh_xp = 0.0
            
            if predictor and predictor.is_trained and all_player_predictions:
                for pid in fh_xi_ids:
                    if pid in all_player_predictions:
                        preds = all_player_predictions[pid].get('predictions', [])
                        if target_gw_offset <= len(preds) and target_gw_offset > 0:
                            target_gw_fh_xp += preds[target_gw_offset - 1]
            
            # Add EV analysis
            squad_result['ev_analysis'] = {
                'current_squad_xp': round(target_gw_current_xp, 1),
                'optimized_xp': round(target_gw_fh_xp, 1),
                'potential_gain': round(target_gw_fh_xp - target_gw_current_xp, 1),
                'horizon': f'GW{free_hit_gw}'
            }
            
            # Add GW comparison data
            squad_result['gw_comparison'] = {
                'target_gw': free_hit_gw,
                'current_squad_xp': round(target_gw_current_xp, 1),
                'free_hit_xp': round(target_gw_fh_xp, 1),
            }
            
            free_hit_permutations.append(squad_result)
            
            budget_info = squad_result.get('budget', {})
            print(f"  [{strat.upper()}] {budget_info.get('spent', 0):.1f}m spent, "
                  f"xP: {target_gw_fh_xp:.1f}, Gain: {target_gw_fh_xp - target_gw_current_xp:+.1f}")
        
        # Use the primary strategy as the "main" free_hit_team for backward compatibility
        primary_idx = strategies.index(args.free_hit_strategy)
        free_hit_team = free_hit_permutations[primary_idx]
        free_hit_team['all_permutations'] = free_hit_permutations
        
        print(f"  Single-GW analysis for target GW{free_hit_gw}")
        
        # Generate the Free Hit comparison plot (using primary strategy)
        gw_comparison = free_hit_team.get('gw_comparison', {})
        if gw_comparison:
            fh_plot_path = plot_gen.generate_free_hit_gw_comparison(gw_comparison)
            if fh_plot_path:
                free_hit_team['gw_plot'] = fh_plot_path
                print(f"  Free Hit analysis plot saved")
        
        budget_info = free_hit_team.get('budget', {})
        league_analysis = free_hit_team.get('league_analysis', {})
        print(f"  Free Hit squad built: {budget_info.get('spent', 0):.1f}m spent, {budget_info.get('remaining', 0):.1f}m ITB")
        print(f"  Formation: {free_hit_team.get('formation', 'N/A')}")
        print(f"  Captain: {free_hit_team.get('captain', {}).get('name', 'N/A')}")
        ev = free_hit_team.get('ev_analysis', {})
        print(f"  GW{free_hit_gw} xP (ML): {ev.get('current_squad_xp', 0):.1f} -> {ev.get('optimized_xp', 0):.1f} (+{ev.get('potential_gain', 0):.1f})")
        print(f"  Differentials: {len(league_analysis.get('differentials', []))}")
        print(f"  Template picks: {len(league_analysis.get('template_picks', []))}")
        
    except Exception as e:
        print(f"[WARNING] Free Hit squad generation failed: {e}")
        free_hit_team = None
    phase_timer.stop("Free Hit optimization")

    # Generate optional intelligence narratives (section-level, model-backed)
    phase_timer.start("Intelligence layer")
    intelligence_payload = {}
    intelligence_meta = {}
    if intelligence_config.get('enabled'):
        log("Generating intelligence layer narratives...", verbose)
        try:
            intelligence_layer = IntelligenceLayer(
                settings=intelligence_config,
                logger=lambda msg: log(msg, verbose),
            )
            intelligence_result = intelligence_layer.run(
                context={
                    'multi_week_strategy': multi_week_strategy,
                    'wildcard_team': wildcard_team,
                    'free_hit_team': free_hit_team,
                    'chip_analysis': chip_analysis,
                    'squad_analysis': squad_analysis,
                    'gw_history': gw_history,
                    'competitive_data': competitive_data,
                    'top_global_data': top_global_data,
                }
            )
            intelligence_payload = intelligence_result.payload
            intelligence_meta = intelligence_result.meta

            sections_cfg = intelligence_config.get('sections', {})
            for section_key, section_meta in intelligence_meta.items():
                if section_meta.get('source') == 'ai':
                    model = section_meta.get('model', '?')
                    cache_label = " (cache)" if section_meta.get('from_cache') else ""
                    print(f"  [AI] {section_key}: {model}{cache_label}")
                elif sections_cfg.get(section_key, True):
                    reason = section_meta.get('error', 'fallback')
                    print(f"  [AI-FALLBACK] {section_key}: deterministic ({reason})")
        except Exception as e:
            print(f"[WARNING] Intelligence layer failed globally: {e}")
            intelligence_payload = {}
            intelligence_meta = {}
    phase_timer.stop("Intelligence layer")

    # Cross-section consistency pass: filter captain picks that conflict with transfers
    if multi_week_strategy and captain_picks:
        mip_rec = multi_week_strategy.get('mip_recommendation')
        if mip_rec and mip_rec.get('status') == 'optimal':
            transfers_out_next_gw = set()
            scenarios = mip_rec.get('scenarios', {})
            # Check each scenario for first-week transfers out
            for scenario_name in ('balanced', 'aggressive', 'conservative'):
                scenario_data = scenarios.get(scenario_name, {})
                scenario_plans = scenario_data.get('weekly_plans', [])
                if scenario_plans:
                    first_week = scenario_plans[0]
                    for p_out in first_week.get('transfers_out', []):
                        out_name = p_out.get('name', '')
                        # Match by name since MIP plans don't always have player IDs
                        for pick in captain_picks:
                            if pick.get('name') == out_name:
                                transfers_out_next_gw.add(pick.get('player_id'))
                    break

            if transfers_out_next_gw:
                original_count = len(captain_picks)
                captain_picks = [p for p in captain_picks
                                 if p.get('player_id') not in transfers_out_next_gw]
                filtered = original_count - len(captain_picks)
                if filtered:
                    print(f"  Consistency: filtered {filtered} captain picks conflicting with transfers")

    # Compute captain ROI analysis from season_history
    captain_roi = None
    if season_history:
        try:
            captain_points_total = 0
            optimal_captain_total = 0
            captain_hit_count = 0
            gws_counted = 0
            for gw_entry in season_history:
                gw_squad = gw_entry.get('squad', [])
                if not gw_squad:
                    continue
                gws_counted += 1
                # Find captain and their bonus points
                captain_base = 0
                best_base = 0
                for p in gw_squad:
                    base_pts = p.get('stats', {}).get('event_points', 0) or 0
                    pos_in_squad = p.get('position_in_squad', 99)
                    if pos_in_squad <= 11:
                        best_base = max(best_base, base_pts)
                    mult = p.get('multiplier', 1)
                    if mult >= 2:  # Captain
                        captain_base = base_pts
                captain_bonus = captain_base  # The extra points from captaincy (1x base)
                optimal_bonus = best_base
                captain_points_total += captain_bonus
                optimal_captain_total += optimal_bonus
                # "Hit" = captain was top-3 scorer among starters
                starter_pts = sorted(
                    [p.get('stats', {}).get('event_points', 0) or 0
                     for p in gw_squad if p.get('position_in_squad', 99) <= 11],
                    reverse=True
                )
                if captain_base >= (starter_pts[2] if len(starter_pts) >= 3 else 0):
                    captain_hit_count += 1

            if gws_counted > 0:
                captain_roi = {
                    'captain_points': captain_points_total,
                    'optimal_points': optimal_captain_total,
                    'points_left_on_table': optimal_captain_total - captain_points_total,
                    'hit_rate': round(captain_hit_count / gws_counted * 100, 1),
                    'gws_counted': gws_counted,
                }
                print(f"  Captain ROI: {captain_roi['hit_rate']}% hit rate, {captain_roi['points_left_on_table']} pts left on table")
        except Exception as e:
            print(f"  [WARN] Captain ROI calculation failed: {e}")

    # Compute mini-league position context for chip strategy
    ml_position = None
    if competitive_data:
        try:
            user_entry = next((t for t in competitive_data if t.get('team_id') == team_id), None)
            if user_entry:
                all_points = sorted([t.get('total_points', 0) for t in competitive_data], reverse=True)
                user_pts = user_entry.get('total_points', 0)
                leader_pts = all_points[0] if all_points else user_pts
                rank_in_league = all_points.index(user_pts) + 1 if user_pts in all_points else len(all_points)
                gap_to_leader = leader_pts - user_pts
                ml_position = {
                    'rank': rank_in_league,
                    'total_teams': len(competitive_data),
                    'gap_to_leader': gap_to_leader,
                    'strategy': 'chasing' if gap_to_leader > 50 else ('protecting' if gap_to_leader < 20 else 'mid-pack'),
                }
                print(f"  Mini-league position: {ml_position['strategy']} (rank {rank_in_league}/{len(competitive_data)}, {gap_to_leader}pts behind leader)")
        except Exception as e:
            print(f"  [WARN] ML position computation failed: {e}")

    # Inject ml_position into chip_analysis so LaTeX can display league context
    if ml_position and chip_analysis:
        chip_analysis['ml_position'] = ml_position

    # Generate LaTeX report
    phase_timer.start("LaTeX compilation")
    log("Generating LaTeX report...", verbose)
    generator = LaTeXReportGenerator(team_id, gameweek, plot_dir=output_dir, session_cache=session_cache)

    # Add position_in_squad to squad for formation diagram
    for player in squad:
        for analysis in squad_analysis:
            if analysis['player_id'] == player['id']:
                analysis['position_in_squad'] = player['position_in_squad']
                break

    latex_content = generator.compile_report(
        team_info=team_info,
        gw_history=gw_history,
        squad=squad,
        squad_analysis=squad_analysis,
        recommendations=recommendations,
        captain_picks=captain_picks,
        chips_used=chips_used,
        transfers=transfers,
        multi_week_strategy=multi_week_strategy,
        competitive_data=competitive_data,
        wildcard_team=wildcard_team,
        free_hit_team=free_hit_team,
        season_history=season_history,
        top_global_data=top_global_data,
        chip_analysis=chip_analysis,
        intelligence_payload=intelligence_payload,
        intelligence_meta=intelligence_meta,
        captain_roi=captain_roi,
    )

    # Write LaTeX file
    output_filename = args.output or f"report_{team_id}.tex"
    output_path = Path(__file__).parent / "reports" / output_filename

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)

    print(f"\n[SUCCESS] LaTeX report saved to: {output_path}")
    phase_timer.stop("LaTeX compilation")
    
    # Save session cache to disk
    phase_timer.start("Session cache save")
    if session_cache is not None:
        log("Saving session cache...", verbose)
        session_cache.save()
        stats = session_cache.get_stats()
        print(f"[INFO] Session cache saved: {stats['entries_in_memory']} entries, {stats['session_size_mb']:.2f} MB")
    phase_timer.stop("Session cache save")

    # Compile to PDF if requested
    phase_timer.start("PDF compilation")
    if not args.no_pdf:
        log("Compiling PDF with pdflatex...", verbose)
        try:
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', output_filename],
                cwd=output_path.parent,
                capture_output=True,
                text=True
            )

            # Run twice for proper references
            subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', output_filename],
                cwd=output_path.parent,
                capture_output=True,
                text=True
            )

            pdf_path = output_path.with_suffix('.pdf')
            if pdf_path.exists():
                print(f"[SUCCESS] PDF report saved to: {pdf_path}")

                # Clean up auxiliary files
                for ext in ['.aux', '.log', '.out']:
                    aux_file = output_path.with_suffix(ext)
                    if aux_file.exists():
                        aux_file.unlink()
            else:
                print("[WARNING] PDF compilation may have failed. Check the .log file.")
                if verbose:
                    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

        except FileNotFoundError:
            print("[WARNING] pdflatex not found. Install TeX Live or MacTeX to compile PDFs.")
            print(f"  LaTeX source available at: {output_path}")
    phase_timer.stop("PDF compilation")

    phase_timer.print_summary()

    # Print summary
    print(f"\n{'='*60}")
    print("  Report Summary")
    print(f"{'='*60}")
    print(f"  Team: {team_info['team_name']}")
    print(f"  Gameweek: {gameweek}")
    print(f"  Total Points: {team_info['overall_points']}")
    print(f"  Overall Rank: {team_info['overall_rank']:,}")
    print(f"  Underperformers: {len(underperformers)}")
    print(f"  Transfer Suggestions: {sum(len(r.get('in_options', [])) for r in recommendations)}")

    if multi_week_strategy:
        ev_data = multi_week_strategy.get('expected_value', {})
        metrics = multi_week_strategy.get('model_metrics', {})
        print(f"\n  Transfer Strategy (5-GW Horizon):")
        print(f"    Current Squad xP: {ev_data.get('current_squad', 0):.1f}")
        print(f"    Optimized xP: {ev_data.get('optimized_squad', 0):.1f}")
        print(f"    Potential Gain: +{ev_data.get('potential_gain', 0):.1f} pts")
        print(f"    Model MAE: {metrics.get('mae', 'N/A')}")
        print(f"    Model R²: {metrics.get('r2', 'N/A')}")

    if captain_picks:
        print(f"\n  Top Captain Picks:")
        for i, pick in enumerate(captain_picks[:3], 1):
            print(f"    {i}. {pick['name']} ({pick['position']})")

    if competitive_data:
        print(f"\n  Competitive Analysis:")
        print(f"    Teams compared: {len(competitive_data)}")
        for entry in competitive_data:
            ti = entry.get('team_info', {})
            print(f"    - {ti.get('team_name', 'Unknown')}: {ti.get('overall_points', 0)} pts (Rank: {ti.get('overall_rank', 0):,})")

    if wildcard_team:
        budget = wildcard_team.get('budget', {})
        print(f"\n  Wildcard Draft:")
        print(f"    Budget: {budget.get('total', 0):.1f}m | Spent: {budget.get('spent', 0):.1f}m | ITB: {budget.get('remaining', 0):.1f}m")
        print(f"    Formation: {wildcard_team.get('formation', 'N/A')}")
        print(f"    Captain: {wildcard_team.get('captain', {}).get('name', 'N/A')}")
        print(f"    Vice Captain: {wildcard_team.get('vice_captain', {}).get('name', 'N/A')}")

    if free_hit_team:
        budget = free_hit_team.get('budget', {})
        league_analysis = free_hit_team.get('league_analysis', {})
        print(f"\n  Free Hit Draft (GW{free_hit_team.get('target_gw', '?')}):")
        print(f"    Budget: {budget.get('total', 0):.1f}m | Spent: {budget.get('spent', 0):.1f}m | ITB: {budget.get('remaining', 0):.1f}m")
        print(f"    Formation: {free_hit_team.get('formation', 'N/A')}")
        print(f"    Captain: {free_hit_team.get('captain', {}).get('name', 'N/A')}")
        print(f"    Strategy: {free_hit_team.get('strategy', 'N/A')}")
        print(f"    League sample: {league_analysis.get('sample_size', 0)} teams")
        diffs = league_analysis.get('differentials', [])
        if diffs:
            diff_names = ', '.join([d['name'] for d in diffs[:3]])
            print(f"    Top differentials: {diff_names}")

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
