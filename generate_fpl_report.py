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
from pathlib import Path
from datetime import datetime
import copy
from typing import List, Optional

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
    return parser.parse_args()


def log(message: str, verbose: bool = True):
    """Print log message if verbose mode is enabled."""
    if verbose:
        print(f"[INFO] {message}")


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

    print(f"\n{'='*60}")
    print(f"  FPL Report Generator")
    print(f"  Team ID: {team_id} | Season: {season}")
    if use_cache:
        print(f"  Cache: Enabled (Session-based)")
    else:
        print(f"  Cache: Disabled")
    print(f"{'='*60}\n")

    # Initialize temporary fetcher to get current gameweek if not specified
    if preliminary_gameweek is None:
        temp_fetcher = FPLDataFetcher(team_id, season, use_cache=False)
        preliminary_gameweek = temp_fetcher.get_current_gameweek()
        log(f"Auto-detected gameweek: {preliminary_gameweek}", verbose)
    
    # Initialize session cache manager
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
    
    # Initialize components
    log("Initializing data fetcher...", verbose)
    fetcher = FPLDataFetcher(team_id, season, use_cache=use_cache, session_cache=session_cache)
    
    # Get current gameweek if not specified
    gameweek = args.gw or fetcher.get_current_gameweek()
    log(f"Analyzing Gameweek {gameweek}", verbose)
    
    # Fetch FPL Core Insights data (enhanced datasets with match stats + Elo)
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
    
    # Extract current gameweek data for convenience
    fpl_core_gw_data = all_gw_data.get(gameweek, {})
    
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

    # Fetch team info
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

    # Initialize analyzer
    log("Initializing player analyzer...", verbose)
    analyzer = PlayerAnalyzer(fetcher)

    # Analyze each player
    log("Analyzing squad players...", verbose)
    squad_analysis = []
    for player in squad:
        player_id = player['id']
        position = player['position']
        log(f"  Analyzing {player['name']}...", verbose)
        analysis = analyzer.generate_player_summary(player_id, position)
        analysis['position_in_squad'] = player.get('position_in_squad', 0)
        analysis['is_captain'] = player.get('is_captain', False)
        analysis['is_vice_captain'] = player.get('is_vice_captain', False)

        # Fetch GW history for cumulative points chart
        player_history = fetcher.get_player_history(player_id)
        if not player_history.empty:
            analysis['gw_history'] = player_history.to_dict('records')
        else:
            analysis['gw_history'] = []

        squad_analysis.append(analysis)

    # Add FPL API fixtures as fallback for future GW predictions
    # This allows the FPL Core predictor to use fixture data even when
    # Core Insights hasn't updated with future GW fixtures yet
    fpl_core_season_data['fpl_api_fixtures'] = fetcher.fixtures_df
    
    # Initialize recommender with FPL Core data for enhanced predictions
    # Pass previous season data for cross-season training (if available)
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

    # Generate multi-week transfer strategy
    log("Generating multi-week transfer strategy...", verbose)
    strategy_planner = TransferStrategyPlanner(fetcher, analyzer, recommender)
    use_mip = not args.no_mip
    try:
        multi_week_strategy = strategy_planner.generate_strategy(
            squad_analysis, 
            num_weeks=5,
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

    # Generate 5-GW predictions for ALL top players (for Wildcard/Free Hit drafts)
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

    # Generate Plots
    log("Generating visualizations...", verbose)
    output_dir = Path(__file__).parent / "reports" / "plots"
    plot_gen = PlotGenerator(output_dir)
    
    # Generate specific plots
    plot_gen.generate_points_per_gw(gw_history, chips_used)
    plot_gen.generate_contribution_heatmap(season_history)
    # Pass season_history to treemap so it only counts points when players were in starting XI
    plot_gen.generate_treemap(season_history)
    plot_gen.generate_transfer_matrix(transfers)
    
    # Generate advanced finishing & creativity plots from FPL Core Insights data
    log("Generating advanced finishing & creativity analysis...", verbose)
    squad_ids = [p['id'] for p in squad]
    try:
        # Clinical vs Wasteful Goals (league-wide context)
        plot_gen.generate_clinical_wasteful_chart(
            fpl_core_season_data,
            fpl_core_gw_data,
            squad_ids,
            gameweek
        )
        # Clinical vs Wasteful Goals (squad-only)
        plot_gen.generate_clinical_wasteful_chart_squad_only(
            fpl_core_season_data,
            fpl_core_gw_data,
            squad_ids,
            gameweek
        )
        # Clutch vs Frustrated Assists (league-wide context)
        plot_gen.generate_clutch_frustrated_chart(
            fpl_core_season_data,
            fpl_core_gw_data,
            squad_ids,
            gameweek
        )
        # Clutch vs Frustrated Assists (squad-only)
        plot_gen.generate_clutch_frustrated_chart_squad_only(
            fpl_core_season_data,
            fpl_core_gw_data,
            squad_ids,
            gameweek
        )
        # Usage vs Output Scatter (league-wide context)
        plot_gen.generate_usage_output_scatter(
            all_gw_data,
            fpl_core_season_data,
            squad_ids
        )
        # Usage vs Output Scatter (recent form - league-wide, last 5 GWs)
        plot_gen.generate_usage_output_scatter_recent(
            all_gw_data,
            fpl_core_season_data,
            squad_ids,
            last_n_gw=5
        )
        # Usage vs Output Scatter (squad-only)
        plot_gen.generate_usage_output_scatter_squad_only(
            all_gw_data,
            fpl_core_season_data,
            squad_ids,
            gameweek
        )
        # Usage vs Output Scatter (recent form - squad-only, last 5 GWs)
        plot_gen.generate_usage_output_scatter_squad_recent(
            all_gw_data,
            fpl_core_season_data,
            squad_ids,
            gameweek,
            last_n_gw=5
        )
        # Defensive Value Charts (season)
        plot_gen.generate_defensive_value_scatter(
            all_gw_data,
            fpl_core_season_data,
            squad_ids
        )
        # Defensive Value Charts (recent form - last 5 GWs)
        plot_gen.generate_defensive_value_scatter_recent(
            all_gw_data,
            fpl_core_season_data,
            squad_ids,
            last_n_gw=5
        )
        # Goalkeeper Shot-Stopping Charts (season)
        plot_gen.generate_goalkeeper_value_scatter(
            all_gw_data,
            fpl_core_season_data,
            squad_ids
        )
        # Goalkeeper Shot-Stopping Charts (recent form - last 5 GWs)
        plot_gen.generate_goalkeeper_value_scatter_recent(
            all_gw_data,
            fpl_core_season_data,
            squad_ids,
            last_n_gw=5
        )
        log("  Advanced analysis plots generated successfully (league + squad + defensive)", verbose)
    except Exception as e:
        log(f"  Warning: Could not generate advanced analysis plots: {e}", verbose)
    
    # Generate hindsight fixture analysis
    log("Generating hindsight fixture analysis...", verbose)
    teams_data = fetcher.bootstrap_data.get('teams', [])
    plot_gen.generate_hindsight_fixture_analysis(
        season_history, 
        fetcher.fixtures_df, 
        teams_data,
        start_gw=max(1, gameweek - 10),  # Last 10 GWs or less
        end_gw=gameweek
    )

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
            plot_gen.generate_competitive_points_per_gw(competitive_data)
            plot_gen.generate_competitive_points_progression(competitive_data)
            plot_gen.generate_competitive_rank_progression(competitive_data)
            
            # Generate treemaps for each competitor
            log("Generating competitor treemaps...", verbose)
            treemap_files = plot_gen.generate_competitive_treemaps(competitive_data)
            print(f"  Generated {len(treemap_files)} competitor treemaps")
        except Exception as e:
            print(f"[WARNING] Competitive analysis failed: {e}")
            competitive_data = None

    # Top Global Managers Competitive Analysis
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
                plot_gen.generate_competitive_points_per_gw(
                    top_global_data, 
                    filename='global_top_points_per_gw.png'
                )
                plot_gen.generate_competitive_points_progression(
                    top_global_data,
                    filename='global_top_points_progression.png'
                )
                plot_gen.generate_competitive_rank_progression(
                    top_global_data,
                    filename='global_top_rank_progression.png'
                )
                
                # Generate treemaps for global top teams
                log("Generating top global manager treemaps...", verbose)
                global_treemap_files = plot_gen.generate_competitive_treemaps(
                    top_global_data,
                    prefix='global_'
                )
                print(f"  Generated {len(global_treemap_files)} global top treemaps")
            else:
                print("  [WARN] Could not fetch top global teams")
        except Exception as e:
            print(f"[WARNING] Top global analysis failed: {e}")
            top_global_data = None

    # Generate Wildcard draft squad
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

    # Generate Free Hit draft squad (league-aware)
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

    # Generate LaTeX report
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
        top_global_data=top_global_data
    )

    # Write LaTeX file
    output_filename = args.output or f"report_{team_id}.tex"
    output_path = Path(__file__).parent / "reports" / output_filename

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)

    print(f"\n[SUCCESS] LaTeX report saved to: {output_path}")
    
    # Save session cache to disk
    if session_cache is not None:
        log("Saving session cache...", verbose)
        session_cache.save()
        stats = session_cache.get_stats()
        print(f"[INFO] Session cache saved: {stats['entries_in_memory']} entries, {stats['session_size_mb']:.2f} MB")

    # Compile to PDF if requested
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
