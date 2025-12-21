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
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fpl_report.data_fetcher import (
    FPLDataFetcher, 
    build_competitive_dataset,
    get_league_entry_ids,
    compute_league_ownership,
    get_top_global_teams
)
from fpl_report.player_analyzer import PlayerAnalyzer
from fpl_report.transfer_recommender import TransferRecommender
from fpl_report.latex_generator import LaTeXReportGenerator
from fpl_report.plot_generator import PlotGenerator
from fpl_report.transfer_strategy import TransferStrategyPlanner, WildcardOptimizer, FreeHitOptimizer
from fpl_report.cache_manager import CacheManager


def load_config() -> dict:
    """Load configuration from config.yml in reports directory.
    
    Returns:
        Dict with config values or empty dict if not found.
    """
    config_path = Path(__file__).parent / 'config.yml'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    return {}


# Load config at module level
_CONFIG = load_config()

# --- Team Settings ---
DEFAULT_TEAM_ID = _CONFIG.get('team_id', 847569)
DEFAULT_COMPETITORS = _CONFIG.get('competitors', [21023, 1827604, 489166])

# --- Gameweek & Season ---
DEFAULT_GAMEWEEK = _CONFIG.get('gameweek')  # None = auto-detect
DEFAULT_SEASON = _CONFIG.get('season', '2025-26')

# --- Free Hit Settings ---
_FREE_HIT = _CONFIG.get('free_hit', {})
FREE_HIT_TARGET_GW = _FREE_HIT.get('target_gw')
FREE_HIT_STRATEGY = _FREE_HIT.get('strategy', 'balanced')

# --- Wildcard Settings ---
_WILDCARD = _CONFIG.get('wildcard', {})
WILDCARD_STRATEGY = _WILDCARD.get('strategy', 'balanced')

# --- Transfer Planner ---
_TRANSFER = _CONFIG.get('transfer_planner', {})
TRANSFER_HORIZON = _TRANSFER.get('horizon', 5)
FREE_TRANSFERS_OVERRIDE = _TRANSFER.get('free_transfers')

# --- MIP Solver ---
_MIP = _CONFIG.get('mip_solver', {})
MIP_ENABLED = _MIP.get('enabled', True)
MIP_TIME_LIMIT = _MIP.get('time_limit', 60)
MIP_CANDIDATE_POOL = _MIP.get('candidate_pool', 30)

# --- League Settings ---
_LEAGUE = _CONFIG.get('league', {})
LEAGUE_ID = _LEAGUE.get('league_id')
LEAGUE_SAMPLE_SIZE = _LEAGUE.get('sample_size', 20)

# --- Output Options ---
_OUTPUT = _CONFIG.get('output', {})
NO_COMPETITIVE = _OUTPUT.get('no_competitive', False)
VERBOSE = _OUTPUT.get('verbose', False)


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


def main():
    """Main entry point."""
    args = parse_args()

    # Handle cache commands
    cache_mgr = CacheManager(enabled=not args.no_cache)
    
    if args.cache_stats:
        stats = cache_mgr.get_stats()
        print(f"\n{'='*60}")
        print(f"  Cache Statistics")
        print(f"{'='*60}")
        print(f"  Enabled: {stats.get('enabled', False)}")
        if stats.get('enabled'):
            print(f"  Cache Directory: {stats.get('cache_dir', 'N/A')}")
            print(f"  Total Entries: {stats.get('total_entries', 0)}")
            print(f"\n  Entries by Type:")
            for cache_type, count in stats.get('by_type', {}).items():
                print(f"    - {cache_type}: {count}")
        print(f"{'='*60}\n")
        return
    
    if args.clear_cache:
        print("[INFO] Clearing cache...")
        cache_mgr.invalidate()
    
    # Validate team_id is provided for report generation
    if not args.team:
        print("[ERROR] --team/-t argument is required for report generation")
        print("Use --cache-stats to view cache statistics without generating a report")
        sys.exit(1)
    
    team_id = args.team
    season = args.season
    verbose = args.verbose
    use_cache = not args.no_cache

    print(f"\n{'='*60}")
    print(f"  FPL Report Generator")
    print(f"  Team ID: {team_id} | Season: {season}")
    if use_cache:
        print(f"  Cache: Enabled")
    else:
        print(f"  Cache: Disabled")
    print(f"{'='*60}\n")

    # Initialize components
    log("Initializing data fetcher...", verbose)
    fetcher = FPLDataFetcher(team_id, season, use_cache=use_cache)

    # Get current gameweek if not specified
    gameweek = args.gw or fetcher.get_current_gameweek()
    log(f"Analyzing Gameweek {gameweek}", verbose)

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

    # Initialize recommender
    log("Generating transfer recommendations...", verbose)
    recommender = TransferRecommender(fetcher, analyzer)

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
            all_player_predictions = predictor.predict_multiple_gws(top_players, num_gws=5)
            print(f"  Generated predictions for {len(all_player_predictions)} players")
        else:
            print("  Predictor not trained, skipping draft predictions")
    except Exception as e:
        print(f"  [WARN] Draft predictions failed: {e}")
        all_player_predictions = {}

    # Generate Plots
    log("Generating visualizations...", verbose)
    output_dir = Path(__file__).parent / "plots"
    plot_gen = PlotGenerator(output_dir)
    
    # Generate specific plots
    plot_gen.generate_points_per_gw(gw_history, chips_used)
    plot_gen.generate_contribution_heatmap(season_history)
    # Pass season_history to treemap so it only counts points when players were in starting XI
    plot_gen.generate_treemap(season_history)
    plot_gen.generate_transfer_matrix(transfers)
    
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
                use_cache=use_cache
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
        log("Building competitive analysis vs Top 5 Global Managers...", verbose)
        try:
            # Fetch top 5 global team entry IDs
            top_teams = get_top_global_teams(n=5, use_cache=use_cache)
            if top_teams:
                top_entry_ids = [t['entry_id'] for t in top_teams]
                # Add user's team for comparison
                if team_id not in top_entry_ids:
                    global_comparison_ids = [team_id] + top_entry_ids
                else:
                    global_comparison_ids = top_entry_ids
                
                print(f"  Comparing user vs top {len(top_entry_ids)} global managers")
                
                # Build competitive dataset for global comparison
                top_global_data = build_competitive_dataset(
                    entry_ids=global_comparison_ids,
                    season=season,
                    gameweek=gameweek,
                    use_cache=use_cache
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
                    use_cache=use_cache
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
                    use_cache=use_cache
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
        print(f"  Strategy: {args.free_hit_strategy}")
        
        # Get teams and fixtures data for FDR display
        teams_data = fetcher.bootstrap_data.get('teams', [])
        fixtures_list = fetcher.fixtures_df.to_dict('records') if not fetcher.fixtures_df.empty else []
        
        # Build optimal Free Hit squad (pass predictions for 5-GW xP)
        fh_optimizer = FreeHitOptimizer(
            players_df=fetcher.players_df,
            total_budget=total_budget,
            league_ownership=league_ownership_data,
            strategy=args.free_hit_strategy,
            teams_data=teams_data,
            fixtures_data=fixtures_list,
            current_gw=gameweek,
            predictions=predictions
        )
        free_hit_team = fh_optimizer.build_squad()
        
        # Add target GW to the result
        free_hit_team['target_gw'] = free_hit_gw
        
        # Calculate current squad 1-GW xP (sum of ep_next for current squad)
        current_squad_1gw_xp = 0.0
        for p in squad_analysis:
            pid = p.get('player_id')
            player_row = fetcher.players_df[fetcher.players_df['id'] == pid]
            if not player_row.empty:
                ep_next = float(player_row['ep_next'].iloc[0] or 0)
                current_squad_1gw_xp += ep_next
        
        # Calculate optimized (Free Hit) 1-GW xP from starting XI
        fh_1gw_xp = sum(p.get('ep_next', 0) for p in free_hit_team.get('starting_xi', []))
        
        # Calculate per-GW xP comparison for 5 gameweeks
        gw_comparison = {
            'current_gw': gameweek,
            'gameweeks': [],
            'current_squad_xp': [],
            'free_hit_xp': [],
            'best_gw': None
        }
        
        # Get current squad player IDs
        current_squad_ids = [p.get('player_id') for p in squad_analysis]
        # Get Free Hit XI player IDs
        fh_xi_ids = [p.get('id') for p in free_hit_team.get('starting_xi', [])]
        
        # Use predictor uniformly for ALL gameweeks in the comparison plot
        # This ensures a fair apples-to-apples comparison across GW17-21
        # Note: The EV Analysis box above uses FPL's ep_next for 1-GW focus,
        # but this plot uses our predictor for consistent multi-GW comparison
        predictor = strategy_planner.predictor if strategy_planner else None
        if predictor and predictor.is_trained and all_player_predictions:
            best_gain = 0
            for gw_offset in range(1, 6):  # Next 5 GWs
                target_gw = gameweek + gw_offset
                gw_comparison['gameweeks'].append(target_gw)
                
                # Use predictor for ALL gameweeks (uniform methodology)
                current_gw_xp = 0.0
                for pid in current_squad_ids:
                    if pid in all_player_predictions:
                        preds = all_player_predictions[pid].get('predictions', [])
                        if gw_offset <= len(preds):
                            current_gw_xp += preds[gw_offset - 1]
                
                fh_gw_xp = 0.0
                for pid in fh_xi_ids:
                    if pid in all_player_predictions:
                        preds = all_player_predictions[pid].get('predictions', [])
                        if gw_offset <= len(preds):
                            fh_gw_xp += preds[gw_offset - 1]
                
                gw_comparison['current_squad_xp'].append(round(current_gw_xp, 1))
                gw_comparison['free_hit_xp'].append(round(fh_gw_xp, 1))
                
                # Track best GW
                gain = fh_gw_xp - current_gw_xp
                if gain > best_gain:
                    best_gain = gain
                    gw_comparison['best_gw'] = target_gw
            
            print(f"  Per-GW comparison generated for GW{gameweek+1}-GW{gameweek+5}")
            
            # Generate the Free Hit GW analysis plot
            fh_plot_path = plot_gen.generate_free_hit_gw_comparison(gw_comparison)
            if fh_plot_path:
                free_hit_team['gw_plot'] = fh_plot_path
                print(f"  Free Hit analysis plot saved")
        
        # Add EV analysis to free_hit_team
        free_hit_team['ev_analysis'] = {
            'current_squad_xp': round(current_squad_1gw_xp, 1),
            'optimized_xp': round(fh_1gw_xp, 1),
            'potential_gain': round(fh_1gw_xp - current_squad_1gw_xp, 1),
            'horizon': '1 GW'
        }
        free_hit_team['gw_comparison'] = gw_comparison
        
        budget_info = free_hit_team.get('budget', {})
        league_analysis = free_hit_team.get('league_analysis', {})
        print(f"  Free Hit squad built: {budget_info.get('spent', 0):.1f}m spent, {budget_info.get('remaining', 0):.1f}m ITB")
        print(f"  Formation: {free_hit_team.get('formation', 'N/A')}")
        print(f"  Captain: {free_hit_team.get('captain', {}).get('name', 'N/A')}")
        print(f"  1-GW xP: {current_squad_1gw_xp:.1f} -> {fh_1gw_xp:.1f} (+{fh_1gw_xp - current_squad_1gw_xp:.1f})")
        print(f"  Differentials: {len(league_analysis.get('differentials', []))}")
        print(f"  Template picks: {len(league_analysis.get('template_picks', []))}")
        
    except Exception as e:
        print(f"[WARNING] Free Hit squad generation failed: {e}")
        free_hit_team = None

    # Generate LaTeX report
    log("Generating LaTeX report...", verbose)
    generator = LaTeXReportGenerator(team_id, gameweek, plot_dir=output_dir)

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
    output_path = Path(__file__).parent / output_filename

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)

    print(f"\n[SUCCESS] LaTeX report saved to: {output_path}")

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
