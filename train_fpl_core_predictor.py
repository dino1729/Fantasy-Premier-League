#!/usr/bin/env python3
"""FPL Core Predictor Training Script

Trains the FPLCorePredictor model using cross-season data and saves
the trained models to disk for later use.

Usage:
    python train_fpl_core_predictor.py                    # Train with defaults
    python train_fpl_core_predictor.py --clear            # Clear old models first
    python train_fpl_core_predictor.py --gw 17            # Specify current gameweek
    python train_fpl_core_predictor.py --tune             # Enable hyperparameter tuning
"""

import argparse
import json
import pickle
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / 'models' / 'artifacts' / 'fpl_core'
FPL_CORE_DATA_DIR = PROJECT_ROOT / 'data' / 'fpl_core'
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_local_2024_25_season_data() -> Dict[str, Optional[pd.DataFrame]]:
    """Load 2024-2025 season-level data from local files.
    
    The local data structure for 2024-2025 is:
    - data/fpl_core/2024-2025/players.csv
    - data/fpl_core/2024-2025/playerstats.csv
    - data/fpl_core/2024-2025/teams.csv
    - data/fpl_core/2024-2025/matches.csv (season aggregate)
    """
    season_dir = FPL_CORE_DATA_DIR / '2024-2025'
    
    results = {}
    
    # Load players
    players_path = season_dir / 'players.csv'
    if players_path.exists():
        results['players'] = pd.read_csv(players_path)
        print(f"  Loaded players: {len(results['players'])} rows")
    else:
        # Fallback to old structure
        old_path = season_dir / 'players' / 'players.csv'
        if old_path.exists():
             results['players'] = pd.read_csv(old_path)
             print(f"  Loaded players (legacy path): {len(results['players'])} rows")
        else:
            results['players'] = None
            print(f"  Warning: {players_path} not found")
    
    # Load playerstats
    playerstats_path = season_dir / 'playerstats.csv'
    if playerstats_path.exists():
        results['playerstats'] = pd.read_csv(playerstats_path)
        print(f"  Loaded playerstats: {len(results['playerstats'])} rows")
    else:
        # Fallback
        old_path = season_dir / 'playerstats' / 'playerstats.csv'
        if old_path.exists():
             results['playerstats'] = pd.read_csv(old_path)
             print(f"  Loaded playerstats (legacy path): {len(results['playerstats'])} rows")
        else:
            results['playerstats'] = None
            print(f"  Warning: {playerstats_path} not found")
    
    # Load teams
    teams_path = season_dir / 'teams.csv'
    if teams_path.exists():
        results['teams'] = pd.read_csv(teams_path)
        print(f"  Loaded teams: {len(results['teams'])} rows")
    else:
        # Fallback
        old_path = season_dir / 'teams' / 'teams.csv'
        if old_path.exists():
             results['teams'] = pd.read_csv(old_path)
             print(f"  Loaded teams (legacy path): {len(results['teams'])} rows")
        else:
            results['teams'] = None
            print(f"  Warning: {teams_path} not found")
    
    # Load season-level matches (aggregated)
    matches_path = season_dir / 'matches.csv'
    if matches_path.exists():
        results['matches'] = pd.read_csv(matches_path)
        print(f"  Loaded matches: {len(results['matches'])} rows")
    else:
        # Fallback
        old_path = season_dir / 'matches' / 'matches.csv'
        if old_path.exists():
             results['matches'] = pd.read_csv(old_path)
             print(f"  Loaded matches (legacy path): {len(results['matches'])} rows")
        else:
            results['matches'] = None
            print(f"  Warning: {matches_path} not found")
    
    return results


def load_local_2024_25_gameweek_data(up_to_gw: int = 38) -> Dict[int, Dict[str, Optional[pd.DataFrame]]]:
    """Load 2024-2025 gameweek-level data from local files.
    
    Local structure:
    - data/fpl_core/2024-2025/gw{n}/playermatchstats.csv
    - data/fpl_core/2024-2025/gw{n}/matches.csv
    - Season-level data is loaded once and attached to each GW
    
    Also generates synthetic player_gameweek_stats from playermatchstats
    for compatibility with the predictor's minutes-based filtering.
    """
    season_dir = FPL_CORE_DATA_DIR / '2024-2025'
    
    # Load season-level data once
    players_df = None
    playerstats_df = None
    teams_df = None
    
    # Try new path then old path
    for path_variant in [season_dir / 'players.csv', season_dir / 'players' / 'players.csv']:
        if path_variant.exists():
            players_df = pd.read_csv(path_variant)
            break
            
    for path_variant in [season_dir / 'playerstats.csv', season_dir / 'playerstats' / 'playerstats.csv']:
        if path_variant.exists():
            playerstats_df = pd.read_csv(path_variant)
            break
            
    for path_variant in [season_dir / 'teams.csv', season_dir / 'teams' / 'teams.csv']:
        if path_variant.exists():
            teams_df = pd.read_csv(path_variant)
            break
    
    results = {}
    
    for gw in range(1, up_to_gw + 1):
        gw_data = {}
        
        # Load playermatchstats for this GW
        # Try new structure: gw{n}/playermatchstats.csv
        pms_path = season_dir / f'gw{gw}' / 'playermatchstats.csv'
        pms_df = None
        
        if pms_path.exists():
            pms_df = pd.read_csv(pms_path)
        else:
            # Fallback to old structure: playermatchstats/GW{n}/playermatchstats.csv
            old_pms_path = season_dir / 'playermatchstats' / f'GW{gw}' / 'playermatchstats.csv'
            if old_pms_path.exists():
                pms_df = pd.read_csv(old_pms_path)
        
        if pms_df is not None:
            gw_data['playermatchstats'] = pms_df
        else:
            gw_data['playermatchstats'] = None
        
        # Load matches for this GW
        # Try new structure: gw{n}/matches.csv
        matches_path = season_dir / f'gw{gw}' / 'matches.csv'
        matches_df = None
        
        if matches_path.exists():
            matches_df = pd.read_csv(matches_path)
        else:
            # Fallback to old structure: matches/GW{n}/matches.csv
            old_matches_path = season_dir / 'matches' / f'GW{gw}' / 'matches.csv'
            if old_matches_path.exists():
                matches_df = pd.read_csv(old_matches_path)
        
        if matches_df is not None:
            gw_data['matches'] = matches_df
        else:
            gw_data['matches'] = None
        
        # Create synthetic player_gameweek_stats from playermatchstats
        # This allows the predictor to calculate minutes per player
        if pms_df is not None and not pms_df.empty:
            # Aggregate per-match stats to per-player-per-GW
            pgs = pms_df.groupby('player_id').agg({
                'minutes_played': 'sum',
                'goals': 'sum',
                'assists': 'sum',
                'xg': 'sum',
                'xa': 'sum',
            }).reset_index()
            pgs = pgs.rename(columns={'player_id': 'id', 'minutes_played': 'minutes'})
            
            # Add total_points placeholder (needed by predictor)
            # We'll estimate from goals/assists/minutes
            pgs['total_points'] = (
                pgs['goals'] * 4 +  # Rough estimate
                pgs['assists'] * 3 +
                (pgs['minutes'] >= 60).astype(int) * 2 +
                ((pgs['minutes'] > 0) & (pgs['minutes'] < 60)).astype(int) * 1
            )
            pgs['gw'] = gw
            
            # Add BPS placeholder
            pgs['bps'] = 0
            
            gw_data['player_gameweek_stats'] = pgs
        else:
            gw_data['player_gameweek_stats'] = None
        
        # Attach season-level data to each GW (predictor needs this)
        gw_data['players'] = players_df
        gw_data['playerstats'] = playerstats_df
        gw_data['teams'] = teams_df
        
        # Fixtures don't exist in 2024-25 local structure
        gw_data['fixtures'] = None
        
        results[gw] = gw_data
    
    # Count loaded GWs
    loaded_gws = sum(1 for gw, data in results.items() 
                    if data.get('playermatchstats') is not None)
    print(f"  Loaded {loaded_gws} gameweeks with playermatchstats")
    
    return results


def clear_models():
    """Clear existing FPL Core model files."""
    print("Clearing existing FPL Core model files...")
    
    model_files = list(MODELS_DIR.glob('*.pkl'))
    json_files = list(MODELS_DIR.glob('*.json'))
    
    cleared = 0
    for f in model_files + json_files:
        try:
            f.unlink()
            print(f"  Deleted: {f.name}")
            cleared += 1
        except Exception as e:
            print(f"  Failed to delete {f.name}: {e}")
    
    print(f"Cleared {cleared} files.\n")


def save_predictor(predictor, suffix: str = ''):
    """Save trained FPLCorePredictor to disk.
    
    Saves each position model stack and the fallback model separately.
    
    Args:
        predictor: Trained FPLCorePredictor instance.
        suffix: Optional suffix for the model files (e.g., timestamp).
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = suffix or timestamp
    
    print(f"Saving models with suffix: {suffix}")
    
    # Save position-specific models
    for pos in predictor.POSITIONS:
        model_data = predictor.position_models.get(pos)
        if model_data and model_data.get('is_trained', False):
            filepath = MODELS_DIR / f'model_{pos}_{suffix}.pkl'
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"  Saved: {filepath.name}")
    
    # Save fallback model
    if predictor.fallback_models.get('is_trained', False):
        filepath = MODELS_DIR / f'model_fallback_{suffix}.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(predictor.fallback_models, f)
        print(f"  Saved: {filepath.name}")
    
    # Save metadata and metrics
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'suffix': suffix,
        'is_trained': predictor.is_trained,
        'season_start_year': predictor._season_start_year,
        'feature_cols': predictor.feature_cols,
        'metrics': predictor.metrics,
        'positions': predictor.POSITIONS,
    }
    
    metadata_path = MODELS_DIR / f'training_metadata_{suffix}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Saved: {metadata_path.name}")
    
    # Create/update 'latest' symlink-style reference
    latest_path = MODELS_DIR / 'latest.json'
    with open(latest_path, 'w') as f:
        json.dump({'suffix': suffix, 'trained_at': datetime.now().isoformat()}, f, indent=2)
    print(f"  Updated: latest.json -> {suffix}")
    
    print(f"\nModels saved to: {MODELS_DIR}")


def load_predictor(suffix: str = None):
    """Load a trained FPLCorePredictor from disk.
    
    Args:
        suffix: Model suffix to load. If None, loads the latest.
        
    Returns:
        Loaded FPLCorePredictor instance, or None if not found.
    """
    from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
    
    # Find suffix to load
    if suffix is None:
        latest_path = MODELS_DIR / 'latest.json'
        if latest_path.exists():
            with open(latest_path, 'r') as f:
                latest = json.load(f)
                suffix = latest.get('suffix')
        else:
            print("No latest.json found, cannot determine which models to load.")
            return None
    
    print(f"Loading models with suffix: {suffix}")
    
    predictor = FPLCorePredictor()
    
    # Load position-specific models
    for pos in predictor.POSITIONS:
        filepath = MODELS_DIR / f'model_{pos}_{suffix}.pkl'
        if filepath.exists():
            with open(filepath, 'rb') as f:
                predictor.position_models[pos] = pickle.load(f)
            print(f"  Loaded: {filepath.name}")
    
    # Load fallback model
    fallback_path = MODELS_DIR / f'model_fallback_{suffix}.pkl'
    if fallback_path.exists():
        with open(fallback_path, 'rb') as f:
            predictor.fallback_models = pickle.load(f)
        print(f"  Loaded: {fallback_path.name}")
    
    # Load metadata
    metadata_path = MODELS_DIR / f'training_metadata_{suffix}.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            predictor._season_start_year = metadata.get('season_start_year', 0)
            predictor.metrics = metadata.get('metrics', {})
            predictor.is_trained = metadata.get('is_trained', False)
        print(f"  Loaded: {metadata_path.name}")
    
    return predictor


def train_and_save(
    gameweek: int = None,
    tune: bool = False,
    tune_n_iter: int = 20,
    tune_time_budget: float = 60.0,
    cross_season: bool = True,
):
    """Train the FPLCorePredictor and save to disk.
    
    Args:
        gameweek: Current gameweek (auto-detected if None).
        tune: Whether to perform hyperparameter tuning.
        tune_n_iter: Number of hyperparameter combinations to try.
        tune_time_budget: Max time for tuning per position.
        cross_season: Whether to use cross-season training (recommended).
    """
    from etl.fetchers import FPLCoreInsightsFetcher, FPLFetcher
    from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
    
    print("=" * 60)
    print("  FPL Core Predictor Training")
    print("=" * 60)
    
    # Auto-detect current gameweek if not specified
    if gameweek is None:
        print("\nAuto-detecting current gameweek...")
        try:
            fpl_fetcher = FPLFetcher()
            gameweek = fpl_fetcher.get_current_gameweek()
            print(f"  Detected gameweek: {gameweek}")
        except Exception as e:
            print(f"  Failed to detect gameweek: {e}")
            print("  Please specify --gw argument")
            return None
    
    # Determine seasons
    current_season = "2025-2026"
    current_season_start_year = 2025
    prev_season = "2024-2025"
    prev_season_start_year = 2024
    
    print(f"\nCurrent season: {current_season}")
    print(f"Previous season: {prev_season}")
    print(f"Current gameweek: {gameweek}")
    print(f"Cross-season training: {'Enabled' if cross_season else 'Disabled'}")
    print(f"Hyperparameter tuning: {'Enabled' if tune else 'Disabled'}")
    
    # Fetch current season data
    print(f"\nFetching current season data ({current_season})...")
    try:
        current_fetcher = FPLCoreInsightsFetcher(season=current_season)
        current_season_data = current_fetcher.fetch_all()
        current_gw_data = current_fetcher.fetch_all_gameweeks(up_to_gw=gameweek)
        
        gw_count = len([gw for gw, data in current_gw_data.items() 
                       if data.get('playermatchstats') is not None])
        print(f"  Fetched {gw_count} gameweeks with match stats")
    except Exception as e:
        print(f"  Error fetching current season: {e}")
        return None
    
    # Fetch previous season data (for cross-season training)
    prev_season_data = None
    prev_gw_data = None
    
    if cross_season:
        print(f"\nLoading previous season data ({prev_season}) from local files...")
        try:
            # Load from local files (already downloaded)
            prev_season_data = load_local_2024_25_season_data()
            prev_gw_data = load_local_2024_25_gameweek_data(up_to_gw=38)
            
            if prev_season_data.get('players') is None:
                raise ValueError("Previous season players data not found")
        except Exception as e:
            print(f"  Warning: Could not load previous season: {e}")
            print("  Falling back to single-season training")
            cross_season = False
    
    # Initialize predictor
    print("\nInitializing FPLCorePredictor...")
    predictor = FPLCorePredictor()
    
    # Train
    print("\nTraining model...")
    print("-" * 40)
    
    if cross_season and prev_gw_data and prev_season_data:
        predictor.train_cross_season(
            prev_all_gw_data=prev_gw_data,
            prev_fpl_core_season_data=prev_season_data,
            prev_season_start_year=prev_season_start_year,
            current_all_gw_data=current_gw_data,
            current_fpl_core_season_data=current_season_data,
            current_season_start_year=current_season_start_year,
            current_gw=gameweek,
            tune=tune,
            tune_n_iter=tune_n_iter,
            tune_time_budget=tune_time_budget,
        )
    else:
        predictor.train(
            current_gw_data,
            current_season_data,
            gameweek,
            tune=tune,
            tune_n_iter=tune_n_iter,
            tune_time_budget=tune_time_budget,
        )
    
    print("-" * 40)
    
    # Print metrics
    print("\nTraining Results:")
    metrics = predictor.get_model_metrics()
    print(f"  MAE: {metrics.get('mae', 'N/A')}")
    print(f"  RMSE: {metrics.get('rmse', 'N/A')}")
    print(f"  R²: {metrics.get('r2', 'N/A')}")
    print(f"  Training samples: {metrics.get('training_samples', 0)}")
    print(f"  Validation samples: {metrics.get('validation_samples', 0)}")
    
    if 'cross_season' in metrics:
        cs = metrics['cross_season']
        print(f"  Previous season samples: {cs.get('prev_season_samples', 0)}")
        print(f"  Current season samples: {cs.get('current_season_samples', 0)}")
    
    # Save models
    print("\n" + "-" * 40)
    save_predictor(predictor)
    
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    
    return predictor


def main():
    parser = argparse.ArgumentParser(
        description='Train FPL Core Predictor and save models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--gw', type=int, default=None,
        help='Current gameweek (auto-detected if not specified)'
    )
    parser.add_argument(
        '--clear', action='store_true',
        help='Clear existing model files before training'
    )
    parser.add_argument(
        '--tune', action='store_true',
        help='Enable hyperparameter tuning (slower but potentially better)'
    )
    parser.add_argument(
        '--tune-n-iter', type=int, default=20,
        help='Number of hyperparameter combinations to try (default: 20)'
    )
    parser.add_argument(
        '--tune-time-budget', type=float, default=60.0,
        help='Max time for tuning per position in seconds (default: 60)'
    )
    # Load default from config
    from utils.config import DATA_USE_CROSS_SEASON
    
    parser.add_argument(
        '--load', type=str, default=None,
        help='Load and display info for a saved model (suffix or "latest")'
    )
    
    args = parser.parse_args()
    
    # Logic: 
    # 1. Start with config value
    cross_season = DATA_USE_CROSS_SEASON
    
    # If just loading, show info and exit
    if args.load:
        suffix = None if args.load == 'latest' else args.load
        predictor = load_predictor(suffix)
        if predictor:
            print("\nLoaded model metrics:")
            metrics = predictor.get_model_metrics()
            for k, v in metrics.items():
                if k != 'by_position':
                    print(f"  {k}: {v}")
        return
    
    # Clear if requested
    if args.clear:
        clear_models()
    
    # Train and save
    train_and_save(
        gameweek=args.gw,
        tune=args.tune,
        tune_n_iter=args.tune_n_iter,
        tune_time_budget=args.tune_time_budget,
        cross_season=cross_season,
    )


if __name__ == '__main__':
    main()

