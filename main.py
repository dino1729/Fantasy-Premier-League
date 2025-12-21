#!/usr/bin/env python3
"""FPL Bot - Main Orchestrator

Single entry point to run the complete FPL optimization pipeline:
1. ETL Pipeline - Fetch fresh data from FPL API + ClubElo
2. ML Models - Generate xP predictions using trained models
3. MIP Solver - Optimize transfers over planning horizon
4. Report - Generate actionable strategy report

Usage:
    # Full pipeline with fresh data
    python main.py --update-data --team 847569
    
    # Quick run (cached data)
    python main.py --team 847569
    
    # Train ML models (after ETL)
    python main.py --train-models
    
    # Generate report only
    python main.py --team 847569 --report-only
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

# Project root
PROJECT_ROOT = Path(__file__).parent


def load_config() -> dict:
    """Load configuration from config.yml in project root.
    
    Returns:
        Dict with config values or empty dict if not found.
    """
    config_path = PROJECT_ROOT / 'config.yml'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    return {}


# Load config at module level
_CONFIG = load_config()
DEFAULT_TEAM_ID = _CONFIG.get('team_id', 847569)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def run_etl_pipeline(fetch: bool = True, 
                     transform: bool = True,
                     generate_projections: bool = False) -> Dict[str, str]:
    """Run the ETL pipeline to refresh data.
    
    Args:
        fetch: Whether to fetch fresh data from APIs.
        transform: Whether to transform data to parquet.
        generate_projections: Whether to generate heuristic projections.
        
    Returns:
        Dict with paths to generated files.
    """
    logger.info("🚀 Running ETL Pipeline...")
    
    from etl.pipeline import ETLPipeline
    
    pipeline = ETLPipeline()
    result = pipeline.run(
        fetch=fetch,
        transform=transform,
        generate_projections=generate_projections
    )
    
    logger.info("✅ ETL Pipeline complete")
    return result


def train_ml_models(min_minutes: int = 180) -> Dict[str, Any]:
    """Train position-specific ML models.
    
    Args:
        min_minutes: Minimum minutes for player inclusion.
        
    Returns:
        Training summary with metrics.
    """
    logger.info("🧠 Training ML Models...")
    
    from models.train import train_models_from_warehouse
    
    trainer = train_models_from_warehouse(min_minutes=min_minutes)
    
    if trainer:
        logger.info("✅ ML Models trained successfully")
        for pos, metrics in trainer.training_summary.items():
            logger.info(f"   {pos}: MAE={metrics['mae']:.2f}, R²={metrics['r2']:.3f}")
        return trainer.training_summary
    else:
        logger.warning("⚠️ ML training failed - will use heuristics")
        return {}


def generate_predictions(horizon: int = 5, 
                         min_minutes: int = 45) -> Path:
    """Generate xP predictions for all players.
    
    Args:
        horizon: Number of future gameweeks.
        min_minutes: Minimum minutes for player inclusion.
        
    Returns:
        Path to projections_horizon.parquet.
    """
    logger.info("📊 Generating ML Predictions...")
    
    from models.inference import FPLInferencePipeline
    
    pipeline = FPLInferencePipeline(use_fallback=True)
    df = pipeline.run(horizon=horizon, min_minutes=min_minutes)
    
    filepath = PROJECT_ROOT / 'data' / 'parquet' / 'projections_horizon.parquet'
    logger.info(f"✅ Saved {len(df)} predictions to {filepath}")
    
    return filepath


def run_solver(team_id: int,
               gameweek: Optional[int] = None,
               horizon: int = 5,
               free_transfers: int = 1,
               mip_time_limit: float = 60.0) -> Dict[str, Any]:
    """Run the MIP solver to optimize transfers.
    
    Args:
        team_id: FPL team ID.
        gameweek: Target gameweek (None for current).
        horizon: Planning horizon in weeks.
        free_transfers: Available free transfers.
        mip_time_limit: Solver time limit in seconds.
        
    Returns:
        Solver result dict with recommendations.
    """
    logger.info("🧮 Optimizing Squad with MIP Solver...")
    
    from reports.fpl_report.data_fetcher import FPLDataFetcher
    from reports.fpl_report.player_analyzer import PlayerAnalyzer
    from reports.fpl_report.transfer_recommender import TransferRecommender
    from reports.fpl_report.transfer_strategy import TransferStrategyPlanner
    
    # Initialize fetcher
    fetcher = FPLDataFetcher(team_id)
    
    if gameweek is None:
        gameweek = fetcher.get_current_gameweek()
    
    # Initialize analyzer and recommender
    analyzer = PlayerAnalyzer(fetcher)
    recommender = TransferRecommender(fetcher, analyzer)
    
    # Get current squad and analyze
    current_squad = fetcher.get_current_squad()
    squad_analysis = []
    for player in current_squad:
        analysis = analyzer.generate_player_summary(player['id'], player['position'])
        analysis['position_in_squad'] = player.get('position_in_squad', 0)
        analysis['is_captain'] = player.get('is_captain', False)
        analysis['is_vice_captain'] = player.get('is_vice_captain', False)
        squad_analysis.append(analysis)
    
    # Get strategy
    planner = TransferStrategyPlanner(
        fetcher,
        analyzer,
        recommender
    )
    
    strategy = planner.generate_strategy(
        squad_analysis=squad_analysis,
        num_weeks=horizon,
        use_mip=True,
        mip_time_limit=mip_time_limit,
        mip_candidate_pool=50,
        free_transfers=free_transfers
    )
    
    # Extract MIP result
    mip_result = strategy.get('mip_recommendation', {})
    
    if mip_result.get('status') == 'optimal':
        logger.info("✅ MIP Solver found optimal solution")
        logger.info(f"   Expected Points: {mip_result.get('expected_points', 0):.1f}")
        logger.info(f"   Transfers: {mip_result.get('num_transfers', 0)}")
        logger.info(f"   Hit Cost: {mip_result.get('hit_cost', 0)}")
    else:
        logger.warning(f"⚠️ MIP Solver status: {mip_result.get('status', 'unknown')}")
    
    return {
        'strategy': strategy,
        'mip_result': mip_result,
        'current_gw': gameweek,
        'team_id': team_id,
        'fetcher': fetcher
    }


def print_strategy_report(solver_output: Dict[str, Any], 
                          use_colors: bool = True):
    """Generate and print the strategy report.
    
    Args:
        solver_output: Output from run_solver().
        use_colors: Whether to use terminal colors.
    """
    logger.info("📝 Generating Strategy Report...")
    
    from solver.interpreter import SolverInterpreter
    from reports.analytics import ROICalculator
    from reports.strategy_reporter import StrategyReporter
    
    mip_result = solver_output.get('mip_result', {})
    current_gw = solver_output.get('current_gw', 17)
    fetcher = solver_output.get('fetcher')
    
    # Interpret solver output
    interpreter = SolverInterpreter()
    plan = interpreter.interpret_mip_result(mip_result, current_gw, horizon=5)
    
    # Get current squad for ROI analysis
    if fetcher:
        try:
            current_squad = fetcher.get_current_squad()
        except Exception:
            current_squad = []
    else:
        current_squad = []
    
    # Calculate ROI
    roi = None
    if current_squad and mip_result.get('transfers_in'):
        try:
            calculator = ROICalculator()
            roi = calculator.analyze_strategy(
                current_squad,
                mip_result.get('transfers_out', []),
                mip_result.get('transfers_in', []),
                current_gw + 1,
                horizon=5
            )
        except Exception as e:
            logger.warning(f"ROI calculation failed: {e}")
    
    # Print report
    reporter = StrategyReporter(use_colors=use_colors)
    reporter.print_full_report(plan, roi)
    
    # Save plan
    plan.save()
    
    return plan


def main():
    """Main entry point for the FPL bot."""
    parser = argparse.ArgumentParser(
        description='FPL Bot - Optimal Transfer Strategy Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with fresh data
  python main.py --update-data --team 847569
  
  # Quick run with cached data
  python main.py --team 847569
  
  # Train ML models
  python main.py --train-models
  
  # Generate predictions only
  python main.py --predict-only --horizon 8
"""
    )
    
    # Data options
    parser.add_argument('--update-data', action='store_true',
                       help='Fetch fresh data from FPL API and ClubElo')
    parser.add_argument('--train-models', action='store_true',
                       help='Train/retrain ML models')
    parser.add_argument('--predict-only', action='store_true',
                       help='Only generate ML predictions (skip solver)')
    
    # Team options
    parser.add_argument('--team', '-t', type=int, default=DEFAULT_TEAM_ID,
                       help=f'FPL team ID (default: {DEFAULT_TEAM_ID} from config.yml)')
    parser.add_argument('--gameweek', '-g', type=int, default=None,
                       help='Target gameweek (default: current)')
    
    # Solver options
    parser.add_argument('--horizon', type=int, default=5,
                       help='Planning horizon in gameweeks (default: 5)')
    parser.add_argument('--free-transfers', type=int, default=1,
                       help='Available free transfers (default: 1)')
    parser.add_argument('--mip-time-limit', type=float, default=60.0,
                       help='MIP solver time limit in seconds (default: 60)')
    parser.add_argument('--no-mip', action='store_true',
                       help='Skip MIP solver, use heuristics only')
    
    # Report options
    parser.add_argument('--report-only', action='store_true',
                       help='Only generate report from cached data')
    parser.add_argument('--no-colors', action='store_true',
                       help='Disable terminal colors')
    
    # General options
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Print banner
    if not args.quiet:
        print()
        print("╔═══════════════════════════════════════════════════════════╗")
        print("║           FPL BOT - Transfer Strategy Optimizer           ║")
        print("║              Powered by ML + MIP Optimization             ║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print()
        print(f"  Team ID: {args.team}")
        print(f"  Horizon: {args.horizon} gameweeks")
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    try:
        # Step 1: Update data (optional)
        if args.update_data:
            run_etl_pipeline(fetch=True, transform=True, generate_projections=False)
            print()
        
        # Step 2: Train models (optional)
        if args.train_models:
            train_ml_models()
            print()
        
        # Step 3: Generate predictions
        if not args.report_only:
            generate_predictions(horizon=args.horizon)
            print()
        
        # Step 4: Predict only mode
        if args.predict_only:
            logger.info("✅ Predictions generated (--predict-only mode)")
            return 0
        
        # Step 5: Run solver
        if not args.report_only:
            solver_output = run_solver(
                team_id=args.team,
                gameweek=args.gameweek,
                horizon=args.horizon,
                free_transfers=args.free_transfers,
                mip_time_limit=args.mip_time_limit
            )
            print()
        else:
            # Load cached strategy
            logger.info("📂 Loading cached strategy...")
            solver_output = {'mip_result': {}, 'current_gw': args.gameweek or 17}
        
        # Step 6: Generate report
        plan = print_strategy_report(
            solver_output,
            use_colors=not args.no_colors
        )
        
        if not args.quiet:
            print()
            logger.info("🎯 Strategy plan saved to reports/strategy_plan_gw*.json")
            print()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

