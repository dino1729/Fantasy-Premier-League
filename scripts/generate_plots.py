#!/usr/bin/env python3
"""
Standalone script for quickly generating FPL analysis plots.

Usage:
    python scripts/generate_plots.py --team 847569 --gw 17
    python scripts/generate_plots.py --team 847569 --gw 17 --squad-only
    python scripts/generate_plots.py --team 847569 --gw 17 --type clinical
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from reports.fpl_report.plot_generator import PlotGenerator
from reports.fpl_report.data_fetcher import FPLDataFetcher
from etl.fetchers import FPLCoreInsightsFetcher


def main():
    parser = argparse.ArgumentParser(description='Generate FPL analysis plots quickly')
    parser.add_argument('--team', type=int, required=True, help='FPL Team ID')
    parser.add_argument('--gw', type=int, required=True, help='Gameweek number')
    parser.add_argument('--season', default='2025-2026', help='Season (default: 2025-2026)')
    parser.add_argument('--type', choices=['clinical', 'clutch', 'usage', 'all'], 
                       default='all', help='Plot type to generate (default: all)')
    parser.add_argument('--squad-only', action='store_true', 
                       help='Generate squad-only plots (no league context)')
    parser.add_argument('--output-dir', default='reports/plots', 
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Configuration
    team_id = args.team
    gameweek = args.gw
    season = args.season
    output_dir = Path(args.output_dir)
    
    print(f'FPL Plot Generator')
    print(f'='*60)
    print(f'Team ID: {team_id}')
    print(f'Gameweek: {gameweek}')
    print(f'Season: {season}')
    print(f'Squad-only: {args.squad_only}')
    print(f'Plot type: {args.type}')
    print(f'='*60)
    
    # Initialize components
    plot_gen = PlotGenerator(output_dir)
    
    # Fetch FPL data
    print('\nFetching FPL data...')
    fetcher = FPLDataFetcher(team_id=team_id)
    squad = fetcher.get_current_squad(gameweek)
    squad_ids = [p['id'] for p in squad]
    print(f'  Squad: {len(squad_ids)} players')
    
    # Fetch FPL Core data
    print('Fetching FPL Core Insights data...')
    fpl_core = FPLCoreInsightsFetcher(season=season)
    fpl_core_season_data = fpl_core.fetch_all()
    all_gw_data = fpl_core.fetch_all_gameweeks(up_to_gw=gameweek)
    fpl_core_gw_data = all_gw_data.get(gameweek, {})
    print('  Data fetched successfully')
    
    # Generate plots based on type
    print(f'\nGenerating plots...')
    
    if args.type in ['clinical', 'all']:
        print('\n  Clinical vs Wasteful (Goals)...')
        if args.squad_only:
            files = plot_gen.generate_clinical_wasteful_chart_squad_only(
                fpl_core_season_data, fpl_core_gw_data, squad_ids, gameweek
            )
            print(f'    ✓ Squad-only: {files}')
        else:
            files = plot_gen.generate_clinical_wasteful_chart(
                fpl_core_season_data, fpl_core_gw_data, squad_ids, gameweek
            )
            print(f'    ✓ League context: {files}')
            
            # Also generate squad-only for comparison
            squad_files = plot_gen.generate_clinical_wasteful_chart_squad_only(
                fpl_core_season_data, fpl_core_gw_data, squad_ids, gameweek
            )
            print(f'    ✓ Squad-only: {squad_files}')
    
    if args.type in ['clutch', 'all']:
        print('\n  Clutch vs Frustrated (Assists)...')
        if args.squad_only:
            files = plot_gen.generate_clutch_frustrated_chart_squad_only(
                fpl_core_season_data, fpl_core_gw_data, squad_ids, gameweek
            )
            print(f'    ✓ Squad-only: {files}')
        else:
            files = plot_gen.generate_clutch_frustrated_chart(
                fpl_core_season_data, fpl_core_gw_data, squad_ids, gameweek
            )
            print(f'    ✓ League context: {files}')
            
            # Also generate squad-only for comparison
            squad_files = plot_gen.generate_clutch_frustrated_chart_squad_only(
                fpl_core_season_data, fpl_core_gw_data, squad_ids, gameweek
            )
            print(f'    ✓ Squad-only: {squad_files}')
    
    if args.type in ['usage', 'all']:
        print('\n  Usage vs Output scatter...')
        if args.squad_only:
            file = plot_gen.generate_usage_output_scatter_squad_only(
                all_gw_data, fpl_core_season_data, squad_ids, gameweek
            )
            print(f'    ✓ Squad-only: {file}')
            recent_file = plot_gen.generate_usage_output_scatter_squad_recent(
                all_gw_data, fpl_core_season_data, squad_ids, gameweek, last_n_gw=5
            )
            print(f'    ✓ Squad-only (last 5 GWs): {recent_file}')
        else:
            file = plot_gen.generate_usage_output_scatter(
                all_gw_data, fpl_core_season_data, squad_ids
            )
            print(f'    ✓ League context: {file}')
            recent_file = plot_gen.generate_usage_output_scatter_recent(
                all_gw_data, fpl_core_season_data, squad_ids, last_n_gw=5
            )
            print(f'    ✓ League context (last 5 GWs): {recent_file}')
            
            # Also generate squad-only for comparison
            squad_file = plot_gen.generate_usage_output_scatter_squad_only(
                all_gw_data, fpl_core_season_data, squad_ids, gameweek
            )
            print(f'    ✓ Squad-only: {squad_file}')
            squad_recent_file = plot_gen.generate_usage_output_scatter_squad_recent(
                all_gw_data, fpl_core_season_data, squad_ids, gameweek, last_n_gw=5
            )
            print(f'    ✓ Squad-only (last 5 GWs): {squad_recent_file}')
    
    print(f'\n{"="*60}')
    print(f'SUCCESS! All plots generated')
    print(f'{"="*60}')
    print(f'Output directory: {output_dir.absolute()}')
    print(f'\nGenerated files:')
    for f in sorted(output_dir.glob('*.png')):
        if any(pattern in f.name for pattern in ['clinical', 'clutch', 'usage']):
            size_kb = f.stat().st_size / 1024
            print(f'  - {f.name} ({size_kb:.0f} KB)')


if __name__ == '__main__':
    main()

