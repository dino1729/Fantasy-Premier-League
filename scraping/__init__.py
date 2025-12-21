"""Scraping module - FPL API and external data source fetchers.

This module contains scripts for fetching data from:
- FPL API (fpl_api.py) - Core FPL data
- Understat (understat.py) - xG/xA data
- FBref (fbref.py) - Detailed player statistics
- Top managers data (top_managers.py)
- Team-specific data (teams_scraper.py)

Main entry point: global_scraper.py
"""

from .fpl_api import (
    get_data,
    get_individual_player_data,
    get_entry_data,
    get_entry_personal_data,
    get_entry_gws_data,
    get_entry_transfers_data,
    get_fixtures_data,
    get_classic_league_standings,
    get_entry_picks_for_gw
)

