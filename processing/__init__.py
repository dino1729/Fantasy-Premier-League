"""Processing module - Data cleaning, parsing, and merging utilities.

This module contains scripts for:
- Data cleaning (cleaners.py)
- Data parsing (parsers.py)
- Data merging (mergers.py)
- GW data collection (collector.py)
- Data aggregation (aggregated_points_goals.py)

Main entry points:
- global_merger.py - Merge multi-season data
"""

from .cleaners import clean_players, id_players, get_player_ids
from .parsers import (
    parse_players,
    parse_player_history,
    parse_player_gw_history,
    parse_fixtures,
    parse_team_data
)
from .mergers import (
    import_merged_gw,
    clean_players_name_string,
    filter_players_exist_latest,
    get_opponent_team_name,
    export_cleaned_data
)
from .collector import (
    get_teams,
    get_fixtures,
    get_positions,
    collect_gw,
    merge_gw,
    collect_all_gws,
    merge_all_gws
)

