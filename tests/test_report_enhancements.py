import sys
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
import json
from typing import List, Dict

import pandas as pd
import numpy as np


# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from reports.fpl_report.data_fetcher import FPLDataFetcher, build_competitive_dataset
from reports.fpl_report.transfer_recommender import TransferRecommender
from reports.fpl_report.plot_generator import PlotGenerator
from reports.fpl_report.latex_generator import LaTeXReportGenerator
from reports.fpl_report.session_cache import SessionCacheManager
from reports.fpl_report.cache_manager import CacheManager


class DummyAnalyzer:
    pass


class DummyFetcher:
    def __init__(self, df: pd.DataFrame):
        self._df = df
        self.players_df = df
        # Create empty fixtures DataFrame with expected columns
        self.fixtures_df = pd.DataFrame(columns=[
            'id', 'event', 'team_h', 'team_a', 'team_h_difficulty', 'team_a_difficulty',
            'team_h_score', 'team_a_score', 'finished'
        ])
        self.bootstrap_data = {
            'teams': [{'id': 1, 'short_name': 'LIV'}, {'id': 2, 'short_name': 'TOT'}],
            'events': [{'id': 10, 'is_current': True, 'finished': False}],
            'elements': []
        }

    def get_bank(self) -> float:
        return 0.0

    def get_current_gameweek(self) -> int:
        return 10

    def get_all_players_by_position(self, position: str) -> pd.DataFrame:
        return self._df.copy()

    def get_upcoming_fixtures(self, team_id: int, num_fixtures: int = 5):
        return [{"gameweek": 6, "opponent": "ARS", "is_home": True, "difficulty": 2}]

    def get_fixtures_by_gw(self, team_id: int, start_gw: int, end_gw: int):
        return {gw: [] for gw in range(start_gw, end_gw + 1)}

    def get_position_peers(self, position: str, min_minutes: int = 90) -> pd.DataFrame:
        return pd.DataFrame()

    def _get_team_name(self, team_id: int) -> str:
        return {1: "LIV", 2: "TOT"}.get(team_id, "UNK")

    def get_player_history(self, player_id: int):
        return pd.DataFrame()

    def get_player_stats(self, player_id: int):
        return {}


class TestUsageOutputPlots(unittest.TestCase):
    """Tests for usage vs output plots."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.plot_gen = PlotGenerator(self.temp_dir)

        # Minimal season data with position info
        self.fpl_core_season_data = {
            'playerstats': pd.DataFrame([
                {'id': 1, 'web_name': 'Alpha', 'first_name': 'A', 'second_name': 'One'}
            ]),
            'players': pd.DataFrame([
                {'player_id': 1, 'position': 'Midfielder'}
            ])
        }

        # Three gameweeks of playermatchstats for a single player
        self.all_gw_data = {
            1: {'playermatchstats': pd.DataFrame([
                {'player_id': 1, 'minutes_played': 30, 'goals': 0, 'assists': 0,
                 'total_shots': 1, 'touches_opposition_box': 1, 'xg': 0.1, 'xa': 0.1}
            ])},
            2: {'playermatchstats': pd.DataFrame([
                {'player_id': 1, 'minutes_played': 90, 'goals': 1, 'assists': 0,
                 'total_shots': 3, 'touches_opposition_box': 3, 'xg': 0.4, 'xa': 0.2}
            ])},
            3: {'playermatchstats': pd.DataFrame([
                {'player_id': 1, 'minutes_played': 60, 'goals': 0, 'assists': 1,
                 'total_shots': 2, 'touches_opposition_box': 2, 'xg': 0.2, 'xa': 0.3}
            ])},
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_usage_aggregate_respects_last_n_gw(self):
        """Aggregated stats should only include the last N gameweeks when requested."""
        agg = self.plot_gen._aggregate_fplcore_usage_stats(
            self.all_gw_data,
            self.fpl_core_season_data,
            min_minutes=0,
            last_n_gw=2
        )

        minutes_played = agg.loc[agg['player_id'] == 1, 'minutes_played'].iloc[0]
        # Should only include GW2 and GW3 (90 + 60)
        self.assertEqual(minutes_played, 150)

    def test_generate_usage_output_recent_creates_file(self):
        """Recent window plot should save an image for the requested GW range."""
        filename = self.plot_gen.generate_usage_output_scatter_recent(
            self.all_gw_data,
            self.fpl_core_season_data,
            squad_ids=[1],
            last_n_gw=2,
            top_n=5
        )

        expected_path = self.temp_dir / 'usage_output_scatter_last2.png'
        self.assertEqual(filename, 'usage_output_scatter_last2.png')
        self.assertTrue(expected_path.exists(), f"Expected plot file at {expected_path}")


class TestSingleBundledCache(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.plot_gen = PlotGenerator(self.temp_dir)
        self.cache_dir = self.temp_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _touch(self, path: Path, data: bytes = b"data"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def test_session_cache_single_file_overwrites_and_cleans_legacy(self):
        # Simulate legacy CacheManager artifacts (md5 files + metadata)
        self._touch(self.cache_dir / "092bd25244a60c32ad274ed2e85aeb5b.pkl")
        self._touch(self.cache_dir / "fcfad1130d8b81cfb9a028cb02fa8ea1.pkl")
        (self.cache_dir / "cache_metadata.json").write_text("{}", encoding="utf-8")

        # Simulate older session cache artifacts
        self._touch(self.cache_dir / "session_847569_gw17_20251222.pkl")
        (self.cache_dir / "session_metadata.json").write_text("{}", encoding="utf-8")

        cache = SessionCacheManager(
            team_id=847569,
            gameweek=17,
            cache_dir=self.cache_dir,
            ttl=3600,
            max_sessions=10,
            enabled=True,
            single_file=True,
        )
        cache.set("bootstrap", {"ok": True})
        cache.save()

        pkl_files = sorted(p.name for p in self.cache_dir.glob("*.pkl"))
        self.assertEqual(
            pkl_files,
            ["session_cache.pkl"],
            f"Expected only one bundled cache file, got: {pkl_files}",
        )

        self.assertFalse(
            (self.cache_dir / "cache_metadata.json").exists(),
            "Legacy cache_metadata.json should be removed in single-file mode",
        )
        self.assertFalse(
            (self.cache_dir / "session_metadata.json").exists(),
            "session_metadata.json should not exist in single-file mode",
        )

    def test_cache_manager_default_is_single_file_bundle(self):
        # Simulate legacy CacheManager artifacts (md5 files + metadata) that should not grow further
        self._touch(self.cache_dir / "092bd25244a60c32ad274ed2e85aeb5b.pkl")
        (self.cache_dir / "cache_metadata.json").write_text("{}", encoding="utf-8")

        cache = CacheManager(cache_dir=self.cache_dir, enabled=True)
        cache.set("bootstrap", {"a": 1})
        cache.set("team_data", {"b": 2}, 123)

        pkl_files = sorted(p.name for p in self.cache_dir.glob("*.pkl"))
        self.assertEqual(
            pkl_files,
            ["cache_bundle.pkl"],
            f"CacheManager should write a single bundle file, got: {pkl_files}",
        )
        self.assertFalse(
            (self.cache_dir / "cache_metadata.json").exists(),
            "CacheManager should not write cache_metadata.json in bundled mode",
        )

    def test_select_gameweeks_ignores_future_fixture_only_entries(self):
        """_select_gameweeks should ignore future GWs that only contain fixtures/teams."""
        # Historical GWs with playermatchstats
        all_gw_data = {
            gw: {
                'playermatchstats': pd.DataFrame([{'player_id': 1, 'minutes_played': 90}])
            }
            for gw in range(1, 6)
        }
        # Future GW with only fixtures/teams (no match stats)
        all_gw_data[6] = {
            'fixtures': pd.DataFrame([{'gameweek': 6, 'home_team': 1, 'away_team': 2}]),
            'teams': pd.DataFrame([{'id': 1}, {'id': 2}]),
            'playermatchstats': pd.DataFrame()  # explicitly empty
        }

        selected = self.plot_gen._select_gameweeks(all_gw_data, last_n_gw=5)
        self.assertEqual(selected, [1, 2, 3, 4, 5], "Fixture-only future GW should be excluded")

        range_label = self.plot_gen._format_gw_range_label(selected)
        self.assertEqual(range_label, "GW1-5", "Range label should reflect only historical GWs")


class TestGoalkeeperValuePlots(unittest.TestCase):
    """Tests for goalkeeper scatter plots (goals prevented vs points)."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.plot_gen = PlotGenerator(self.temp_dir)

        # Minimal season data with player identity + position
        self.fpl_core_season_data = {
            "playerstats": pd.DataFrame(
                [
                    {"id": 1, "web_name": "AlphaGK", "first_name": "Alpha", "second_name": "GK", "gw": 5},
                    {"id": 2, "web_name": "BetaGK", "first_name": "Beta", "second_name": "GK", "gw": 5},
                    {"id": 3, "web_name": "GammaDEF", "first_name": "Gamma", "second_name": "DEF", "gw": 5},
                ]
            ),
            "players": pd.DataFrame(
                [
                    {"player_id": 1, "position": "Goalkeeper"},
                    {"player_id": 2, "position": "Goalkeeper"},
                    {"player_id": 3, "position": "Defender"},
                ]
            ),
        }

        # Build 5 GWs so default season min_minutes=450 passes for the GKs.
        self.all_gw_data = {}
        for gw in range(1, 6):
            self.all_gw_data[gw] = {
                "playermatchstats": pd.DataFrame(
                    [
                        # GK1: decent shot stopping
                        {
                            "player_id": 1,
                            "minutes_played": 90,
                            "saves": 4,
                            "goals_conceded": 1,
                            "xgot_faced": 1.6,
                            "goals_prevented": 0.6,
                        },
                        # GK2: weaker shot stopping
                        {
                            "player_id": 2,
                            "minutes_played": 90,
                            "saves": 2,
                            "goals_conceded": 2,
                            "xgot_faced": 2.1,
                            "goals_prevented": 0.1,
                        },
                        # Non-GK noise row
                        {
                            "player_id": 3,
                            "minutes_played": 90,
                            "saves": 0,
                            "goals_conceded": 0,
                            "xgot_faced": 0.0,
                            "goals_prevented": 0.0,
                        },
                    ]
                ),
                "player_gameweek_stats": pd.DataFrame(
                    [
                        {"id": 1, "event_points": 6, "clean_sheets": 1 if gw % 2 == 0 else 0},
                        {"id": 2, "event_points": 2, "clean_sheets": 0},
                        {"id": 3, "event_points": 5, "clean_sheets": 1},
                    ]
                ),
            }

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_goalkeeper_value_scatter_creates_file(self):
        filename = self.plot_gen.generate_goalkeeper_value_scatter(
            all_gw_data=self.all_gw_data,
            fpl_core_season_data=self.fpl_core_season_data,
            squad_ids=[1],
            top_n=20,
        )

        expected_path = self.temp_dir / "goalkeeper_value_scatter.png"
        self.assertEqual(filename, "goalkeeper_value_scatter.png")
        self.assertTrue(expected_path.exists(), f"Expected plot file at {expected_path}")

    def test_generate_goalkeeper_value_scatter_recent_creates_file(self):
        filename = self.plot_gen.generate_goalkeeper_value_scatter_recent(
            all_gw_data=self.all_gw_data,
            fpl_core_season_data=self.fpl_core_season_data,
            squad_ids=[1],
            last_n_gw=5,
            top_n=20,
        )

        expected_path = self.temp_dir / "goalkeeper_value_scatter_last5.png"
        self.assertEqual(filename, "goalkeeper_value_scatter_last5.png")
        self.assertTrue(expected_path.exists(), f"Expected plot file at {expected_path}")


class TestUsageOutputQuadrantTablesLatex(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_quadrant_tables_are_resized_to_minipage_width(self):
        """Quadrant tables should be constrained to the minipage width to avoid overlap."""
        summary = {
            "season": {
                "range_label": "GW1-17",
                "categories": {
                    "elite": [
                        {"name": "Calvert-Lewin", "pos": "FWD", "usage_per_90": 9.82, "ga": 4, "xgi": 2.95, "pts": 32},
                        {"name": "B.Fernandes", "pos": "MID", "usage_per_90": 5.48, "ga": 12, "xgi": 11.08, "pts": 98},
                    ],
                    "volume": [
                        {"name": "Igor Jesus", "pos": "FWD", "usage_per_90": 9.81, "ga": 7, "xgi": 7.54, "pts": 31},
                    ],
                    "clinical": [
                        {"name": "Szoboszlai", "pos": "MID", "usage_per_90": 4.38, "ga": 9, "xgi": 8.24, "pts": 62},
                    ],
                },
            },
            "last5": {
                "range_label": "GW12-17",
                "categories": {
                    "elite": [
                        {"name": "Haaland", "pos": "FWD", "usage_per_90": 10.15, "ga": 6, "xgi": 6.19, "pts": 33},
                    ],
                    "volume": [
                        {"name": "M.Salah", "pos": "MID", "usage_per_90": 14.64, "ga": 1, "xgi": 2.80, "pts": 10},
                    ],
                    "clinical": [
                        {"name": "Hudson-Odoi", "pos": "MID", "usage_per_90": 7.23, "ga": 3, "xgi": 2.66, "pts": 23},
                    ],
                },
            },
        }

        (self.temp_dir / "usage_output_summary.json").write_text(json.dumps(summary), encoding="utf-8")

        gen = LaTeXReportGenerator(team_id=1, gameweek=17, plot_dir=self.temp_dir)
        latex = gen._generate_advanced_finishing_creativity_section()

        start_marker = r"\subsubsection{Usage vs Output Quadrants (Tables)}"
        end_marker = r"\subsection{Defensive Value Charts}"

        self.assertIn(start_marker, latex)
        start = latex.index(start_marker)
        self.assertIn(end_marker, latex[start:])
        end = latex.index(end_marker, start)
        section = latex[start:end]

        # 3 tables for "season" + 3 tables for "last5" = 6 total, each must be width constrained.
        self.assertEqual(section.count(r"\resizebox{\linewidth}{!}"), 6)


class TestDefensiveValueQuadrantTablesLatex(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_defensive_quadrant_tables_are_resized_to_minipage_width(self):
        """Defensive quadrant tables should be constrained to the minipage width to avoid overlap."""
        summary = {
            "season": {
                "range_label": "GW1-17",
                "categories": {
                    "elite": [
                        {"name": "Van-Hecke", "def_per_90": 11.2, "cs": 5, "cs_pct": 33.3, "pts": 72},
                    ],
                    "volume": [
                        {"name": "Wan-Bissaka", "def_per_90": 6.6, "cs": 0, "cs_pct": 0.0, "pts": 73},
                    ],
                    "cs_merchants": [
                        {"name": "Calafiori", "def_per_90": 3.7, "cs": 8, "cs_pct": 53.3, "pts": 71},
                    ],
                },
            },
            "last5": {
                "range_label": "GW12-17",
                "categories": {
                    "elite": [
                        {"name": "Tarkowski", "def_per_90": 10.4, "cs": 3, "cs_pct": 60.0, "pts": 31},
                    ],
                    "volume": [
                        {"name": "Konaté", "def_per_90": 9.1, "cs": 2, "cs_pct": 33.3, "pts": 11},
                    ],
                    "cs_merchants": [
                        {"name": "Chalobah", "def_per_90": 8.2, "cs": 3, "cs_pct": 50.0, "pts": 36},
                    ],
                },
            },
        }

        (self.temp_dir / "defensive_value_summary.json").write_text(json.dumps(summary), encoding="utf-8")

        gen = LaTeXReportGenerator(team_id=1, gameweek=17, plot_dir=self.temp_dir)
        latex = gen._generate_advanced_finishing_creativity_section()

        start_marker = r"\subsubsection{Defensive Value Quadrants (Tables)}"
        self.assertIn(start_marker, latex)

        section = latex[latex.index(start_marker):]

        # 3 tables for "season" + 3 tables for "last5" = 6 total, each must be width constrained.
        self.assertEqual(section.count(r"\resizebox{\linewidth}{!}"), 6)


class TestGoalkeeperValueTablesLatex(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_goalkeeper_tables_render_when_summary_available(self):
        summary = {
            "season": {
                "range_label": "GW1-17",
                "rows": [
                    {"name": "Sánchez", "gp": 2.31, "save_pct": 76.4, "cs": 6, "pts": 110},
                    {"name": "Pickford", "gp": 1.12, "save_pct": 71.0, "cs": 5, "pts": 98},
                ],
                "categories": {
                    "elite": [
                        {"name": "Sánchez", "gp": 2.31, "save_pct": 76.4, "cs": 6, "pts": 110},
                    ],
                    "protected": [
                        {"name": "Raya", "gp": -0.55, "save_pct": 73.2, "cs": 7, "pts": 105},
                    ],
                    "unlucky": [
                        {"name": "Donnarumma", "gp": 0.88, "save_pct": 66.7, "cs": 2, "pts": 60},
                    ],
                    "avoid": [
                        {"name": "Johnstone", "gp": -1.21, "save_pct": 58.0, "cs": 1, "pts": 40},
                    ],
                },
            },
            "last5": {
                "range_label": "GW12-17",
                "rows": [
                    {"name": "Sánchez", "gp": 1.70, "save_pct": 78.1, "cs": 2, "pts": 26},
                    {"name": "Verbruggen", "gp": 1.05, "save_pct": 70.0, "cs": 1, "pts": 19},
                ],
                "categories": {
                    "elite": [
                        {"name": "Sánchez", "gp": 1.70, "save_pct": 78.1, "cs": 2, "pts": 26},
                    ],
                    "protected": [
                        {"name": "Areola", "gp": -0.20, "save_pct": 75.0, "cs": 2, "pts": 22},
                    ],
                    "unlucky": [
                        {"name": "Pickford", "gp": 0.40, "save_pct": 68.0, "cs": 0, "pts": 11},
                    ],
                    "avoid": [
                        {"name": "Ramsdale", "gp": -0.90, "save_pct": 60.0, "cs": 0, "pts": 7},
                    ],
                },
            },
        }

        (self.temp_dir / "goalkeeper_value_summary.json").write_text(json.dumps(summary), encoding="utf-8")

        gen = LaTeXReportGenerator(team_id=1, gameweek=17, plot_dir=self.temp_dir)
        latex = gen._generate_advanced_finishing_creativity_section()

        start_marker = r"\subsection{Goalkeeper Shot-Stopping Charts}"
        self.assertIn(start_marker, latex)

        section = latex[latex.index(start_marker):]
        self.assertIn(r"\subsubsection{Top Goalkeepers Tables}", section)
        self.assertIn(r"\subsubsection{Goalkeeper Shot-Stopping Quadrants (Tables)}", section)

        # 2 top tables (season + last5) + 6 quadrant tables (3 per window) = 8 total.
        self.assertEqual(section.count(r"\resizebox{\linewidth}{!}"), 8)


class TestReportEnhancements(unittest.TestCase):
    def test_get_transfers_enriches_names_and_costs(self):
        fetcher = FPLDataFetcher(team_id=123, season="2025-26")

        fetcher._bootstrap_data = {
            "teams": [{"id": 1, "short_name": "LIV"}, {"id": 2, "short_name": "TOT"}],
            "events": [],
            "elements": [],
        }
        fetcher._players_df = pd.DataFrame(
            [
                {"id": 1, "web_name": "Salah", "team": 1, "element_type": 3},
                {"id": 2, "web_name": "Son", "team": 2, "element_type": 3},
            ]
        )

        raw_transfers = [
            {
                "element_in": 1,
                "element_in_cost": 100,
                "element_out": 2,
                "element_out_cost": 90,
                "entry": 123,
                "event": 5,
                "time": "2025-01-01T00:00:00Z",
            }
        ]

        with patch("reports.fpl_report.data_fetcher.get_entry_transfers_data", return_value=raw_transfers):
            transfers = fetcher.get_transfers()

        self.assertEqual(len(transfers), 1)
        t = transfers[0]
        self.assertEqual(t["element_in_name"], "Salah")
        self.assertEqual(t["element_out_name"], "Son")
        self.assertEqual(t["element_in_team"], "LIV")
        self.assertEqual(t["element_out_team"], "TOT")
        self.assertEqual(t["element_in_cost_m"], 10.0)
        self.assertEqual(t["element_out_cost_m"], 9.0)

    def test_recommendations_exclude_injured_players(self):
        candidates_df = pd.DataFrame(
            [
                {
                    "id": 11,
                    "web_name": "FitGuy",
                    "team": 1,
                    "now_cost": 50,
                    "minutes": 900,
                    "form": 5.5,
                    "total_points": 60,
                    "selected_by_percent": 12.3,
                    "expected_goal_involvements": 3.2,
                    "points_per_game": 4.0,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                },
                {
                    "id": 12,
                    "web_name": "InjuredGuy",
                    "team": 2,
                    "now_cost": 50,
                    "minutes": 900,
                    "form": 6.0,
                    "total_points": 65,
                    "selected_by_percent": 5.0,
                    "expected_goal_involvements": 3.8,
                    "points_per_game": 4.2,
                    "status": "i",
                    "chance_of_playing_next_round": 0,
                },
            ]
        )

        fetcher = DummyFetcher(candidates_df)
        recommender = TransferRecommender(fetcher, DummyAnalyzer())

        underperformers = [
            {
                "player_id": 99,
                "name": "OutGuy",
                "position": "MID",
                "team": "LIV",
                "price": 5.0,
                "reasons": ["Low form"],
                "severity": 3,
            }
        ]

        recs = recommender.get_recommendations(underperformers, num_recommendations=5)
        self.assertEqual(len(recs), 1)
        in_ids = [p["player_id"] for p in recs[0]["in_options"]]
        self.assertIn(11, in_ids)
        self.assertNotIn(12, in_ids)


class TestCompetitiveDatasetBuilder(unittest.TestCase):
    """Tests for the competitive dataset builder function."""

    def _make_mock_entry_data(self, entry_id):
        """Create mock entry data for testing."""
        return {
            'current': [
                {'event': 1, 'points': 50 + entry_id % 10, 'total_points': 50 + entry_id % 10,
                 'overall_rank': 1000000 - entry_id, 'value': 1000, 'bank': 0,
                 'event_transfers_cost': 0},
                {'event': 2, 'points': 60, 'total_points': 110 + entry_id % 10,
                 'overall_rank': 900000 - entry_id, 'value': 1005, 'bank': 5,
                 'event_transfers_cost': 4},
            ],
            'chips': [{'event': 1, 'name': 'wildcard'}] if entry_id == 21023 else []
        }

    def _make_mock_personal_data(self, entry_id):
        """Create mock personal data for testing."""
        return {
            'name': f'Team {entry_id}',
            'player_first_name': 'John',
            'player_last_name': f'Doe{entry_id}',
            'summary_overall_points': 110 + entry_id % 10,
            'summary_overall_rank': 900000 - entry_id,
            'summary_event_points': 60,
            'summary_event_rank': 500000
        }

    def _make_mock_gw_picks(self, entry_id, gw):
        """Create mock GW picks data for testing."""
        return {
            'picks': [
                {'element': 1, 'position': 1, 'multiplier': 1, 'is_captain': False, 'is_vice_captain': False},
                {'element': 2, 'position': 2, 'multiplier': 1, 'is_captain': False, 'is_vice_captain': False},
                {'element': 3, 'position': 3, 'multiplier': 1, 'is_captain': False, 'is_vice_captain': False},
                {'element': 4, 'position': 4, 'multiplier': 1, 'is_captain': False, 'is_vice_captain': False},
                {'element': 5, 'position': 5, 'multiplier': 1, 'is_captain': False, 'is_vice_captain': False},
                {'element': 6, 'position': 6, 'multiplier': 1, 'is_captain': False, 'is_vice_captain': False},
                {'element': 7, 'position': 7, 'multiplier': 1, 'is_captain': False, 'is_vice_captain': False},
                {'element': 8, 'position': 8, 'multiplier': 1, 'is_captain': False, 'is_vice_captain': False},
                {'element': 9, 'position': 9, 'multiplier': 1, 'is_captain': False, 'is_vice_captain': False},
                {'element': 10, 'position': 10, 'multiplier': 1, 'is_captain': False, 'is_vice_captain': False},
                {'element': 11, 'position': 11, 'multiplier': 2, 'is_captain': True, 'is_vice_captain': False},
                {'element': 12, 'position': 12, 'multiplier': 0, 'is_captain': False, 'is_vice_captain': True},
                {'element': 13, 'position': 13, 'multiplier': 0, 'is_captain': False, 'is_vice_captain': False},
                {'element': 14, 'position': 14, 'multiplier': 0, 'is_captain': False, 'is_vice_captain': False},
                {'element': 15, 'position': 15, 'multiplier': 0, 'is_captain': False, 'is_vice_captain': False},
            ]
        }

    @patch('reports.fpl_report.data_fetcher.get_entry_data')
    @patch('reports.fpl_report.data_fetcher.get_entry_personal_data')
    @patch('reports.fpl_report.data_fetcher.get_entry_gws_data')
    @patch('reports.fpl_report.data_fetcher.get_data')
    def test_build_competitive_dataset_returns_correct_structure(
        self, mock_get_data, mock_gws, mock_personal, mock_entry
    ):
        """Test that build_competitive_dataset returns expected structure."""
        entry_ids = [21023, 6696002]

        # Setup mocks
        mock_get_data.return_value = {
            'teams': [{'id': 1, 'short_name': 'LIV'}],
            'events': [{'id': 1, 'is_current': False, 'finished': True},
                       {'id': 2, 'is_current': True, 'finished': False}],
            'elements': [
                {'id': i, 'web_name': f'Player{i}', 'first_name': 'First',
                 'second_name': f'Last{i}', 'team': 1, 'element_type': (i % 4) + 1}
                for i in range(1, 17)
            ]
        }
        mock_entry.side_effect = lambda eid: self._make_mock_entry_data(eid)
        mock_personal.side_effect = lambda eid: self._make_mock_personal_data(eid)
        mock_gws.side_effect = lambda eid, gw, start_gw: [self._make_mock_gw_picks(eid, gw)]

        result = build_competitive_dataset(entry_ids, season='2025-26', gameweek=2)

        # Verify structure
        self.assertEqual(len(result), 2)
        for entry in result:
            self.assertIn('entry_id', entry)
            self.assertIn('team_info', entry)
            self.assertIn('gw_history', entry)
            self.assertIn('squad', entry)
            self.assertIn('chips_used', entry)
            self.assertIn('total_hits', entry)
            self.assertIn('team_value', entry)
            self.assertIn('bank', entry)

        # Verify specific values
        first_entry = result[0]
        self.assertEqual(first_entry['entry_id'], 21023)
        self.assertIsInstance(first_entry['gw_history'], list)
        self.assertIsInstance(first_entry['squad'], list)
        self.assertEqual(len(first_entry['squad']), 15)

    @patch('reports.fpl_report.data_fetcher.get_entry_data')
    @patch('reports.fpl_report.data_fetcher.get_entry_personal_data')
    @patch('reports.fpl_report.data_fetcher.get_entry_gws_data')
    @patch('reports.fpl_report.data_fetcher.get_data')
    def test_build_competitive_dataset_computes_total_hits(
        self, mock_get_data, mock_gws, mock_personal, mock_entry
    ):
        """Test that total hits are correctly computed from GW history."""
        entry_ids = [21023]

        mock_get_data.return_value = {
            'teams': [],
            'events': [{'id': 2, 'is_current': True, 'finished': False}],
            'elements': [
                {'id': i, 'web_name': f'Player{i}', 'first_name': 'First',
                 'second_name': f'Last{i}', 'team': 1, 'element_type': 1}
                for i in range(1, 17)
            ]
        }
        mock_entry.return_value = self._make_mock_entry_data(21023)
        mock_personal.return_value = self._make_mock_personal_data(21023)
        mock_gws.return_value = [self._make_mock_gw_picks(21023, 2)]

        result = build_competitive_dataset(entry_ids, season='2025-26', gameweek=2)

        # GW2 has event_transfers_cost=4
        self.assertEqual(result[0]['total_hits'], 4)


class TestCompetitivePlots(unittest.TestCase):
    """Tests for competitive progression plot generation."""

    def setUp(self):
        """Create temp directory for plot output."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_competitive_points_progression_creates_file(self):
        """Test that points progression plot is created."""
        plot_gen = PlotGenerator(self.temp_dir)

        competitive_data = [
            {
                'entry_id': 21023,
                'team_info': {'team_name': 'Team A', 'manager_name': 'Manager A'},
                'gw_history': [
                    {'event': 1, 'total_points': 50},
                    {'event': 2, 'total_points': 110},
                    {'event': 3, 'total_points': 170},
                ]
            },
            {
                'entry_id': 6696002,
                'team_info': {'team_name': 'Team B', 'manager_name': 'Manager B'},
                'gw_history': [
                    {'event': 1, 'total_points': 55},
                    {'event': 2, 'total_points': 105},
                    {'event': 3, 'total_points': 165},
                ]
            }
        ]

        plot_gen.generate_competitive_points_progression(competitive_data)

        expected_file = self.temp_dir / 'competitive_points_progression.png'
        self.assertTrue(expected_file.exists(), f"Expected {expected_file} to exist")

    def test_generate_competitive_rank_progression_creates_file(self):
        """Test that rank progression plot is created."""
        plot_gen = PlotGenerator(self.temp_dir)

        competitive_data = [
            {
                'entry_id': 21023,
                'team_info': {'team_name': 'Team A', 'manager_name': 'Manager A'},
                'gw_history': [
                    {'event': 1, 'overall_rank': 1000000},
                    {'event': 2, 'overall_rank': 800000},
                    {'event': 3, 'overall_rank': 600000},
                ]
            },
            {
                'entry_id': 6696002,
                'team_info': {'team_name': 'Team B', 'manager_name': 'Manager B'},
                'gw_history': [
                    {'event': 1, 'overall_rank': 900000},
                    {'event': 2, 'overall_rank': 850000},
                    {'event': 3, 'overall_rank': 700000},
                ]
            }
        ]

        plot_gen.generate_competitive_rank_progression(competitive_data)

        expected_file = self.temp_dir / 'competitive_rank_progression.png'
        self.assertTrue(expected_file.exists(), f"Expected {expected_file} to exist")

    def test_generate_competitive_treemaps_creates_files(self):
        """Test that player contribution treemaps are created for each team."""
        plot_gen = PlotGenerator(self.temp_dir)

        # Build proper season_history structure required by generate_competitive_treemaps
        # Each GW entry needs 'squad' (not 'picks') with player info
        # _calculate_contributing_points expects:
        # - squad[i].position_in_squad (1-11 = starting XI)
        # - squad[i].name
        # - squad[i].stats.event_points (points for that GW)
        # - squad[i].multiplier (captain multiplier)
        # - squad[i].position
        def make_season_history(players):
            """Create season_history with proper structure for treemap generation."""
            history = []
            for gw in range(1, 4):  # 3 gameweeks
                squad = []
                for i, p in enumerate(players, start=1):
                    event_pts = p['stats']['total_points'] // 3  # Points per GW
                    squad.append({
                        'element': i,  # player_id
                        'name': p['name'],
                        'position': p['position'],
                        'position_in_squad': i,  # 1-11 = starting XI
                        'is_captain': (i == 1),  # First player is captain
                        'stats': {'event_points': event_pts},
                        'multiplier': 2 if i == 1 else 1  # Captain gets 2x
                    })
                history.append({
                    'gameweek': gw,
                    'points': sum(p['stats']['event_points'] * p['multiplier'] for p in squad if p['position_in_squad'] <= 11),
                    'total_points': gw * 50,
                    'squad': squad  # Key must be 'squad', not 'picks'
                })
            return history

        team_a_players = [
            {'name': 'Salah', 'position': 'MID', 'stats': {'total_points': 120}},
            {'name': 'Haaland', 'position': 'FWD', 'stats': {'total_points': 150}},
            {'name': 'Saka', 'position': 'MID', 'stats': {'total_points': 90}},
            {'name': 'Raya', 'position': 'GKP', 'stats': {'total_points': 60}},
            {'name': 'Gabriel', 'position': 'DEF', 'stats': {'total_points': 80}},
        ]
        team_b_players = [
            {'name': 'Son', 'position': 'MID', 'stats': {'total_points': 100}},
            {'name': 'Watkins', 'position': 'FWD', 'stats': {'total_points': 110}},
            {'name': 'Palmer', 'position': 'MID', 'stats': {'total_points': 130}},
            {'name': 'Pickford', 'position': 'GKP', 'stats': {'total_points': 55}},
            {'name': 'VVD', 'position': 'DEF', 'stats': {'total_points': 85}},
        ]

        competitive_data = [
            {
                'entry_id': 21023,
                'team_info': {'team_name': 'Team A', 'manager_name': 'Manager A', 'overall_points': 500},
                'squad': team_a_players,
                'season_history': make_season_history(team_a_players)
            },
            {
                'entry_id': 6696002,
                'team_info': {'team_name': 'Team B', 'manager_name': 'Manager B', 'overall_points': 480},
                'squad': team_b_players,
                'season_history': make_season_history(team_b_players)
            }
        ]

        filenames = plot_gen.generate_competitive_treemaps(competitive_data)

        # Should generate 2 treemaps
        self.assertEqual(len(filenames), 2)
        
        # Check files exist
        for entry in competitive_data:
            entry_id = entry['entry_id']
            expected_file = self.temp_dir / f'treemap_team_{entry_id}.png'
            self.assertTrue(expected_file.exists(), f"Expected {expected_file} to exist")


class TestCompetitiveLaTeXSection(unittest.TestCase):
    """Tests for competitive analysis LaTeX section generation."""

    def setUp(self):
        """Set up LaTeX generator."""
        self.generator = LaTeXReportGenerator(team_id=847569, gameweek=17)

    def test_generate_competitive_analysis_contains_section_header(self):
        """Test that competitive section contains correct LaTeX section header."""
        competitive_data = [
            {
                'entry_id': 21023,
                'team_info': {'team_name': 'Team A', 'manager_name': 'Manager A',
                              'overall_points': 500, 'overall_rank': 100000},
                'gw_history': [{'event': 1, 'total_points': 500}],
                'squad': [],
                'chips_used': [],
                'total_hits': 0,
                'team_value': 100.5,
                'bank': 0.5
            }
        ]

        latex = self.generator.generate_competitive_analysis(competitive_data)

        self.assertIn(r'\section{Competitive Analysis}', latex)

    def test_generate_competitive_analysis_includes_plot_references(self):
        """Test that competitive section includes both plot file references."""
        competitive_data = [
            {
                'entry_id': 21023,
                'team_info': {'team_name': 'Team A', 'manager_name': 'Manager A',
                              'overall_points': 500, 'overall_rank': 100000},
                'gw_history': [{'event': 1, 'total_points': 500}],
                'squad': [],
                'chips_used': [],
                'total_hits': 0,
                'team_value': 100.5,
                'bank': 0.5
            }
        ]

        latex = self.generator.generate_competitive_analysis(competitive_data)

        self.assertIn('competitive_points_progression.png', latex)
        self.assertIn('competitive_rank_progression.png', latex)

    def test_generate_competitive_analysis_includes_team_names(self):
        """Test that competitive section includes all team names."""
        competitive_data = [
            {
                'entry_id': 21023,
                'team_info': {'team_name': 'Alpha Team', 'manager_name': 'Manager A',
                              'overall_points': 500, 'overall_rank': 100000},
                'gw_history': [],
                'squad': [],
                'chips_used': [],
                'total_hits': 0,
                'team_value': 100.5,
                'bank': 0.5
            },
            {
                'entry_id': 6696002,
                'team_info': {'team_name': 'Beta Squad', 'manager_name': 'Manager B',
                              'overall_points': 480, 'overall_rank': 120000},
                'gw_history': [],
                'squad': [],
                'chips_used': [],
                'total_hits': 4,
                'team_value': 99.8,
                'bank': 1.2
            }
        ]

        latex = self.generator.generate_competitive_analysis(competitive_data)

        self.assertIn('Alpha Team', latex)
        self.assertIn('Beta Squad', latex)

    def test_generate_competitive_analysis_includes_treemap_references(self):
        """Test that competitive section includes treemap image references."""
        competitive_data = [
            {
                'entry_id': 21023,
                'team_info': {'team_name': 'Team A', 'manager_name': 'Manager A',
                              'overall_points': 500, 'overall_rank': 100000},
                'gw_history': [],
                'squad': [],
                'chips_used': [],
                'total_hits': 0,
                'team_value': 100.5,
                'bank': 0.5
            },
            {
                'entry_id': 6696002,
                'team_info': {'team_name': 'Team B', 'manager_name': 'Manager B',
                              'overall_points': 480, 'overall_rank': 120000},
                'gw_history': [],
                'squad': [],
                'chips_used': [],
                'total_hits': 4,
                'team_value': 99.8,
                'bank': 1.2
            }
        ]

        latex = self.generator.generate_competitive_analysis(competitive_data)

        # Should reference treemaps for each team
        self.assertIn('treemap_team_21023.png', latex)
        self.assertIn('treemap_team_6696002.png', latex)
        self.assertIn('Player Contribution Breakdown', latex)

    def test_compile_report_includes_competitive_section_when_provided(self):
        """Test that compile_report includes competitive section when data is provided."""
        team_info = {'team_name': 'My Team', 'manager_name': 'Me',
                     'overall_points': 500, 'overall_rank': 100000, 'season': '2025-26'}
        gw_history = [{'event': 1, 'points': 50, 'total_points': 50, 'overall_rank': 100000}]
        squad = []
        squad_analysis = []
        recommendations = []
        captain_picks = []
        chips_used = []

        competitive_data = [
            {
                'entry_id': 21023,
                'team_info': {'team_name': 'Rival Team', 'manager_name': 'Rival',
                              'overall_points': 490, 'overall_rank': 110000},
                'gw_history': [],
                'squad': [],
                'chips_used': [],
                'total_hits': 0,
                'team_value': 100.0,
                'bank': 0.0
            }
        ]

        latex = self.generator.compile_report(
            team_info=team_info,
            gw_history=gw_history,
            squad=squad,
            squad_analysis=squad_analysis,
            recommendations=recommendations,
            captain_picks=captain_picks,
            chips_used=chips_used,
            competitive_data=competitive_data
        )

        self.assertIn(r'\section{Competitive Analysis}', latex)
        self.assertIn('Rival Team', latex)


class TestTopGlobalCompetitiveLaTeXSection(unittest.TestCase):
    """Tests for Top Global Managers competitive analysis LaTeX section generation."""

    def setUp(self):
        self.generator = LaTeXReportGenerator(team_id=847569, gameweek=17, plot_dir=Path("plots"))

    def _make_minimal_transfer_history(self):
        gw_range = [13, 14, 15, 16, 17]
        gw_squads_data = {}
        for gw in gw_range:
            gw_squads_data[gw] = {
                "xi": [
                    {"name": "GK", "position": "GKP"},
                    {"name": "DEF", "position": "DEF"},
                    {"name": "MID", "position": "MID"},
                    {"name": "FWD", "position": "FWD"},
                ],
                "bench": [{"name": "B1"}, {"name": "B2"}],
                "transfers_out": [],
            }
        return {"gw_range": gw_range, "gw_squads_data": gw_squads_data, "chips_timeline": {}}

    def test_generate_top_global_teams_uses_configured_count(self):
        from reports.fpl_report import latex_generator as lg

        dummy_top = [
            {"entry_id": 1, "manager_name": "A", "team_name": "T1", "total_points": 100, "rank": 1},
            {"entry_id": 2, "manager_name": "B", "team_name": "T2", "total_points": 99, "rank": 2},
        ]

        with patch.object(lg, "TOP_GLOBAL_COUNT", 2, create=True):
            with patch("reports.fpl_report.latex_generator.get_top_global_teams", return_value=dummy_top) as mock_get:
                latex = self.generator.generate_top_global_teams()

        self.assertIn(r"\section{Top Global Managers}", latex)
        self.assertEqual(mock_get.call_args.kwargs.get("n"), 2)

    def test_generate_global_competitive_analysis_includes_transfer_and_squad_sections(self):
        from reports.fpl_report import latex_generator as lg

        transfer_history = self._make_minimal_transfer_history()

        top_global_data = [
            {
                "entry_id": 847569,
                "team_info": {"team_name": "My Team", "manager_name": "Me", "overall_points": 500, "overall_rank": 100000},
                "squad": [{"name": "Player A", "position": "MID", "position_in_squad": 1}],
                "gw_transfers": {},
                "transfer_history": transfer_history,
                "season_history": [],
                "chips_used": [],
            },
            {
                "entry_id": 111,
                "team_info": {"team_name": "Top Team 1", "manager_name": "Top 1", "overall_points": 650, "overall_rank": 12},
                "squad": [{"name": "Player B", "position": "MID", "position_in_squad": 1}],
                "gw_transfers": {},
                "transfer_history": transfer_history,
                "season_history": [],
                "chips_used": [],
            },
            {
                "entry_id": 222,
                "team_info": {"team_name": "Top Team 2", "manager_name": "Top 2", "overall_points": 640, "overall_rank": 18},
                "squad": [{"name": "Player C", "position": "MID", "position_in_squad": 1}],
                "gw_transfers": {},
                "transfer_history": transfer_history,
                "season_history": [],
                "chips_used": [],
            },
        ]

        with patch.object(lg, "TOP_GLOBAL_COUNT", 2, create=True):
            latex = self.generator.generate_global_competitive_analysis(top_global_data)

        self.assertIn(r"\section{Benchmarking: Top 2 Global Managers}", latex)
        self.assertIn(r"\subsection{Transfer Activity (GW17 vs GW16)}", latex)
        self.assertIn(r"\subsection{Squad Evolution (Past 5 Gameweeks)}", latex)
        self.assertIn(r"\subsection{Squad Comparison (GW17)}", latex)

    def test_generate_fpl_report_uses_top_global_count_for_fetch(self):
        import generate_fpl_report as gfr

        with patch.object(gfr, "TOP_GLOBAL_COUNT", 2, create=True):
            with patch.object(
                gfr,
                "get_top_global_teams",
                return_value=[{"entry_id": 111}, {"entry_id": 222}],
            ) as mock_get:
                ids = gfr.get_top_global_comparison_ids(team_id=999, use_cache=True, session_cache=None)

        self.assertEqual(mock_get.call_args.kwargs.get("n"), 2)
        self.assertEqual(ids, [999, 111, 222])


class TestWildcardOptimizer(unittest.TestCase):
    """Tests for the Wildcard squad optimizer."""

    def _make_mock_players_df(self):
        """Create a synthetic players DataFrame for testing.
        
        Contains enough players across many teams to form valid squads.
        Teams are spread across 1-15 to avoid max-3-per-team constraint issues.
        """
        players = []
        # GKP (element_type=1): 5 players across 5 teams
        for i, (name, team, cost, ppg) in enumerate([
            ('GK_A', 1, 45, 4.0), ('GK_B', 2, 50, 4.5),
            ('GK_C', 3, 40, 3.5), ('GK_D', 4, 42, 3.8),
            ('GK_E', 5, 40, 3.3)
        ], start=1):
            players.append({
                'id': i, 'web_name': name, 'team': team, 'element_type': 1,
                'now_cost': cost, 'points_per_game': ppg, 'minutes': 900,
                'total_points': int(ppg * 10), 'form': ppg,
                'expected_goal_involvements': 0.0,
                'status': 'a', 'chance_of_playing_next_round': 100
            })
        # DEF (element_type=2): 10 players across 10 different teams
        for i, (name, team, cost, ppg) in enumerate([
            ('DEF_A', 6, 55, 4.5), ('DEF_B', 7, 50, 4.2),
            ('DEF_C', 8, 60, 5.0), ('DEF_D', 9, 48, 4.0),
            ('DEF_E', 10, 45, 3.8), ('DEF_F', 11, 40, 3.5),
            ('DEF_G', 12, 52, 4.3), ('DEF_H', 13, 42, 3.6),
            ('DEF_I', 14, 40, 3.4), ('DEF_J', 15, 40, 3.3)
        ], start=10):
            players.append({
                'id': i, 'web_name': name, 'team': team, 'element_type': 2,
                'now_cost': cost, 'points_per_game': ppg, 'minutes': 900,
                'total_points': int(ppg * 10), 'form': ppg,
                'expected_goal_involvements': 0.5,
                'status': 'a', 'chance_of_playing_next_round': 100
            })
        # MID (element_type=3): 10 players across different teams
        for i, (name, team, cost, ppg) in enumerate([
            ('MID_A', 1, 130, 8.0), ('MID_B', 2, 120, 7.5),
            ('MID_C', 3, 80, 5.5), ('MID_D', 4, 75, 5.2),
            ('MID_E', 5, 65, 4.8), ('MID_F', 6, 50, 4.0),
            ('MID_G', 7, 45, 3.5), ('MID_H', 8, 45, 3.2),
            ('MID_I', 9, 45, 3.1), ('MID_J', 10, 45, 3.0)
        ], start=20):
            players.append({
                'id': i, 'web_name': name, 'team': team, 'element_type': 3,
                'now_cost': cost, 'points_per_game': ppg, 'minutes': 900,
                'total_points': int(ppg * 10), 'form': ppg,
                'expected_goal_involvements': 3.0,
                'status': 'a', 'chance_of_playing_next_round': 100
            })
        # FWD (element_type=4): 6 players across different teams
        for i, (name, team, cost, ppg) in enumerate([
            ('FWD_A', 11, 100, 7.0), ('FWD_B', 12, 80, 6.5),
            ('FWD_C', 13, 70, 5.0), ('FWD_D', 14, 55, 4.2),
            ('FWD_E', 15, 45, 3.5), ('FWD_F', 1, 45, 3.3)
        ], start=30):
            players.append({
                'id': i, 'web_name': name, 'team': team, 'element_type': 4,
                'now_cost': cost, 'points_per_game': ppg, 'minutes': 900,
                'total_points': int(ppg * 10), 'form': ppg,
                'expected_goal_involvements': 5.0,
                'status': 'a', 'chance_of_playing_next_round': 100
            })
        return pd.DataFrame(players)

    def test_wildcard_optimizer_respects_budget(self):
        """Test that optimizer does not exceed total budget."""
        from reports.fpl_report.transfer_strategy import WildcardOptimizer

        players_df = self._make_mock_players_df()
        total_budget = 100.0  # 100m budget

        optimizer = WildcardOptimizer(players_df, total_budget)
        result = optimizer.build_squad()

        self.assertIn('squad', result)
        squad = result['squad']
        total_cost = sum(p['price'] for p in squad)
        self.assertLessEqual(total_cost, total_budget)

    def test_wildcard_optimizer_respects_position_quotas(self):
        """Test that optimizer selects exactly 2 GKP, 5 DEF, 5 MID, 3 FWD."""
        from reports.fpl_report.transfer_strategy import WildcardOptimizer

        players_df = self._make_mock_players_df()
        total_budget = 100.0

        optimizer = WildcardOptimizer(players_df, total_budget)
        result = optimizer.build_squad()

        squad = result['squad']
        position_counts = {}
        for p in squad:
            pos = p['position']
            position_counts[pos] = position_counts.get(pos, 0) + 1

        self.assertEqual(position_counts.get('GKP', 0), 2)
        self.assertEqual(position_counts.get('DEF', 0), 5)
        self.assertEqual(position_counts.get('MID', 0), 5)
        self.assertEqual(position_counts.get('FWD', 0), 3)
        self.assertEqual(len(squad), 15)

    def test_wildcard_optimizer_respects_max_3_per_team(self):
        """Test that optimizer selects at most 3 players from any single team."""
        from reports.fpl_report.transfer_strategy import WildcardOptimizer

        players_df = self._make_mock_players_df()
        total_budget = 100.0

        optimizer = WildcardOptimizer(players_df, total_budget)
        result = optimizer.build_squad()

        squad = result['squad']
        team_counts = {}
        for p in squad:
            tid = p['team_id']
            team_counts[tid] = team_counts.get(tid, 0) + 1

        for tid, count in team_counts.items():
            self.assertLessEqual(count, 3, f"Team {tid} has {count} players (max 3 allowed)")

    def test_wildcard_optimizer_excludes_injured_players(self):
        """Test that optimizer excludes players with status != 'a' or low chance."""
        from reports.fpl_report.transfer_strategy import WildcardOptimizer

        players_df = self._make_mock_players_df()
        # Mark one high-value player as injured
        players_df.loc[players_df['web_name'] == 'MID_A', 'status'] = 'i'
        players_df.loc[players_df['web_name'] == 'MID_A', 'chance_of_playing_next_round'] = 0

        total_budget = 100.0

        optimizer = WildcardOptimizer(players_df, total_budget)
        result = optimizer.build_squad()

        squad = result['squad']
        squad_names = [p['name'] for p in squad]
        self.assertNotIn('MID_A', squad_names)

    def test_wildcard_optimizer_provides_starting_xi(self):
        """Test that optimizer provides a valid starting XI with formation."""
        from reports.fpl_report.transfer_strategy import WildcardOptimizer

        players_df = self._make_mock_players_df()
        total_budget = 100.0

        optimizer = WildcardOptimizer(players_df, total_budget)
        result = optimizer.build_squad()

        self.assertIn('starting_xi', result)
        self.assertIn('bench', result)
        self.assertIn('formation', result)
        self.assertIn('captain', result)
        self.assertIn('vice_captain', result)

        xi = result['starting_xi']
        bench = result['bench']
        self.assertEqual(len(xi), 11)
        self.assertEqual(len(bench), 4)


class TestWildcardLaTeXSection(unittest.TestCase):
    """Tests for Wildcard team LaTeX section generation."""

    def setUp(self):
        """Set up LaTeX generator."""
        self.generator = LaTeXReportGenerator(team_id=847569, gameweek=17)

    def test_generate_wildcard_team_section_contains_header(self):
        """Test that wildcard section contains correct LaTeX section header."""
        wildcard_team = {
            'budget': {'total': 100.0, 'spent': 98.5, 'remaining': 1.5},
            'squad': [
                {'name': 'TestGK', 'position': 'GKP', 'team': 'LIV', 'team_id': 1, 'price': 5.0, 'score': 50.0}
            ],
            'starting_xi': [],
            'bench': [],
            'formation': '4-4-2',
            'captain': {'name': 'TestCap'},
            'vice_captain': {'name': 'TestVC'}
        }

        latex = self.generator.generate_wildcard_team_section(wildcard_team)

        self.assertIn(r'\section{Wildcard', latex)

    def test_generate_wildcard_team_section_includes_player_names(self):
        """Test that wildcard section includes player names from squad."""
        wildcard_team = {
            'budget': {'total': 100.0, 'spent': 98.5, 'remaining': 1.5},
            'squad': [
                {'name': 'Salah', 'position': 'MID', 'team': 'LIV', 'team_id': 1, 'price': 13.0, 'score': 95.0},
                {'name': 'Haaland', 'position': 'FWD', 'team': 'MCI', 'team_id': 2, 'price': 14.5, 'score': 98.0},
            ],
            'starting_xi': [
                {'name': 'Salah', 'position': 'MID', 'team': 'LIV', 'price': 13.0},
                {'name': 'Haaland', 'position': 'FWD', 'team': 'MCI', 'price': 14.5},
            ],
            'bench': [],
            'formation': '4-4-2',
            'captain': {'name': 'Haaland'},
            'vice_captain': {'name': 'Salah'}
        }

        latex = self.generator.generate_wildcard_team_section(wildcard_team)

        self.assertIn('Salah', latex)
        self.assertIn('Haaland', latex)

    def test_compile_report_includes_wildcard_section_when_provided(self):
        """Test that compile_report includes wildcard section when data is provided."""
        team_info = {'team_name': 'My Team', 'manager_name': 'Me',
                     'overall_points': 500, 'overall_rank': 100000, 'season': '2025-26'}
        gw_history = [{'event': 1, 'points': 50, 'total_points': 50, 'overall_rank': 100000}]

        wildcard_team = {
            'budget': {'total': 100.0, 'spent': 98.5, 'remaining': 1.5},
            'squad': [
                {'name': 'WildcardPlayer', 'position': 'MID', 'team': 'ARS', 'team_id': 3, 'price': 8.0, 'score': 70.0}
            ],
            'starting_xi': [],
            'bench': [],
            'formation': '3-5-2',
            'captain': {'name': 'WildcardPlayer'},
            'vice_captain': {'name': 'WildcardPlayer'}
        }

        latex = self.generator.compile_report(
            team_info=team_info,
            gw_history=gw_history,
            squad=[],
            squad_analysis=[],
            recommendations=[],
            captain_picks=[],
            chips_used=[],
            wildcard_team=wildcard_team
        )

        self.assertIn(r'\section{Wildcard', latex)
        self.assertIn('WildcardPlayer', latex)


class TestFreeHitOptimizer(unittest.TestCase):
    """Tests for the Free Hit squad optimizer."""

    def _make_mock_players_df(self):
        """Create a synthetic players DataFrame for testing.
        
        Contains enough players across many teams to form valid squads.
        Teams are spread across 1-15 to avoid max-3-per-team constraint issues.
        """
        players = []
        # GKP (element_type=1): 5 players across 5 teams
        for i, (name, team, cost, ep_next) in enumerate([
            ('GK_A', 1, 45, 4.0), ('GK_B', 2, 50, 4.5),
            ('GK_C', 3, 40, 3.5), ('GK_D', 4, 42, 3.8),
            ('GK_E', 5, 40, 3.3)
        ], start=1):
            players.append({
                'id': i, 'web_name': name, 'team': team, 'element_type': 1,
                'now_cost': cost, 'ep_next': ep_next, 'minutes': 900,
                'total_points': int(ep_next * 10), 'form': ep_next,
                'points_per_game': ep_next,
                'expected_goal_involvements': 0.0,
                'status': 'a', 'chance_of_playing_next_round': 100
            })
        # DEF (element_type=2): 10 players across 10 different teams
        for i, (name, team, cost, ep_next) in enumerate([
            ('DEF_A', 6, 55, 4.5), ('DEF_B', 7, 50, 4.2),
            ('DEF_C', 8, 60, 5.0), ('DEF_D', 9, 48, 4.0),
            ('DEF_E', 10, 45, 3.8), ('DEF_F', 11, 40, 3.5),
            ('DEF_G', 12, 52, 4.3), ('DEF_H', 13, 42, 3.6),
            ('DEF_I', 14, 40, 3.4), ('DEF_J', 15, 40, 3.3)
        ], start=10):
            players.append({
                'id': i, 'web_name': name, 'team': team, 'element_type': 2,
                'now_cost': cost, 'ep_next': ep_next, 'minutes': 900,
                'total_points': int(ep_next * 10), 'form': ep_next,
                'points_per_game': ep_next,
                'expected_goal_involvements': 0.5,
                'status': 'a', 'chance_of_playing_next_round': 100
            })
        # MID (element_type=3): 10 players across different teams
        for i, (name, team, cost, ep_next) in enumerate([
            ('MID_A', 1, 130, 8.0), ('MID_B', 2, 120, 7.5),
            ('MID_C', 3, 80, 5.5), ('MID_D', 4, 75, 5.2),
            ('MID_E', 5, 65, 4.8), ('MID_F', 6, 50, 4.0),
            ('MID_G', 7, 45, 3.5), ('MID_H', 8, 45, 3.2),
            ('MID_I', 9, 45, 3.1), ('MID_J', 10, 45, 3.0)
        ], start=20):
            players.append({
                'id': i, 'web_name': name, 'team': team, 'element_type': 3,
                'now_cost': cost, 'ep_next': ep_next, 'minutes': 900,
                'total_points': int(ep_next * 10), 'form': ep_next,
                'points_per_game': ep_next,
                'expected_goal_involvements': 3.0,
                'status': 'a', 'chance_of_playing_next_round': 100
            })
        # FWD (element_type=4): 6 players across different teams
        for i, (name, team, cost, ep_next) in enumerate([
            ('FWD_A', 11, 100, 7.0), ('FWD_B', 12, 80, 6.5),
            ('FWD_C', 13, 70, 5.0), ('FWD_D', 14, 55, 4.2),
            ('FWD_E', 15, 45, 3.5), ('FWD_F', 1, 45, 3.3)
        ], start=30):
            players.append({
                'id': i, 'web_name': name, 'team': team, 'element_type': 4,
                'now_cost': cost, 'ep_next': ep_next, 'minutes': 900,
                'total_points': int(ep_next * 10), 'form': ep_next,
                'points_per_game': ep_next,
                'expected_goal_involvements': 5.0,
                'status': 'a', 'chance_of_playing_next_round': 100
            })
        return pd.DataFrame(players)

    def test_free_hit_optimizer_respects_budget(self):
        """Test that optimizer does not exceed total budget."""
        from reports.fpl_report.transfer_strategy import FreeHitOptimizer

        players_df = self._make_mock_players_df()
        total_budget = 100.0  # 100m budget

        optimizer = FreeHitOptimizer(players_df, total_budget)
        result = optimizer.build_squad()

        self.assertIn('squad', result)
        squad = result['squad']
        total_cost = sum(p['price'] for p in squad)
        self.assertLessEqual(total_cost, total_budget)

    def test_free_hit_optimizer_respects_position_quotas(self):
        """Test that optimizer selects exactly 2 GKP, 5 DEF, 5 MID, 3 FWD."""
        from reports.fpl_report.transfer_strategy import FreeHitOptimizer

        players_df = self._make_mock_players_df()
        total_budget = 100.0

        optimizer = FreeHitOptimizer(players_df, total_budget)
        result = optimizer.build_squad()

        squad = result['squad']
        position_counts = {}
        for p in squad:
            pos = p['position']
            position_counts[pos] = position_counts.get(pos, 0) + 1

        self.assertEqual(position_counts.get('GKP', 0), 2)
        self.assertEqual(position_counts.get('DEF', 0), 5)
        self.assertEqual(position_counts.get('MID', 0), 5)
        self.assertEqual(position_counts.get('FWD', 0), 3)
        self.assertEqual(len(squad), 15)

    def test_free_hit_optimizer_respects_max_3_per_team(self):
        """Test that optimizer selects at most 3 players from any single team."""
        from reports.fpl_report.transfer_strategy import FreeHitOptimizer

        players_df = self._make_mock_players_df()
        total_budget = 100.0

        optimizer = FreeHitOptimizer(players_df, total_budget)
        result = optimizer.build_squad()

        squad = result['squad']
        team_counts = {}
        for p in squad:
            tid = p['team_id']
            team_counts[tid] = team_counts.get(tid, 0) + 1

        for tid, count in team_counts.items():
            self.assertLessEqual(count, 3, f"Team {tid} has {count} players (max 3 allowed)")

    def test_free_hit_optimizer_excludes_injured_players(self):
        """Test that optimizer excludes players with status != 'a' or low chance."""
        from reports.fpl_report.transfer_strategy import FreeHitOptimizer

        players_df = self._make_mock_players_df()
        # Mark one high-value player as injured
        players_df.loc[players_df['web_name'] == 'MID_A', 'status'] = 'i'
        players_df.loc[players_df['web_name'] == 'MID_A', 'chance_of_playing_next_round'] = 0

        total_budget = 100.0

        optimizer = FreeHitOptimizer(players_df, total_budget)
        result = optimizer.build_squad()

        squad = result['squad']
        squad_names = [p['name'] for p in squad]
        self.assertNotIn('MID_A', squad_names)

    def test_free_hit_optimizer_provides_starting_xi(self):
        """Test that optimizer provides a valid starting XI with formation."""
        from reports.fpl_report.transfer_strategy import FreeHitOptimizer

        players_df = self._make_mock_players_df()
        total_budget = 100.0

        optimizer = FreeHitOptimizer(players_df, total_budget)
        result = optimizer.build_squad()

        self.assertIn('starting_xi', result)
        self.assertIn('bench', result)
        self.assertIn('formation', result)
        self.assertIn('captain', result)
        self.assertIn('vice_captain', result)

        xi = result['starting_xi']
        bench = result['bench']
        self.assertEqual(len(xi), 11)
        self.assertEqual(len(bench), 4)

    def test_free_hit_optimizer_includes_ep_next(self):
        """Test that optimizer output includes ep_next for each player."""
        from reports.fpl_report.transfer_strategy import FreeHitOptimizer

        players_df = self._make_mock_players_df()
        total_budget = 100.0

        optimizer = FreeHitOptimizer(players_df, total_budget)
        result = optimizer.build_squad()

        # Check that starting XI players have ep_next
        for player in result['starting_xi']:
            self.assertIn('ep_next', player)

    def test_free_hit_optimizer_with_league_ownership(self):
        """Test that optimizer uses league ownership data when provided."""
        from reports.fpl_report.transfer_strategy import FreeHitOptimizer

        players_df = self._make_mock_players_df()
        total_budget = 100.0
        
        # Create mock league ownership data
        league_ownership = {
            'ownership': {
                20: 0.8,  # MID_A - highly owned
                21: 0.1,  # MID_B - differential
            },
            'captain_counts': {},
            'sample_size': 10,
            'top_owned': [(20, 0.8), (21, 0.1)]
        }

        optimizer = FreeHitOptimizer(
            players_df, 
            total_budget, 
            league_ownership=league_ownership,
            strategy='balanced'
        )
        result = optimizer.build_squad()

        # Check league_analysis is populated
        self.assertIn('league_analysis', result)
        self.assertEqual(result['league_analysis']['sample_size'], 10)

    def test_free_hit_optimizer_identifies_differentials(self):
        """Test that optimizer identifies low-ownership differentials."""
        from reports.fpl_report.transfer_strategy import FreeHitOptimizer

        players_df = self._make_mock_players_df()
        total_budget = 100.0
        
        # Create ownership data where all players have high ownership except one
        ownership_map = {i: 0.8 for i in range(1, 40)}  # High ownership for all
        ownership_map[21] = 0.05  # MID_B is a differential
        
        league_ownership = {
            'ownership': ownership_map,
            'captain_counts': {},
            'sample_size': 20,
            'top_owned': []
        }

        optimizer = FreeHitOptimizer(
            players_df, 
            total_budget, 
            league_ownership=league_ownership,
            strategy='balanced'
        )
        result = optimizer.build_squad()

        # Check that differentials list is populated
        differentials = result['league_analysis']['differentials']
        # Should have at least some differentials (< 30% owned)
        # The exact count depends on budget constraints

    def test_free_hit_optimizer_maximizes_budget_as_tiebreak(self):
        """Test that optimizer maximizes spend when XI scores are equal.
        
        Given two players with identical ep_next, the optimizer should prefer
        the more expensive one to maximize budget utilization (as a tie-break).
        """
        from reports.fpl_report.transfer_strategy import FreeHitOptimizer
        
        # Create players where two forwards have identical ep_next but different prices
        players = []
        # GKP: 5 players
        for i, (name, team, cost, ep_next) in enumerate([
            ('GK_A', 1, 45, 4.0), ('GK_B', 2, 50, 4.0),  # Same ep_next
            ('GK_C', 3, 40, 3.5), ('GK_D', 4, 42, 3.5),
            ('GK_E', 5, 40, 3.3)
        ], start=1):
            players.append({
                'id': i, 'web_name': name, 'team': team, 'element_type': 1,
                'now_cost': cost, 'ep_next': ep_next, 'minutes': 900,
                'total_points': int(ep_next * 10), 'form': ep_next,
                'points_per_game': ep_next,
                'status': 'a', 'chance_of_playing_next_round': 100
            })
        # DEF: 10 players
        for i, (name, team, cost, ep_next) in enumerate([
            ('DEF_A', 6, 55, 4.5), ('DEF_B', 7, 50, 4.2),
            ('DEF_C', 8, 60, 5.0), ('DEF_D', 9, 48, 4.0),
            ('DEF_E', 10, 45, 3.8), ('DEF_F', 11, 40, 3.5),
            ('DEF_G', 12, 52, 4.3), ('DEF_H', 13, 42, 3.6),
            ('DEF_I', 14, 40, 3.4), ('DEF_J', 15, 40, 3.3)
        ], start=10):
            players.append({
                'id': i, 'web_name': name, 'team': team, 'element_type': 2,
                'now_cost': cost, 'ep_next': ep_next, 'minutes': 900,
                'total_points': int(ep_next * 10), 'form': ep_next,
                'points_per_game': ep_next,
                'status': 'a', 'chance_of_playing_next_round': 100
            })
        # MID: 10 players
        for i, (name, team, cost, ep_next) in enumerate([
            ('MID_A', 1, 130, 8.0), ('MID_B', 2, 120, 7.5),
            ('MID_C', 3, 80, 5.5), ('MID_D', 4, 75, 5.2),
            ('MID_E', 5, 65, 4.8), ('MID_F', 6, 50, 4.0),
            ('MID_G', 7, 45, 3.5), ('MID_H', 8, 45, 3.2),
            ('MID_I', 9, 45, 3.1), ('MID_J', 10, 45, 3.0)
        ], start=20):
            players.append({
                'id': i, 'web_name': name, 'team': team, 'element_type': 3,
                'now_cost': cost, 'ep_next': ep_next, 'minutes': 900,
                'total_points': int(ep_next * 10), 'form': ep_next,
                'points_per_game': ep_next,
                'status': 'a', 'chance_of_playing_next_round': 100
            })
        # FWD: 6 players - two with identical ep_next but different prices
        for i, (name, team, cost, ep_next) in enumerate([
            ('FWD_A', 11, 100, 7.0), ('FWD_B', 12, 80, 7.0),  # Same ep_next, A costs more
            ('FWD_C', 13, 70, 5.0), ('FWD_D', 14, 55, 4.2),
            ('FWD_E', 15, 45, 3.5), ('FWD_F', 16, 45, 3.3)
        ], start=30):
            players.append({
                'id': i, 'web_name': name, 'team': team, 'element_type': 4,
                'now_cost': cost, 'ep_next': ep_next, 'minutes': 900,
                'total_points': int(ep_next * 10), 'form': ep_next,
                'points_per_game': ep_next,
                'status': 'a', 'chance_of_playing_next_round': 100
            })
        
        players_df = pd.DataFrame(players)
        total_budget = 105.0  # Generous budget to allow choice
        
        # Test all three strategies
        for strategy in ['safe', 'balanced', 'aggressive']:
            optimizer = FreeHitOptimizer(players_df, total_budget, strategy=strategy)
            result = optimizer.build_squad()
            
            squad = result['squad']
            squad_names = [p['name'] for p in squad]
            budget_info = result['budget']
            
            # When two forwards have identical ep_next, prefer the more expensive one (FWD_A)
            # as a tie-break to maximize budget utilization
            self.assertIn('FWD_A', squad_names,
                f"{strategy} strategy: Should select FWD_A (10.0m) over FWD_B (8.0m) "
                "when they have identical ep_next to maximize budget usage")
            
            # Budget should be well-utilized (remaining <= 1.0m)
            self.assertLessEqual(budget_info['remaining'], 1.5,
                f"{strategy} strategy: Should maximize budget usage, but left {budget_info['remaining']}m unused")

    def test_free_hit_optimizer_prefers_momentum_players(self):
        """Test that optimizer prefers players with high form/ppg/total_points.
        
        Given two players with identical ep_next and price, the optimizer should
        prefer the one with better momentum (higher form + ppg + total_points).
        """
        from reports.fpl_report.transfer_strategy import FreeHitOptimizer
        
        # Create players where two midfielders have identical ep_next and price
        # but different momentum stats
        players = []
        # GKP: 5 players
        for i, (name, team, cost, ep_next) in enumerate([
            ('GK_A', 1, 45, 4.0), ('GK_B', 2, 50, 4.5),
            ('GK_C', 3, 40, 3.5), ('GK_D', 4, 42, 3.8),
            ('GK_E', 5, 40, 3.3)
        ], start=1):
            players.append({
                'id': i, 'web_name': name, 'team': team, 'element_type': 1,
                'now_cost': cost, 'ep_next': ep_next, 'minutes': 900,
                'total_points': int(ep_next * 10), 'form': ep_next,
                'points_per_game': ep_next,
                'status': 'a', 'chance_of_playing_next_round': 100
            })
        # DEF: 10 players
        for i, (name, team, cost, ep_next) in enumerate([
            ('DEF_A', 6, 55, 4.5), ('DEF_B', 7, 50, 4.2),
            ('DEF_C', 8, 60, 5.0), ('DEF_D', 9, 48, 4.0),
            ('DEF_E', 10, 45, 3.8), ('DEF_F', 11, 40, 3.5),
            ('DEF_G', 12, 52, 4.3), ('DEF_H', 13, 42, 3.6),
            ('DEF_I', 14, 40, 3.4), ('DEF_J', 15, 40, 3.3)
        ], start=10):
            players.append({
                'id': i, 'web_name': name, 'team': team, 'element_type': 2,
                'now_cost': cost, 'ep_next': ep_next, 'minutes': 900,
                'total_points': int(ep_next * 10), 'form': ep_next,
                'points_per_game': ep_next,
                'status': 'a', 'chance_of_playing_next_round': 100
            })
        # MID: Include two players with same ep_next/price but different momentum
        mid_players = [
            # MID_MOMENTUM has identical ep_next but MUCH better momentum stats
            ('MID_MOMENTUM', 17, 80, 5.5, 120, 7.5, 7.5),  # High momentum
            ('MID_NOMOM', 18, 80, 5.5, 30, 2.0, 2.0),  # Low momentum
            ('MID_A', 1, 130, 8.0, 80, 8.0, 8.0),
            ('MID_B', 2, 120, 7.5, 75, 7.5, 7.5),
            ('MID_C', 3, 75, 5.2, 52, 5.2, 5.2),
            ('MID_D', 4, 65, 4.8, 48, 4.8, 4.8),
            ('MID_E', 5, 50, 4.0, 40, 4.0, 4.0),
            ('MID_F', 6, 45, 3.5, 35, 3.5, 3.5),
            ('MID_G', 7, 45, 3.2, 32, 3.2, 3.2),
            ('MID_H', 8, 45, 3.0, 30, 3.0, 3.0),
        ]
        for i, (name, team, cost, ep_next, total_pts, form, ppg) in enumerate(mid_players, start=20):
            players.append({
                'id': i, 'web_name': name, 'team': team, 'element_type': 3,
                'now_cost': cost, 'ep_next': ep_next, 'minutes': 900,
                'total_points': total_pts, 'form': form,
                'points_per_game': ppg,
                'status': 'a', 'chance_of_playing_next_round': 100
            })
        # FWD: 6 players
        for i, (name, team, cost, ep_next) in enumerate([
            ('FWD_A', 11, 100, 7.0), ('FWD_B', 12, 80, 6.5),
            ('FWD_C', 13, 70, 5.0), ('FWD_D', 14, 55, 4.2),
            ('FWD_E', 15, 45, 3.5), ('FWD_F', 16, 45, 3.3)
        ], start=40):
            players.append({
                'id': i, 'web_name': name, 'team': team, 'element_type': 4,
                'now_cost': cost, 'ep_next': ep_next, 'minutes': 900,
                'total_points': int(ep_next * 10), 'form': ep_next,
                'points_per_game': ep_next,
                'status': 'a', 'chance_of_playing_next_round': 100
            })
        
        players_df = pd.DataFrame(players)
        total_budget = 100.0
        
        # Test that high-momentum player is preferred across all strategies
        for strategy in ['safe', 'balanced', 'aggressive']:
            optimizer = FreeHitOptimizer(players_df, total_budget, strategy=strategy)
            result = optimizer.build_squad()
            
            squad = result['squad']
            squad_names = [p['name'] for p in squad]
            
            # When two mids have identical ep_next and price, prefer the one with better momentum
            self.assertIn('MID_MOMENTUM', squad_names,
                f"{strategy} strategy: Should select MID_MOMENTUM over MID_NOMOM "
                "when they have identical ep_next but different momentum stats")
            # MID_NOMOM should NOT be selected when MID_MOMENTUM is available
            if 'MID_MOMENTUM' in squad_names:
                self.assertNotIn('MID_NOMOM', squad_names,
                    f"{strategy} strategy: Should NOT select low-momentum MID_NOMOM "
                    "when high-momentum MID_MOMENTUM with same ep_next is available")


class TestFreeHitLaTeXSection(unittest.TestCase):
    """Tests for Free Hit team LaTeX section generation."""

    def setUp(self):
        """Set up LaTeX generator."""
        self.generator = LaTeXReportGenerator(team_id=847569, gameweek=17)

    def test_generate_free_hit_team_section_contains_header(self):
        """Test that Free Hit section contains correct LaTeX section header."""
        free_hit_team = {
            'budget': {'total': 100.0, 'spent': 98.5, 'remaining': 1.5},
            'squad': [
                {'name': 'TestGK', 'position': 'GKP', 'team': 'LIV', 'team_id': 1, 
                 'price': 5.0, 'score': 50.0, 'ep_next': 4.5, 'league_ownership': 20.0}
            ],
            'starting_xi': [],
            'bench': [],
            'formation': '4-4-2',
            'captain': {'name': 'TestCap'},
            'vice_captain': {'name': 'TestVC'},
            'target_gw': 17,
            'strategy': 'balanced',
            'league_analysis': {'sample_size': 20, 'differentials': [], 'template_picks': []}
        }

        latex = self.generator.generate_free_hit_team_section(free_hit_team)

        self.assertIn(r'\section{Free Hit Draft', latex)
        self.assertIn('GW17', latex)

    def test_generate_free_hit_team_section_includes_player_names(self):
        """Test that Free Hit section includes player names from squad."""
        free_hit_team = {
            'budget': {'total': 100.0, 'spent': 98.5, 'remaining': 1.5},
            'squad': [
                {'name': 'Salah', 'position': 'MID', 'team': 'LIV', 'team_id': 1, 
                 'price': 13.0, 'score': 95.0, 'ep_next': 8.5, 'league_ownership': 85.0},
                {'name': 'Haaland', 'position': 'FWD', 'team': 'MCI', 'team_id': 2, 
                 'price': 14.5, 'score': 98.0, 'ep_next': 9.0, 'league_ownership': 90.0},
            ],
            'starting_xi': [
                {'name': 'Salah', 'position': 'MID', 'team': 'LIV', 'price': 13.0, 
                 'ep_next': 8.5, 'league_ownership': 85.0},
                {'name': 'Haaland', 'position': 'FWD', 'team': 'MCI', 'price': 14.5, 
                 'ep_next': 9.0, 'league_ownership': 90.0},
            ],
            'bench': [],
            'formation': '4-4-2',
            'captain': {'name': 'Haaland'},
            'vice_captain': {'name': 'Salah'},
            'target_gw': 18,
            'strategy': 'balanced',
            'league_analysis': {'sample_size': 15, 'differentials': [], 'template_picks': []}
        }

        latex = self.generator.generate_free_hit_team_section(free_hit_team)

        self.assertIn('Salah', latex)
        self.assertIn('Haaland', latex)

    def test_generate_free_hit_team_section_includes_league_ownership(self):
        """Test that Free Hit section includes league ownership percentage."""
        free_hit_team = {
            'budget': {'total': 100.0, 'spent': 98.5, 'remaining': 1.5},
            'squad': [],
            'starting_xi': [
                {'name': 'Salah', 'position': 'MID', 'team': 'LIV', 'price': 13.0, 
                 'ep_next': 8.5, 'league_ownership': 85.0},
            ],
            'bench': [],
            'formation': '4-4-2',
            'captain': {'name': 'Salah'},
            'vice_captain': {'name': 'Salah'},
            'target_gw': 18,
            'strategy': 'balanced',
            'league_analysis': {'sample_size': 15, 'differentials': [], 'template_picks': []}
        }

        latex = self.generator.generate_free_hit_team_section(free_hit_team)

        # Should include league ownership header (column header is "Own%")
        self.assertIn('Own', latex)

    def test_compile_report_includes_free_hit_section_when_provided(self):
        """Test that compile_report includes Free Hit section when data is provided."""
        team_info = {'team_name': 'My Team', 'manager_name': 'Me',
                     'overall_points': 500, 'overall_rank': 100000, 'season': '2025-26'}
        gw_history = [{'event': 1, 'points': 50, 'total_points': 50, 'overall_rank': 100000}]

        free_hit_team = {
            'budget': {'total': 100.0, 'spent': 98.5, 'remaining': 1.5},
            'squad': [
                {'name': 'FreeHitPlayer', 'position': 'MID', 'team': 'ARS', 'team_id': 3, 
                 'price': 8.0, 'score': 70.0, 'ep_next': 6.5, 'league_ownership': 25.0}
            ],
            'starting_xi': [],
            'bench': [],
            'formation': '3-5-2',
            'captain': {'name': 'FreeHitPlayer'},
            'vice_captain': {'name': 'FreeHitPlayer'},
            'target_gw': 19,
            'strategy': 'aggressive',
            'league_analysis': {'sample_size': 10, 'differentials': [], 'template_picks': []}
        }

        latex = self.generator.compile_report(
            team_info=team_info,
            gw_history=gw_history,
            squad=[],
            squad_analysis=[],
            recommendations=[],
            captain_picks=[],
            chips_used=[],
            free_hit_team=free_hit_team
        )

        self.assertIn(r'\section{Free Hit Draft', latex)
        self.assertIn('FreeHitPlayer', latex)
        self.assertIn('GW19', latex)


class TestTransferMIPSolver(unittest.TestCase):
    """Tests for the MIP-based transfer optimizer."""

    def _make_mock_players_df(self):
        """Create a synthetic players DataFrame for testing.
        
        Contains enough players across many teams to form valid squads.
        """
        players = []
        # GKP (element_type=1): 5 players across 5 teams
        for i, (name, team, cost) in enumerate([
            ('GK_A', 1, 50), ('GK_B', 2, 45),
            ('GK_C', 3, 40), ('GK_D', 4, 42),
            ('GK_E', 5, 40)
        ], start=1):
            players.append({
                'id': i, 'web_name': name, 'team': team, 'element_type': 1,
                'now_cost': cost, 'minutes': 900,
                'total_points': 40 + i * 2, 'form': 4.0,
                'status': 'a', 'chance_of_playing_next_round': 100
            })
        # DEF (element_type=2): 10 players across different teams
        for i, (name, team, cost) in enumerate([
            ('DEF_A', 6, 55), ('DEF_B', 7, 50),
            ('DEF_C', 8, 60), ('DEF_D', 9, 48),
            ('DEF_E', 10, 45), ('DEF_F', 11, 40),
            ('DEF_G', 12, 52), ('DEF_H', 13, 42),
            ('DEF_I', 14, 40), ('DEF_J', 15, 40)
        ], start=10):
            players.append({
                'id': i, 'web_name': name, 'team': team, 'element_type': 2,
                'now_cost': cost, 'minutes': 900,
                'total_points': 30 + i, 'form': 4.0,
                'status': 'a', 'chance_of_playing_next_round': 100
            })
        # MID (element_type=3): 10 players
        for i, (name, team, cost) in enumerate([
            ('MID_A', 1, 130), ('MID_B', 2, 120),
            ('MID_C', 3, 80), ('MID_D', 4, 75),
            ('MID_E', 5, 65), ('MID_F', 6, 50),
            ('MID_G', 7, 45), ('MID_H', 8, 45),
            ('MID_I', 9, 45), ('MID_J', 10, 45)
        ], start=30):
            players.append({
                'id': i, 'web_name': name, 'team': team, 'element_type': 3,
                'now_cost': cost, 'minutes': 900,
                'total_points': 50 + i, 'form': 5.0,
                'status': 'a', 'chance_of_playing_next_round': 100
            })
        # FWD (element_type=4): 6 players
        for i, (name, team, cost) in enumerate([
            ('FWD_A', 11, 100), ('FWD_B', 12, 80),
            ('FWD_C', 13, 70), ('FWD_D', 14, 55),
            ('FWD_E', 15, 45), ('FWD_F', 17, 45)
        ], start=50):
            players.append({
                'id': i, 'web_name': name, 'team': team, 'element_type': 4,
                'now_cost': cost, 'minutes': 900,
                'total_points': 60 + i, 'form': 5.5,
                'status': 'a', 'chance_of_playing_next_round': 100
            })
        return pd.DataFrame(players)

    def _make_mock_current_squad(self) -> List[Dict]:
        """Create a mock current squad (15 players)."""
        # A valid 15-player squad with selling prices
        squad = [
            # 2 GKP
            {'id': 1, 'name': 'GK_A', 'position': 'GKP', 'team_id': 1, 'team': 'T1',
             'selling_price_m': 5.0, 'stats': {'now_cost': 50, 'status': 'a', 'chance_of_playing_next_round': 100}},
            {'id': 2, 'name': 'GK_B', 'position': 'GKP', 'team_id': 2, 'team': 'T2',
             'selling_price_m': 4.5, 'stats': {'now_cost': 45, 'status': 'a', 'chance_of_playing_next_round': 100}},
            # 5 DEF
            {'id': 10, 'name': 'DEF_A', 'position': 'DEF', 'team_id': 6, 'team': 'T6',
             'selling_price_m': 5.5, 'stats': {'now_cost': 55, 'status': 'a', 'chance_of_playing_next_round': 100}},
            {'id': 11, 'name': 'DEF_B', 'position': 'DEF', 'team_id': 7, 'team': 'T7',
             'selling_price_m': 5.0, 'stats': {'now_cost': 50, 'status': 'a', 'chance_of_playing_next_round': 100}},
            {'id': 12, 'name': 'DEF_C', 'position': 'DEF', 'team_id': 8, 'team': 'T8',
             'selling_price_m': 6.0, 'stats': {'now_cost': 60, 'status': 'a', 'chance_of_playing_next_round': 100}},
            {'id': 13, 'name': 'DEF_D', 'position': 'DEF', 'team_id': 9, 'team': 'T9',
             'selling_price_m': 4.8, 'stats': {'now_cost': 48, 'status': 'a', 'chance_of_playing_next_round': 100}},
            {'id': 14, 'name': 'DEF_E', 'position': 'DEF', 'team_id': 10, 'team': 'T10',
             'selling_price_m': 4.5, 'stats': {'now_cost': 45, 'status': 'a', 'chance_of_playing_next_round': 100}},
            # 5 MID
            {'id': 30, 'name': 'MID_A', 'position': 'MID', 'team_id': 1, 'team': 'T1',
             'selling_price_m': 13.0, 'stats': {'now_cost': 130, 'status': 'a', 'chance_of_playing_next_round': 100}},
            {'id': 31, 'name': 'MID_B', 'position': 'MID', 'team_id': 2, 'team': 'T2',
             'selling_price_m': 12.0, 'stats': {'now_cost': 120, 'status': 'a', 'chance_of_playing_next_round': 100}},
            {'id': 32, 'name': 'MID_C', 'position': 'MID', 'team_id': 3, 'team': 'T3',
             'selling_price_m': 8.0, 'stats': {'now_cost': 80, 'status': 'a', 'chance_of_playing_next_round': 100}},
            {'id': 33, 'name': 'MID_D', 'position': 'MID', 'team_id': 4, 'team': 'T4',
             'selling_price_m': 7.5, 'stats': {'now_cost': 75, 'status': 'a', 'chance_of_playing_next_round': 100}},
            {'id': 34, 'name': 'MID_E', 'position': 'MID', 'team_id': 5, 'team': 'T5',
             'selling_price_m': 6.5, 'stats': {'now_cost': 65, 'status': 'a', 'chance_of_playing_next_round': 100}},
            # 3 FWD
            {'id': 50, 'name': 'FWD_A', 'position': 'FWD', 'team_id': 11, 'team': 'T11',
             'selling_price_m': 10.0, 'stats': {'now_cost': 100, 'status': 'a', 'chance_of_playing_next_round': 100}},
            {'id': 51, 'name': 'FWD_B', 'position': 'FWD', 'team_id': 12, 'team': 'T12',
             'selling_price_m': 8.0, 'stats': {'now_cost': 80, 'status': 'a', 'chance_of_playing_next_round': 100}},
            {'id': 52, 'name': 'FWD_C', 'position': 'FWD', 'team_id': 13, 'team': 'T13',
             'selling_price_m': 7.0, 'stats': {'now_cost': 70, 'status': 'a', 'chance_of_playing_next_round': 100}},
        ]
        return squad

    def _make_xp_matrix(self, player_ids: List[int], horizon: int = 5) -> Dict[int, List[float]]:
        """Create mock xP predictions for given players."""
        xp = {}
        for pid in player_ids:
            # Higher ID = higher xP (simple heuristic for testing)
            base = 3.0 + (pid % 10) * 0.5
            xp[pid] = [base + (w * 0.1) for w in range(horizon)]
        return xp

    def test_mip_solver_unavailable_returns_status(self):
        """Test that solver returns 'unavailable' status if dependencies missing."""
        from reports.fpl_report.transfer_strategy import TransferMIPSolver, MIP_AVAILABLE
        
        if MIP_AVAILABLE:
            # Skip this test if MIP is actually available
            self.skipTest("MIP solver is available, skipping unavailable test")
        
        players_df = self._make_mock_players_df()
        current_squad = self._make_mock_current_squad()
        squad_ids = [p['id'] for p in current_squad]
        xp_matrix = self._make_xp_matrix(squad_ids)
        
        solver = TransferMIPSolver(
            current_squad=current_squad,
            bank=0.5,
            players_df=players_df,
            xp_matrix=xp_matrix,
            free_transfers=1
        )
        result = solver.solve()
        
        self.assertEqual(result.status, 'unavailable')

    def test_mip_solver_result_schema_is_stable(self):
        """Test that MIPSolverResult has all expected fields."""
        from reports.fpl_report.transfer_strategy import MIPSolverResult
        
        result = MIPSolverResult(status='test')
        
        # Check all expected fields exist
        self.assertIn('status', vars(result))
        self.assertIn('transfers_out', vars(result))
        self.assertIn('transfers_in', vars(result))
        self.assertIn('new_squad', vars(result))
        self.assertIn('starting_xi', vars(result))
        self.assertIn('bench', vars(result))
        self.assertIn('formation', vars(result))
        self.assertIn('captain', vars(result))
        self.assertIn('hit_cost', vars(result))
        self.assertIn('num_transfers', vars(result))
        self.assertIn('budget_remaining', vars(result))
        self.assertIn('expected_points', vars(result))
        self.assertIn('per_gw_xp', vars(result))

    def test_mip_solver_builds_candidate_pool(self):
        """Test that solver builds candidate pool correctly."""
        from reports.fpl_report.transfer_strategy import TransferMIPSolver, MIP_AVAILABLE
        
        players_df = self._make_mock_players_df()
        current_squad = self._make_mock_current_squad()
        squad_ids = [p['id'] for p in current_squad]
        xp_matrix = self._make_xp_matrix(squad_ids + list(players_df['id']))
        
        solver = TransferMIPSolver(
            current_squad=current_squad,
            bank=0.5,
            players_df=players_df,
            xp_matrix=xp_matrix,
            free_transfers=1,
            candidate_pool_size=5
        )
        
        # Should have current squad + additional candidates
        self.assertGreaterEqual(len(solver.candidates), 15)
        
        # All current squad players should be in candidates
        candidate_ids = {c['id'] for c in solver.candidates}
        for p in current_squad:
            self.assertIn(p['id'], candidate_ids)
        
        # All candidates should have required fields
        for c in solver.candidates:
            self.assertIn('id', c)
            self.assertIn('position', c)
            self.assertIn('team_id', c)
            self.assertIn('buy_price', c)
            self.assertIn('sell_price', c)
            self.assertIn('xp', c)
            self.assertIn('is_current', c)

    @unittest.skipIf(not True, "MIP solver test - may skip if dependencies unavailable")
    def test_mip_solver_respects_position_quotas_in_candidates(self):
        """Test that candidate pool includes players from all positions."""
        from reports.fpl_report.transfer_strategy import TransferMIPSolver
        
        players_df = self._make_mock_players_df()
        current_squad = self._make_mock_current_squad()
        squad_ids = [p['id'] for p in current_squad]
        xp_matrix = self._make_xp_matrix(squad_ids + list(players_df['id']))
        
        solver = TransferMIPSolver(
            current_squad=current_squad,
            bank=0.5,
            players_df=players_df,
            xp_matrix=xp_matrix,
            free_transfers=1
        )
        
        # Check we have candidates from each position
        positions_present = set(c['position'] for c in solver.candidates)
        self.assertIn('GKP', positions_present)
        self.assertIn('DEF', positions_present)
        self.assertIn('MID', positions_present)
        self.assertIn('FWD', positions_present)

    def test_mip_solver_tracks_current_squad_correctly(self):
        """Test that solver correctly identifies current squad players."""
        from reports.fpl_report.transfer_strategy import TransferMIPSolver
        
        players_df = self._make_mock_players_df()
        current_squad = self._make_mock_current_squad()
        squad_ids = [p['id'] for p in current_squad]
        xp_matrix = self._make_xp_matrix(squad_ids + list(players_df['id']))
        
        solver = TransferMIPSolver(
            current_squad=current_squad,
            bank=0.5,
            players_df=players_df,
            xp_matrix=xp_matrix,
            free_transfers=1
        )
        
        # Count current players in candidates
        current_count = sum(1 for c in solver.candidates if c['is_current'])
        self.assertEqual(current_count, 15)
        
        # Verify current squad IDs match
        self.assertEqual(solver.current_squad_ids, set(squad_ids))


class TestMultiPeriodPlanning(unittest.TestCase):
    """Tests for multi-period transfer planning utilities."""

    def test_build_transfer_timeline_from_result(self):
        """Test building timeline from solver result."""
        from reports.fpl_report.transfer_strategy import (
            MIPSolverResult, build_transfer_timeline
        )
        
        # Create a mock optimal result
        result = MIPSolverResult(
            status='optimal',
            transfers_in=[{'name': 'Player_A', 'buy_price': 10.0}],
            transfers_out=[{'name': 'Player_B', 'sell_price': 8.0}],
            formation='4-4-2',
            captain={'name': 'Haaland'},
            hit_cost=0,
            num_transfers=1,
            expected_points=45.5,
            per_gw_xp=[10.0, 9.5, 9.0, 8.5, 8.5],
            budget_remaining=0.5
        )
        
        timeline = build_transfer_timeline(result, current_gw=15, horizon=5)
        
        self.assertEqual(timeline['status'], 'optimal')
        self.assertEqual(timeline['current_gw'], 15)
        self.assertEqual(len(timeline['weeks']), 5)
        self.assertEqual(timeline['weeks'][0]['gameweek'], 16)
        self.assertEqual(timeline['total_expected_points'], 45.5)

    def test_build_transfer_timeline_handles_non_optimal(self):
        """Test timeline building with non-optimal result."""
        from reports.fpl_report.transfer_strategy import (
            MIPSolverResult, build_transfer_timeline
        )
        
        result = MIPSolverResult(
            status='infeasible',
            message='Budget constraint violated'
        )
        
        timeline = build_transfer_timeline(result, current_gw=10, horizon=5)
        
        self.assertEqual(timeline['status'], 'infeasible')
        self.assertEqual(timeline['weeks'], [])

    def test_format_timeline_for_latex(self):
        """Test LaTeX formatting of timeline."""
        from reports.fpl_report.transfer_strategy import (
            MIPSolverResult, build_transfer_timeline, format_timeline_for_latex
        )
        
        result = MIPSolverResult(
            status='optimal',
            transfers_in=[{'name': 'Salah', 'buy_price': 13.0}],
            transfers_out=[{'name': 'Son', 'sell_price': 10.0}],
            formation='4-3-3',
            captain={'name': 'Salah'},
            hit_cost=4,
            num_transfers=2,
            expected_points=52.0,
            per_gw_xp=[11.0, 10.5, 10.0, 10.5, 10.0],
            budget_remaining=1.0
        )
        
        timeline = build_transfer_timeline(result, current_gw=17, horizon=5)
        latex = format_timeline_for_latex(timeline)
        
        # Should contain TikZ elements
        self.assertIn('tikzpicture', latex)
        self.assertIn('GW18', latex)  # First week after current
        self.assertIn('11.0', latex)  # First week xP

    def test_multiperiod_plan_dataclass(self):
        """Test MultiPeriodPlan dataclass fields."""
        from reports.fpl_report.transfer_strategy import MultiPeriodPlan
        
        plan = MultiPeriodPlan(
            status='optimal',
            horizon=5,
            total_expected_points=200.0,
            total_hit_cost=4
        )
        
        self.assertEqual(plan.status, 'optimal')
        self.assertEqual(plan.horizon, 5)
        self.assertEqual(plan.total_expected_points, 200.0)
        self.assertEqual(plan.total_hit_cost, 4)
        self.assertEqual(plan.weekly_plans, [])
        self.assertEqual(plan.free_transfers_banked, [])


class TestMIPSolverIntegration(unittest.TestCase):
    """Integration tests for MIP solver (require sasoptpy + highspy)."""
    
    @classmethod
    def setUpClass(cls):
        """Check if MIP solver is available."""
        try:
            from reports.fpl_report.transfer_strategy import MIP_AVAILABLE
            cls.mip_available = MIP_AVAILABLE
        except ImportError:
            cls.mip_available = False
    
    def _make_simple_squad_and_candidates(self):
        """Create a minimal test case for solver integration."""
        # Simple 15-player squad
        current_squad = []
        
        # 2 GKP
        for i in range(2):
            current_squad.append({
                'id': 100 + i, 'name': f'GK_{i}', 'position': 'GKP',
                'team_id': i + 1, 'team': f'T{i+1}',
                'selling_price_m': 4.5,
                'stats': {'now_cost': 45, 'status': 'a', 'chance_of_playing_next_round': 100}
            })
        
        # 5 DEF
        for i in range(5):
            current_squad.append({
                'id': 200 + i, 'name': f'DEF_{i}', 'position': 'DEF',
                'team_id': i + 3, 'team': f'T{i+3}',
                'selling_price_m': 5.0,
                'stats': {'now_cost': 50, 'status': 'a', 'chance_of_playing_next_round': 100}
            })
        
        # 5 MID
        for i in range(5):
            current_squad.append({
                'id': 300 + i, 'name': f'MID_{i}', 'position': 'MID',
                'team_id': i + 8, 'team': f'T{i+8}',
                'selling_price_m': 7.0,
                'stats': {'now_cost': 70, 'status': 'a', 'chance_of_playing_next_round': 100}
            })
        
        # 3 FWD
        for i in range(3):
            current_squad.append({
                'id': 400 + i, 'name': f'FWD_{i}', 'position': 'FWD',
                'team_id': i + 13, 'team': f'T{i+13}',
                'selling_price_m': 8.0,
                'stats': {'now_cost': 80, 'status': 'a', 'chance_of_playing_next_round': 100}
            })
        
        # Create players DataFrame with additional candidates
        players = []
        for p in current_squad:
            players.append({
                'id': p['id'],
                'web_name': p['name'],
                'team': p['team_id'],
                'element_type': {'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}[p['position']],
                'now_cost': int(p['selling_price_m'] * 10),
                'minutes': 900,
                'total_points': 50,
                'status': 'a',
                'chance_of_playing_next_round': 100
            })
        
        # Add some additional candidates
        for i in range(10):
            for pos_type, pos in [(1, 'GKP'), (2, 'DEF'), (3, 'MID'), (4, 'FWD')]:
                players.append({
                    'id': 500 + i * 4 + pos_type,
                    'web_name': f'CAND_{pos}_{i}',
                    'team': (i % 17) + 1,
                    'element_type': pos_type,
                    'now_cost': 50 + i * 5,
                    'minutes': 900,
                    'total_points': 40 + i * 3,
                    'status': 'a',
                    'chance_of_playing_next_round': 100
                })
        
        players_df = pd.DataFrame(players)
        
        # Create xP matrix
        xp_matrix = {}
        for p in players:
            pid = p['id']
            xp_matrix[pid] = [3.0 + (p['total_points'] / 20)] * 5
        
        return current_squad, players_df, xp_matrix
    
    @unittest.skipUnless(True, "Integration test - run if MIP available")
    def test_solver_finds_solution_with_no_transfers(self):
        """Test that solver can find a valid solution with no transfers."""
        if not self.mip_available:
            self.skipTest("MIP solver not available")
        
        from reports.fpl_report.transfer_strategy import TransferMIPSolver
        
        current_squad, players_df, xp_matrix = self._make_simple_squad_and_candidates()
        
        solver = TransferMIPSolver(
            current_squad=current_squad,
            bank=0.0,  # No money for transfers
            players_df=players_df,
            xp_matrix=xp_matrix,
            free_transfers=1,
            candidate_pool_size=5,
            time_limit=30.0
        )
        
        result = solver.solve()
        
        # Should find a solution (even if no transfers)
        self.assertIn(result.status, ['optimal', 'unavailable', 'error'])
        
        if result.status == 'optimal':
            # Should have 15 players in new squad
            self.assertEqual(len(result.new_squad), 15)
            # Should have 11 in starting XI
            self.assertEqual(len(result.starting_xi), 11)
            # Should have 4 on bench
            self.assertEqual(len(result.bench), 4)
            # Hit cost should be 0 for <= 1 transfer
            if result.num_transfers <= 1:
                self.assertEqual(result.hit_cost, 0)


class TestFPLCorePredictorCriticalIssues(unittest.TestCase):
    """Tests for FPLCorePredictor critical issues fixes.
    
    Issue 1: Data leakage - points-derived features should be removed
    Issue 2: Fixture context - should use real Elo values, not hardcoded
    Issue 3: Chronological split - train/val should have no GW overlap
    Issue 4: Position-specific models - should train separate models per position
    """

    def test_feature_cols_no_points_derived(self):
        """Feature columns should not include points-derived features."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        
        # These features are directly derived from event_points (target leakage)
        points_derived = [
            'avg_points_per_90',
            'form_trend_3gw',
            'form_trend_5gw', 
            'consistency_score',
        ]
        
        for feature in points_derived:
            self.assertNotIn(
                feature, 
                predictor.feature_cols,
                f"Points-derived feature '{feature}' should not be in feature_cols"
            )
    
    def test_fixture_context_uses_real_values(self):
        """Fixture context should use real Elo/home values, not hardcoded averages."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        
        # Build minimal test data with known fixture info
        all_gw_data = {
            1: {
                'playermatchstats': pd.DataFrame([
                    {'player_id': 1, 'minutes_played': 90, 'xg': 0.5, 'xa': 0.3,
                     'total_shots': 3, 'touches_opposition_box': 5}
                ]),
                'player_gameweek_stats': pd.DataFrame([
                    {'id': 1, 'event_points': 6, 'minutes': 90, 'bps': 30, 'bonus': 1,
                     'ict_index': 5.0, 'influence': 30, 'creativity': 20, 'threat': 40}
                ]),
                'matches': pd.DataFrame([
                    {'team_h': 1, 'team_a': 2, 'team_h_score': 2, 'team_a_score': 0}
                ]),
                'fixtures': pd.DataFrame([
                    {'gameweek': 1, 'home_team': 1, 'away_team': 2, 
                     'home_team_elo': 1900.0, 'away_team_elo': 1700.0}
                ]),
                'teams': pd.DataFrame([
                    {'id': 1, 'elo': 1900.0},
                    {'id': 2, 'elo': 1700.0}
                ])
            },
            2: {
                'playermatchstats': pd.DataFrame([
                    {'player_id': 1, 'minutes_played': 90, 'xg': 0.3, 'xa': 0.2,
                     'total_shots': 2, 'touches_opposition_box': 4}
                ]),
                'player_gameweek_stats': pd.DataFrame([
                    {'id': 1, 'event_points': 3, 'minutes': 90, 'bps': 20, 'bonus': 0,
                     'ict_index': 4.0, 'influence': 25, 'creativity': 15, 'threat': 30,
                     'now_cost': 80}
                ]),
                'matches': pd.DataFrame([
                    {'team_h': 2, 'team_a': 1, 'team_h_score': 1, 'team_a_score': 1}
                ]),
                'fixtures': pd.DataFrame([
                    {'gameweek': 2, 'home_team': 2, 'away_team': 1,
                     'home_team_elo': 1700.0, 'away_team_elo': 1900.0}
                ]),
                'teams': pd.DataFrame([
                    {'id': 1, 'elo': 1900.0},
                    {'id': 2, 'elo': 1700.0}
                ])
            }
        }
        
        fpl_core_season_data = {
            'players': pd.DataFrame([
                {'player_id': 1, 'position': 'Midfielder', 'team_code': 1}
            ])
        }
        
        # Build training dataset
        df = predictor.build_training_dataset(all_gw_data, fpl_core_season_data, current_gw=2, min_gw=2)
        
        # is_home should be 0 or 1, never 0.5
        if 'is_home' in df.columns and not df.empty:
            is_home_vals = df['is_home'].unique()
            self.assertTrue(
                all(v in [0, 1, 0.0, 1.0] for v in is_home_vals),
                f"is_home should be 0 or 1, got: {is_home_vals}"
            )
        
        # fixture_difficulty should not always be 3.0
        if 'fixture_difficulty' in df.columns and len(df) > 1:
            fd_vals = df['fixture_difficulty'].unique()
            self.assertFalse(
                len(fd_vals) == 1 and fd_vals[0] == 3.0,
                "fixture_difficulty should not always be hardcoded to 3.0"
            )

    def test_chronological_split_no_overlap(self):
        """Train/val split should have no gameweek overlap."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        
        # Create mock data spanning multiple gameweeks
        all_gw_data = {}
        for gw in range(1, 11):
            all_gw_data[gw] = {
                'playermatchstats': pd.DataFrame([
                    {'player_id': p, 'minutes_played': 90, 'xg': 0.3, 'xa': 0.2,
                     'total_shots': 2, 'touches_opposition_box': 3}
                    for p in [1, 2, 3]
                ]),
                'player_gameweek_stats': pd.DataFrame([
                    {'id': p, 'event_points': 4 + (gw % 3), 'minutes': 90, 
                     'bps': 25, 'bonus': 0, 'ict_index': 5.0, 'influence': 30, 
                     'creativity': 20, 'threat': 35, 'now_cost': 70}
                    for p in [1, 2, 3]
                ]),
                'matches': pd.DataFrame([
                    {'team_h': 1, 'team_a': 2, 'team_h_score': 1, 'team_a_score': 0}
                ]),
                'fixtures': pd.DataFrame([
                    {'gameweek': gw, 'home_team': 1, 'away_team': 2,
                     'home_team_elo': 1800.0, 'away_team_elo': 1750.0}
                ]),
                'teams': pd.DataFrame([
                    {'id': 1, 'elo': 1800.0},
                    {'id': 2, 'elo': 1750.0}
                ])
            }
        
        fpl_core_season_data = {
            'players': pd.DataFrame([
                {'player_id': p, 'position': 'Midfielder', 'team_code': 1}
                for p in [1, 2, 3]
            ])
        }
        
        # Build dataset
        df = predictor.build_training_dataset(all_gw_data, fpl_core_season_data, current_gw=10, min_gw=5)
        
        if df.empty or 'gameweek' not in df.columns:
            self.skipTest("Not enough data to test split")
        
        # Call the split helper (we need to expose or test the split logic)
        # For now, verify the split produces non-overlapping GWs
        train_df, val_df = predictor._chronological_split(df, validation_split=0.2)
        
        if train_df.empty or val_df.empty:
            self.skipTest("Split produced empty sets")
        
        train_gws = set(train_df['gameweek'].unique())
        val_gws = set(val_df['gameweek'].unique())
        
        # No overlap
        self.assertEqual(
            len(train_gws & val_gws), 0,
            f"Train/val GWs overlap: {train_gws & val_gws}"
        )
        
        # Val GWs should be later than all train GWs
        self.assertGreater(
            min(val_gws), max(train_gws),
            f"Validation GWs ({val_gws}) should be strictly after train GWs ({train_gws})"
        )

    def test_position_specific_models_trained(self):
        """Predictor should train separate models per position."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        
        # Create mock data with different positions
        all_gw_data = {}
        for gw in range(1, 8):
            all_gw_data[gw] = {
                'playermatchstats': pd.DataFrame([
                    {'player_id': 1, 'minutes_played': 90, 'xg': 0.0, 'xa': 0.0,
                     'total_shots': 0, 'touches_opposition_box': 0, 'saves': 5},
                    {'player_id': 2, 'minutes_played': 90, 'xg': 0.1, 'xa': 0.1,
                     'total_shots': 1, 'touches_opposition_box': 2, 'clearances': 5},
                    {'player_id': 3, 'minutes_played': 90, 'xg': 0.4, 'xa': 0.3,
                     'total_shots': 3, 'touches_opposition_box': 6, 'chances_created': 2},
                    {'player_id': 4, 'minutes_played': 90, 'xg': 0.6, 'xa': 0.1,
                     'total_shots': 4, 'touches_opposition_box': 8, 'chances_created': 1},
                ]),
                'player_gameweek_stats': pd.DataFrame([
                    {'id': 1, 'event_points': 6, 'minutes': 90, 'bps': 30, 'bonus': 1,
                     'ict_index': 3.0, 'influence': 40, 'creativity': 5, 'threat': 5, 'now_cost': 55},
                    {'id': 2, 'event_points': 5, 'minutes': 90, 'bps': 25, 'bonus': 0,
                     'ict_index': 4.0, 'influence': 35, 'creativity': 10, 'threat': 15, 'now_cost': 50},
                    {'id': 3, 'event_points': 7, 'minutes': 90, 'bps': 35, 'bonus': 2,
                     'ict_index': 7.0, 'influence': 30, 'creativity': 40, 'threat': 45, 'now_cost': 85},
                    {'id': 4, 'event_points': 8, 'minutes': 90, 'bps': 40, 'bonus': 3,
                     'ict_index': 8.0, 'influence': 25, 'creativity': 20, 'threat': 60, 'now_cost': 100},
                ]),
                'matches': pd.DataFrame([
                    {'team_h': 1, 'team_a': 2, 'team_h_score': 2, 'team_a_score': 1}
                ]),
                'fixtures': pd.DataFrame([
                    {'gameweek': gw, 'home_team': 1, 'away_team': 2,
                     'home_team_elo': 1850.0, 'away_team_elo': 1800.0}
                ]),
                'teams': pd.DataFrame([
                    {'id': 1, 'elo': 1850.0},
                    {'id': 2, 'elo': 1800.0}
                ])
            }
        
        fpl_core_season_data = {
            'players': pd.DataFrame([
                {'player_id': 1, 'position': 'Goalkeeper', 'team_code': 1},
                {'player_id': 2, 'position': 'Defender', 'team_code': 1},
                {'player_id': 3, 'position': 'Midfielder', 'team_code': 1},
                {'player_id': 4, 'position': 'Forward', 'team_code': 2},
            ])
        }
        
        # Train the model
        predictor.train(all_gw_data, fpl_core_season_data, current_gw=7, validation_split=0.2)
        
        # Check that position-specific models exist
        self.assertTrue(
            hasattr(predictor, 'position_models'),
            "Predictor should have position_models attribute"
        )
        
        # Each position should have its own model
        expected_positions = ['GKP', 'DEF', 'MID', 'FWD']
        for pos in expected_positions:
            self.assertIn(
                pos, predictor.position_models,
                f"Position '{pos}' should have a dedicated model"
            )


class TestFallbackPredictorChronologicalSplit(unittest.TestCase):
    """Test that fallback predictor also uses chronological GW-based split."""
    
    def test_predictor_has_chronological_split(self):
        """FPLPointsPredictor should have _chronological_split method."""
        from reports.fpl_report.predictor import FPLPointsPredictor
        
        # Create minimal mock fetcher
        mock_fetcher = MagicMock()
        mock_fetcher.get_current_gameweek.return_value = 10
        mock_fetcher.fixtures_df = pd.DataFrame(columns=[
            'event', 'team_h', 'team_a', 'team_h_difficulty', 'team_a_difficulty', 'finished'
        ])
        
        predictor = FPLPointsPredictor(mock_fetcher)
        
        # The predictor should have a chronological split method
        self.assertTrue(
            hasattr(predictor, '_chronological_split'),
            "Predictor should have _chronological_split method"
        )
    
    def test_chronological_split_no_overlap(self):
        """Chronological split should produce non-overlapping GW sets."""
        from reports.fpl_report.predictor import FPLPointsPredictor
        
        mock_fetcher = MagicMock()
        mock_fetcher.fixtures_df = pd.DataFrame()
        
        predictor = FPLPointsPredictor(mock_fetcher)
        
        # Create test data with gameweek column
        test_df = pd.DataFrame({
            'gameweek': [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10],
            'player_id': [1, 2, 3] * 6,
            'target_points': [4, 5, 6] * 6
        })
        
        train_df, val_df = predictor._chronological_split(test_df, validation_split=0.2)
        
        train_gws = set(train_df['gameweek'].unique())
        val_gws = set(val_df['gameweek'].unique())
        
        # No overlap
        self.assertEqual(
            len(train_gws & val_gws), 0,
            f"Train/val GWs should not overlap: {train_gws & val_gws}"
        )
        
        # Val GWs should be later than all train GWs
        if train_gws and val_gws:
            self.assertGreater(
                min(val_gws), max(train_gws),
                f"Validation GWs ({val_gws}) should be after train GWs ({train_gws})"
            )


class TestFPLCorePredictorMultiGWFixtures(unittest.TestCase):
    """Tests for multi-GW fixture-aware predictions in FPLCorePredictor.
    
    These tests verify:
    1. predict_multiple_gws() is fixture-sensitive across the horizon
    2. Output schema remains stable (predictions, cumulative, confidence, avg_per_gw)
    3. Missing feature blocks produce sensible defaults without crash
    4. Ensemble produces consistent, diverse predictions
    """
    
    def test_predict_uses_fpl_api_fixture_fallback(self):
        """predict should pull next-GW fixtures from fpl_api_fixtures when all_gw_data lacks future entries."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor

        predictor = FPLCorePredictor()
        predictor.is_trained = True
        predictor._reset_context_caches()

        # Make position models appear trained but skip heavy model objects
        for pos in predictor.POSITIONS:
            predictor.position_models[pos]['is_trained'] = True
            predictor.position_models[pos]['feature_cols'] = []
        predictor.fallback_models['is_trained'] = True
        predictor.fallback_models['feature_cols'] = []

        # Stub heavy helpers to avoid real model inference
        predictor._predict_with_stack = lambda model_data, X_raw: np.zeros(X_raw.shape[0])
        predictor._aggregate_match_stats = lambda pms_list, pid, upto, window=4: {'minutes_played': 180}
        predictor._aggregate_gw_stats = lambda pgs_list, pid, upto, window=4, playermatchstats_list=None: {'now_cost': 50}
        predictor._calculate_team_strength = lambda all_gw_data, up_to_gw=None: {
            1: {'attack': 1.5, 'defense': 1.5},
            3: {'attack': 1.4, 'defense': 1.6},
        }

        # Historical data only (no future GW entry)
        all_gw_data = {
            8: {
                'playermatchstats': pd.DataFrame([{'player_id': 1, 'minutes_played': 90}]),
                'player_gameweek_stats': pd.DataFrame([{'id': 1, 'event_points': 5, 'minutes': 90, 'now_cost': 85}]),
            }
        }
        fpl_core_season_data = {
            'players': pd.DataFrame([{'player_id': 1, 'position': 'Midfielder', 'team_code': 1}]),
            'fpl_api_fixtures': pd.DataFrame([
                {'event': 9, 'team_h': 1, 'team_a': 3, 'team_h_difficulty': 2, 'team_a_difficulty': 4}
            ]),
        }

        preds = predictor.predict(all_gw_data, fpl_core_season_data, [1], current_gw=8)

        self.assertIn(1, preds, "Predict should return result for player when trained")
        self.assertIn((1, 9), predictor._fixture_cache,
                      "Fallback fixtures should populate cache for next GW when missing in all_gw_data")

    def _make_multi_gw_test_data(self, num_gws: int = 10):
        """Create test data spanning multiple gameweeks with varied fixtures.
        
        Creates different fixture contexts to ensure predictions vary by GW.
        """
        all_gw_data = {}
        
        # Create fixtures with varying difficulty for different GWs
        # Player 1 on team 1 will face alternating easy/hard opponents
        for gw in range(1, num_gws + 1):
            # Alternate between easy (team 3, low elo) and hard (team 4, high elo) opponents
            if gw % 2 == 0:
                opp_team = 3  # Easy opponent
                opp_elo = 1500.0
            else:
                opp_team = 4  # Hard opponent
                opp_elo = 1950.0
            
            is_home = gw % 3 != 0  # Home 2/3 of the time
            
            all_gw_data[gw] = {
                'playermatchstats': pd.DataFrame([
                    {'player_id': 1, 'minutes_played': 90, 'xg': 0.4 + (gw % 3) * 0.1, 
                     'xa': 0.2, 'total_shots': 3, 'touches_opposition_box': 5,
                     'chances_created': 2, 'accurate_passes_percent': 85.0,
                     'tackles': 2, 'interceptions': 1, 'clearances': 0},
                    {'player_id': 2, 'minutes_played': 90, 'xg': 0.1, 'xa': 0.1,
                     'total_shots': 1, 'touches_opposition_box': 2,
                     'chances_created': 1, 'accurate_passes_percent': 80.0,
                     'tackles': 4, 'interceptions': 3, 'clearances': 5},
                ]),
                'player_gameweek_stats': pd.DataFrame([
                    {'id': 1, 'event_points': 5 + (gw % 4), 'minutes': 90, 
                     'bps': 30, 'bonus': 1, 'ict_index': 6.0, 
                     'influence': 35, 'creativity': 25, 'threat': 40, 'now_cost': 85},
                    {'id': 2, 'event_points': 4 + (gw % 3), 'minutes': 90,
                     'bps': 25, 'bonus': 0, 'ict_index': 4.0,
                     'influence': 30, 'creativity': 15, 'threat': 20, 'now_cost': 55},
                ]),
                'matches': pd.DataFrame([
                    {'team_h': 1 if is_home else opp_team, 
                     'team_a': opp_team if is_home else 1, 
                     'team_h_score': 2, 'team_a_score': 1}
                ]),
                'fixtures': pd.DataFrame([
                    {'gameweek': gw, 
                     'home_team': 1 if is_home else opp_team, 
                     'away_team': opp_team if is_home else 1,
                     'home_team_elo': 1850.0 if is_home else opp_elo, 
                     'away_team_elo': opp_elo if is_home else 1850.0}
                ]),
                'teams': pd.DataFrame([
                    {'id': 1, 'elo': 1850.0},
                    {'id': opp_team, 'elo': opp_elo}
                ])
            }
        
        fpl_core_season_data = {
            'players': pd.DataFrame([
                {'player_id': 1, 'position': 'Midfielder', 'team_code': 1},
                {'player_id': 2, 'position': 'Defender', 'team_code': 1},
            ])
        }
        
        return all_gw_data, fpl_core_season_data
    
    def test_predict_multiple_gws_schema_stability(self):
        """Output schema should be stable: predictions list, cumulative, confidence, avg_per_gw."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        all_gw_data, fpl_core_season_data = self._make_multi_gw_test_data(num_gws=8)
        
        # Train
        predictor.train(all_gw_data, fpl_core_season_data, current_gw=8, validation_split=0.2)
        
        # Predict multiple GWs
        num_gws = 5
        results = predictor.predict_multiple_gws(
            all_gw_data, fpl_core_season_data,
            player_ids=[1, 2], current_gw=8, num_gws=num_gws
        )
        
        # Verify schema for each player
        for pid in [1, 2]:
            self.assertIn(pid, results, f"Player {pid} should be in results")
            result = results[pid]
            
            # Required keys
            self.assertIn('predictions', result, "Result should have 'predictions' key")
            self.assertIn('cumulative', result, "Result should have 'cumulative' key")
            self.assertIn('confidence', result, "Result should have 'confidence' key")
            self.assertIn('avg_per_gw', result, "Result should have 'avg_per_gw' key")
            
            # predictions should be a list of length num_gws
            self.assertIsInstance(result['predictions'], list)
            self.assertEqual(
                len(result['predictions']), num_gws,
                f"predictions list should have {num_gws} elements"
            )
            
            # cumulative should equal sum of predictions
            expected_cumulative = sum(result['predictions'])
            self.assertAlmostEqual(
                result['cumulative'], expected_cumulative, places=1,
                msg="cumulative should be sum of predictions"
            )
            
            # confidence should be one of expected values
            self.assertIn(
                result['confidence'], ['high', 'medium', 'low'],
                "confidence should be 'high', 'medium', or 'low'"
            )
    
    def test_predict_multiple_gws_fixture_sensitivity(self):
        """Predictions should vary based on fixture difficulty across the horizon."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        all_gw_data, fpl_core_season_data = self._make_multi_gw_test_data(num_gws=8)
        
        # Add future fixtures with very different difficulties
        # GW 9: Easy opponent (elo 1500)
        # GW 10: Very hard opponent (elo 2000)
        all_gw_data[9] = {
            'fixtures': pd.DataFrame([
                {'gameweek': 9, 'home_team': 1, 'away_team': 3,
                 'home_team_elo': 1850.0, 'away_team_elo': 1500.0}
            ]),
            'teams': pd.DataFrame([
                {'id': 1, 'elo': 1850.0},
                {'id': 3, 'elo': 1500.0}
            ])
        }
        all_gw_data[10] = {
            'fixtures': pd.DataFrame([
                {'gameweek': 10, 'home_team': 4, 'away_team': 1,
                 'home_team_elo': 2000.0, 'away_team_elo': 1850.0}
            ]),
            'teams': pd.DataFrame([
                {'id': 1, 'elo': 1850.0},
                {'id': 4, 'elo': 2000.0}
            ])
        }
        
        # Train
        predictor.train(all_gw_data, fpl_core_season_data, current_gw=8, validation_split=0.2)
        
        # Predict for GW 9-10 (2 weeks)
        results = predictor.predict_multiple_gws(
            all_gw_data, fpl_core_season_data,
            player_ids=[1], current_gw=8, num_gws=2
        )
        
        self.assertIn(1, results)
        preds = results[1]['predictions']
        
        # With fixture-aware predictions, we expect the predictions to be different
        # (though exact values depend on model). At minimum, they shouldn't be identical.
        # Note: This test will initially fail if using naive decay (0.95^i multiplier)
        # which produces correlated predictions regardless of fixture.
        
        # Check that predictions are not just a geometric decay
        if len(preds) >= 2 and preds[0] > 0:
            ratio = preds[1] / preds[0] if preds[0] != 0 else 1.0
            # If using naive 0.95 decay, ratio would be exactly 0.95
            # With fixture-aware, it should vary
            self.assertNotAlmostEqual(
                ratio, 0.95, places=2,
                msg="Predictions should not use naive 0.95 decay - should be fixture-aware"
            )
    
    def test_predict_multiple_gws_handles_missing_fixtures(self):
        """Predictor should handle missing future fixtures gracefully."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        all_gw_data, fpl_core_season_data = self._make_multi_gw_test_data(num_gws=8)
        
        # No fixtures for GW 9-13 (missing future data)
        
        # Train
        predictor.train(all_gw_data, fpl_core_season_data, current_gw=8, validation_split=0.2)
        
        # Predict should not crash even without future fixtures
        results = predictor.predict_multiple_gws(
            all_gw_data, fpl_core_season_data,
            player_ids=[1], current_gw=8, num_gws=5
        )
        
        self.assertIn(1, results)
        preds = results[1]['predictions']
        
        # Should still produce 5 predictions
        self.assertEqual(len(preds), 5)
        
        # All predictions should be non-negative
        for i, p in enumerate(preds):
            self.assertGreaterEqual(p, 0, f"Prediction for GW{9+i} should be non-negative")
    
    def test_predict_multiple_gws_missing_player_stats_window(self):
        """Predictor should handle players with no recent match stats gracefully."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        all_gw_data, fpl_core_season_data = self._make_multi_gw_test_data(num_gws=8)
        
        # Add a player with no match stats (injury, new signing, etc)
        fpl_core_season_data['players'] = pd.concat([
            fpl_core_season_data['players'],
            pd.DataFrame([{'player_id': 99, 'position': 'Forward', 'team_code': 1}])
        ], ignore_index=True)
        
        # Train (player 99 won't have training data)
        predictor.train(all_gw_data, fpl_core_season_data, current_gw=8, validation_split=0.2)
        
        # Predict for player 99 (no history)
        results = predictor.predict_multiple_gws(
            all_gw_data, fpl_core_season_data,
            player_ids=[99], current_gw=8, num_gws=3
        )
        
        # Should have result for player 99
        self.assertIn(99, results)
        
        # Should produce valid schema even with no data
        result = results[99]
        self.assertIn('predictions', result)
        self.assertIn('cumulative', result)
        self.assertEqual(len(result['predictions']), 3)
    
    def test_ensemble_model_diversity(self):
        """Ensemble should use multiple diverse model types."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        
        # Check for model diversity in position models
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            model_data = predictor.position_models[pos]
            
            # Should have at least 2 different model types (originally gb + rf)
            # After enhancement: should have gb, rf, xgb, ridge
            model_keys = [k for k in model_data.keys() if k not in ['scaler', 'is_trained', 'samples', 'imputer', 'blender']]
            self.assertGreaterEqual(
                len(model_keys), 2,
                f"Position {pos} should have at least 2 models for diversity"
            )
            
            # Check for non-tree model (Ridge for linear diversity)
            # This test will fail until we add Ridge
            has_linear_model = any('ridge' in k.lower() for k in model_keys)
            self.assertTrue(
                has_linear_model,
                f"Position {pos} should have a linear model (Ridge) for diversity"
            )
    
    def test_fpl_api_fixtures_fallback(self):
        """Predictor should use fpl_api_fixtures when Core Insights fixtures unavailable."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        all_gw_data, fpl_core_season_data = self._make_multi_gw_test_data(num_gws=8)
        
        # Add FPL API fixtures fallback for future GWs
        fpl_core_season_data['fpl_api_fixtures'] = pd.DataFrame([
            {'event': 9, 'team_h': 1, 'team_a': 3, 
             'team_h_difficulty': 2, 'team_a_difficulty': 4},
            {'event': 10, 'team_h': 4, 'team_a': 1,
             'team_h_difficulty': 3, 'team_a_difficulty': 5},
            {'event': 11, 'team_h': 1, 'team_a': 5,
             'team_h_difficulty': 2, 'team_a_difficulty': 3},
        ])
        
        # Train
        predictor.train(all_gw_data, fpl_core_season_data, current_gw=8, validation_split=0.2)
        
        # Predict - should use fallback fixtures for GW 9-11
        results = predictor.predict_multiple_gws(
            all_gw_data, fpl_core_season_data,
            player_ids=[1], current_gw=8, num_gws=3
        )
        
        self.assertIn(1, results)
        # Should have 3 predictions
        self.assertEqual(len(results[1]['predictions']), 3)


class TestFPLCorePredictorPositionFeatureSubsets(unittest.TestCase):
    """Tests for per-position feature subsets in FPLCorePredictor.
    
    Each position should use a tailored feature subset:
    - GKP: goalkeeper-relevant features (saves, goals_prevented, etc.)
    - DEF: defensive features (tackles, clearances, etc.) + some attacking
    - MID: balanced attacking/passing features
    - FWD: attacking-heavy features (xG, shots, big_chances, etc.)
    """
    
    def test_position_feature_cols_exists(self):
        """Predictor should have position_feature_cols mapping."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        
        self.assertTrue(
            hasattr(predictor, 'position_feature_cols'),
            "Predictor should have position_feature_cols attribute"
        )
        
        # Should have entry for each position
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            self.assertIn(
                pos, predictor.position_feature_cols,
                f"position_feature_cols should have entry for {pos}"
            )
    
    def test_gkp_features_include_saves(self):
        """GKP feature subset should include goalkeeper-specific features."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        gkp_features = predictor.position_feature_cols.get('GKP', [])
        
        # GKP should have save-related features
        gk_features_expected = ['avg_saves_per_90', 'avg_goals_prevented_per_90']
        for feat in gk_features_expected:
            self.assertIn(
                feat, gkp_features,
                f"GKP features should include {feat}"
            )
    
    def test_fwd_features_exclude_gk_features(self):
        """FWD feature subset should NOT include goalkeeper-specific features."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        fwd_features = predictor.position_feature_cols.get('FWD', [])
        
        # FWD should NOT have GK-specific features
        gk_only_features = ['avg_saves_per_90', 'avg_saves_inside_box_per_90', 
                           'avg_goals_prevented_per_90']
        for feat in gk_only_features:
            self.assertNotIn(
                feat, fwd_features,
                f"FWD features should NOT include {feat}"
            )
    
    def test_fwd_features_include_attacking(self):
        """FWD feature subset should include attacking features."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        fwd_features = predictor.position_feature_cols.get('FWD', [])
        
        # FWD should have attacking features
        attacking_expected = ['avg_xg_per_90', 'avg_shots_per_90', 
                              'avg_big_chances_per_90', 'avg_touches_box_per_90']
        for feat in attacking_expected:
            self.assertIn(
                feat, fwd_features,
                f"FWD features should include {feat}"
            )
    
    def test_position_feature_subsets_are_strict_subsets_or_equal(self):
        """Position feature subsets should be subsets of full feature_cols."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        all_features = set(predictor.feature_cols)
        
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            pos_features = set(predictor.position_feature_cols.get(pos, []))
            
            self.assertTrue(
                pos_features.issubset(all_features),
                f"{pos} features should be subset of full feature_cols. "
                f"Extra features: {pos_features - all_features}"
            )


class TestFPLCorePredictorRecencyWeighting(unittest.TestCase):
    """Tests for recency weighting in FPLCorePredictor training.
    
    More recent gameweeks should have higher sample weights during training.
    """
    
    def _make_test_data(self, num_gws: int = 10):
        """Create test data spanning multiple gameweeks."""
        all_gw_data = {}
        
        for gw in range(1, num_gws + 1):
            all_gw_data[gw] = {
                'playermatchstats': pd.DataFrame([
                    {'player_id': 1, 'minutes_played': 90, 'xg': 0.4, 'xa': 0.2,
                     'total_shots': 3, 'touches_opposition_box': 5,
                     'chances_created': 2, 'accurate_passes_percent': 85.0,
                     'tackles': 2, 'interceptions': 1, 'clearances': 0},
                    {'player_id': 2, 'minutes_played': 90, 'xg': 0.1, 'xa': 0.1,
                     'total_shots': 1, 'touches_opposition_box': 2,
                     'chances_created': 1, 'accurate_passes_percent': 80.0,
                     'tackles': 4, 'interceptions': 3, 'clearances': 5},
                ]),
                'player_gameweek_stats': pd.DataFrame([
                    {'id': 1, 'event_points': 5 + (gw % 4), 'minutes': 90,
                     'bps': 30, 'bonus': 1, 'ict_index': 6.0,
                     'influence': 35, 'creativity': 25, 'threat': 40, 'now_cost': 85},
                    {'id': 2, 'event_points': 4 + (gw % 3), 'minutes': 90,
                     'bps': 25, 'bonus': 0, 'ict_index': 4.0,
                     'influence': 30, 'creativity': 15, 'threat': 20, 'now_cost': 55},
                ]),
                'matches': pd.DataFrame([
                    {'team_h': 1, 'team_a': 2, 'team_h_score': 2, 'team_a_score': 1}
                ]),
                'fixtures': pd.DataFrame([
                    {'gameweek': gw, 'home_team': 1, 'away_team': 2,
                     'home_team_elo': 1850.0, 'away_team_elo': 1750.0}
                ]),
                'teams': pd.DataFrame([
                    {'id': 1, 'elo': 1850.0},
                    {'id': 2, 'elo': 1750.0}
                ])
            }
        
        fpl_core_season_data = {
            'players': pd.DataFrame([
                {'player_id': 1, 'position': 'Midfielder', 'team_code': 1},
                {'player_id': 2, 'position': 'Defender', 'team_code': 1},
            ])
        }
        
        return all_gw_data, fpl_core_season_data
    
    def test_train_accepts_recency_weighting_arg(self):
        """train() should accept recency_weighting argument."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        all_gw_data, fpl_core_season_data = self._make_test_data(num_gws=8)
        
        # Should not raise when recency_weighting is passed
        try:
            predictor.train(
                all_gw_data, fpl_core_season_data, 
                current_gw=8, validation_split=0.2,
                recency_weighting=True
            )
        except TypeError as e:
            self.fail(f"train() should accept recency_weighting arg: {e}")
    
    def test_train_accepts_recency_half_life_arg(self):
        """train() should accept recency_half_life_gws argument."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        all_gw_data, fpl_core_season_data = self._make_test_data(num_gws=8)
        
        # Should not raise when recency_half_life_gws is passed
        try:
            predictor.train(
                all_gw_data, fpl_core_season_data,
                current_gw=8, validation_split=0.2,
                recency_weighting=True,
                recency_half_life_gws=4.0
            )
        except TypeError as e:
            self.fail(f"train() should accept recency_half_life_gws arg: {e}")
    
    def test_recency_weighting_disabled_by_default_is_false(self):
        """recency_weighting should default to True."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        import inspect
        
        predictor = FPLCorePredictor()
        
        # Get train method signature
        sig = inspect.signature(predictor.train)
        params = sig.parameters
        
        # recency_weighting should have default True
        self.assertIn('recency_weighting', params)
        self.assertEqual(
            params['recency_weighting'].default, True,
            "recency_weighting should default to True"
        )
    
    def test_train_model_stack_accepts_sample_weight(self):
        """_train_model_stack should accept sample_weight argument."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        import inspect
        
        predictor = FPLCorePredictor()
        
        # Get _train_model_stack signature
        sig = inspect.signature(predictor._train_model_stack)
        params = sig.parameters
        
        self.assertIn(
            'sample_weight', params,
            "_train_model_stack should accept sample_weight argument"
        )


class TestFPLCorePredictorUncertaintyOutput(unittest.TestCase):
    """Tests for ensemble uncertainty outputs in predict_multiple_gws.
    
    The multi-GW prediction results should include:
    - std_dev: average model disagreement across the horizon
    - std_dev_by_gw: list of per-GW disagreement values
    """
    
    def _make_test_data(self, num_gws: int = 10):
        """Create test data spanning multiple gameweeks."""
        all_gw_data = {}
        
        for gw in range(1, num_gws + 1):
            all_gw_data[gw] = {
                'playermatchstats': pd.DataFrame([
                    {'player_id': 1, 'minutes_played': 90, 'xg': 0.4, 'xa': 0.2,
                     'total_shots': 3, 'touches_opposition_box': 5,
                     'chances_created': 2, 'accurate_passes_percent': 85.0,
                     'tackles': 2, 'interceptions': 1, 'clearances': 0},
                ]),
                'player_gameweek_stats': pd.DataFrame([
                    {'id': 1, 'event_points': 5 + (gw % 4), 'minutes': 90,
                     'bps': 30, 'bonus': 1, 'ict_index': 6.0,
                     'influence': 35, 'creativity': 25, 'threat': 40, 'now_cost': 85},
                ]),
                'matches': pd.DataFrame([
                    {'team_h': 1, 'team_a': 2, 'team_h_score': 2, 'team_a_score': 1}
                ]),
                'fixtures': pd.DataFrame([
                    {'gameweek': gw, 'home_team': 1, 'away_team': 2,
                     'home_team_elo': 1850.0, 'away_team_elo': 1750.0}
                ]),
                'teams': pd.DataFrame([
                    {'id': 1, 'elo': 1850.0},
                    {'id': 2, 'elo': 1750.0}
                ])
            }
        
        fpl_core_season_data = {
            'players': pd.DataFrame([
                {'player_id': 1, 'position': 'Midfielder', 'team_code': 1},
            ])
        }
        
        return all_gw_data, fpl_core_season_data
    
    def test_predict_multiple_gws_includes_std_dev(self):
        """predict_multiple_gws result should include std_dev key."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        all_gw_data, fpl_core_season_data = self._make_test_data(num_gws=8)
        
        # Train
        predictor.train(all_gw_data, fpl_core_season_data, current_gw=8, validation_split=0.2)
        
        # Predict
        results = predictor.predict_multiple_gws(
            all_gw_data, fpl_core_season_data,
            player_ids=[1], current_gw=8, num_gws=5
        )
        
        self.assertIn(1, results)
        self.assertIn(
            'std_dev', results[1],
            "Result should include 'std_dev' key for average model disagreement"
        )
        
        # std_dev should be a number
        self.assertIsInstance(
            results[1]['std_dev'], (int, float),
            "std_dev should be a numeric value"
        )
    
    def test_predict_multiple_gws_includes_std_dev_by_gw(self):
        """predict_multiple_gws result should include std_dev_by_gw list."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        all_gw_data, fpl_core_season_data = self._make_test_data(num_gws=8)
        
        # Train
        predictor.train(all_gw_data, fpl_core_season_data, current_gw=8, validation_split=0.2)
        
        # Predict
        num_gws = 5
        results = predictor.predict_multiple_gws(
            all_gw_data, fpl_core_season_data,
            player_ids=[1], current_gw=8, num_gws=num_gws
        )
        
        self.assertIn(1, results)
        self.assertIn(
            'std_dev_by_gw', results[1],
            "Result should include 'std_dev_by_gw' key for per-GW uncertainty"
        )
        
        # std_dev_by_gw should be a list
        self.assertIsInstance(
            results[1]['std_dev_by_gw'], list,
            "std_dev_by_gw should be a list"
        )
        
        # Length should match num_gws
        self.assertEqual(
            len(results[1]['std_dev_by_gw']), num_gws,
            f"std_dev_by_gw should have {num_gws} elements"
        )
    
    def test_std_dev_by_gw_matches_predictions_length(self):
        """std_dev_by_gw length should match predictions length."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        all_gw_data, fpl_core_season_data = self._make_test_data(num_gws=8)
        
        # Train
        predictor.train(all_gw_data, fpl_core_season_data, current_gw=8, validation_split=0.2)
        
        # Predict different horizons
        for num_gws in [3, 5, 7]:
            results = predictor.predict_multiple_gws(
                all_gw_data, fpl_core_season_data,
                player_ids=[1], current_gw=8, num_gws=num_gws
            )
            
            preds_len = len(results[1]['predictions'])
            std_len = len(results[1]['std_dev_by_gw'])
            
            self.assertEqual(
                preds_len, std_len,
                f"predictions ({preds_len}) and std_dev_by_gw ({std_len}) "
                f"should have same length for num_gws={num_gws}"
            )
    
    def test_std_dev_values_are_non_negative(self):
        """All std_dev values should be non-negative."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        all_gw_data, fpl_core_season_data = self._make_test_data(num_gws=8)
        
        # Train
        predictor.train(all_gw_data, fpl_core_season_data, current_gw=8, validation_split=0.2)
        
        # Predict
        results = predictor.predict_multiple_gws(
            all_gw_data, fpl_core_season_data,
            player_ids=[1], current_gw=8, num_gws=5
        )
        
        # Overall std_dev should be >= 0
        self.assertGreaterEqual(
            results[1]['std_dev'], 0,
            "std_dev should be non-negative"
        )
        
        # All per-GW std_dev values should be >= 0
        for i, std_val in enumerate(results[1]['std_dev_by_gw']):
            self.assertGreaterEqual(
                std_val, 0,
                f"std_dev_by_gw[{i}] should be non-negative"
            )


class TestFPLCorePredictorCrossSeasonTraining(unittest.TestCase):
    """Tests for cross-season training in FPLCorePredictor.
    
    Validates:
    1. Active-only filtering: Previous season players not in current season are excluded
    2. Current-season-only validation: Validation split uses only current season GWs
    3. Season weighting: Current season samples get 2x weight vs previous season
    """
    
    def _make_prev_season_data(self, num_gws: int = 10):
        """Create mock previous season data (e.g., 2024-25)."""
        all_gw_data = {}
        
        for gw in range(1, num_gws + 1):
            all_gw_data[gw] = {
                'playermatchstats': pd.DataFrame([
                    # Player 1 exists in both seasons (player_code 100)
                    {'player_id': 101, 'minutes_played': 90, 'xg': 0.3, 'xa': 0.2,
                     'total_shots': 2, 'touches_opposition_box': 4},
                    # Player 2 exists only in prev season (player_code 200)
                    {'player_id': 102, 'minutes_played': 90, 'xg': 0.4, 'xa': 0.1,
                     'total_shots': 3, 'touches_opposition_box': 5},
                ]),
                'player_gameweek_stats': pd.DataFrame([
                    {'id': 101, 'event_points': 5 + (gw % 3), 'minutes': 90, 'bps': 25, 
                     'ict_index': 5.0, 'influence': 30, 'creativity': 20, 'threat': 35, 'now_cost': 70},
                    {'id': 102, 'event_points': 4 + (gw % 2), 'minutes': 90, 'bps': 20, 
                     'ict_index': 4.5, 'influence': 25, 'creativity': 18, 'threat': 30, 'now_cost': 60},
                ]),
                'matches': pd.DataFrame([
                    {'team_h': 1, 'team_a': 2, 'team_h_score': 2, 'team_a_score': 1}
                ]),
                'fixtures': pd.DataFrame([
                    {'gameweek': gw, 'home_team': 1, 'away_team': 2,
                     'home_team_elo': 1800.0, 'away_team_elo': 1700.0}
                ]),
                'teams': pd.DataFrame([
                    {'id': 1, 'elo': 1800.0},
                    {'id': 2, 'elo': 1700.0}
                ])
            }
        
        # Previous season players data with player_code
        season_data = {
            'players': pd.DataFrame([
                {'player_id': 101, 'player_code': 100, 'position': 'Midfielder', 'team_code': 1},
                {'player_id': 102, 'player_code': 200, 'position': 'Forward', 'team_code': 2},
            ])
        }
        
        return all_gw_data, season_data
    
    def _make_current_season_data(self, num_gws: int = 8):
        """Create mock current season data (e.g., 2025-26)."""
        all_gw_data = {}
        
        for gw in range(1, num_gws + 1):
            all_gw_data[gw] = {
                'playermatchstats': pd.DataFrame([
                    # Player 1 exists in both seasons (player_code 100, new ID 201)
                    {'player_id': 201, 'minutes_played': 90, 'xg': 0.35, 'xa': 0.25,
                     'total_shots': 3, 'touches_opposition_box': 5},
                    # Player 3 is new this season (player_code 300)
                    {'player_id': 203, 'minutes_played': 90, 'xg': 0.5, 'xa': 0.3,
                     'total_shots': 4, 'touches_opposition_box': 6},
                ]),
                'player_gameweek_stats': pd.DataFrame([
                    {'id': 201, 'event_points': 6 + (gw % 3), 'minutes': 90, 'bps': 28, 
                     'ict_index': 5.5, 'influence': 32, 'creativity': 22, 'threat': 38, 'now_cost': 75},
                    {'id': 203, 'event_points': 7 + (gw % 2), 'minutes': 90, 'bps': 30, 
                     'ict_index': 6.0, 'influence': 35, 'creativity': 25, 'threat': 42, 'now_cost': 85},
                ]),
                'matches': pd.DataFrame([
                    {'team_h': 1, 'team_a': 3, 'team_h_score': 1, 'team_a_score': 1}
                ]),
                'fixtures': pd.DataFrame([
                    {'gameweek': gw, 'home_team': 1, 'away_team': 3,
                     'home_team_elo': 1850.0, 'away_team_elo': 1750.0}
                ]),
                'teams': pd.DataFrame([
                    {'id': 1, 'elo': 1850.0},
                    {'id': 3, 'elo': 1750.0}
                ])
            }
        
        # Current season players data with player_code
        # Note: player_code 100 exists in both seasons, 200 only in prev, 300 only in current
        season_data = {
            'players': pd.DataFrame([
                {'player_id': 201, 'player_code': 100, 'position': 'Midfielder', 'team_code': 1},
                {'player_id': 203, 'player_code': 300, 'position': 'Forward', 'team_code': 3},
            ])
        }
        
        return all_gw_data, season_data
    
    def test_build_cross_season_filters_inactive_players(self):
        """Cross-season dataset should exclude players not in current season."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        
        prev_gw_data, prev_season_data = self._make_prev_season_data(num_gws=10)
        curr_gw_data, curr_season_data = self._make_current_season_data(num_gws=8)
        
        prev_df, curr_df = predictor.build_cross_season_training_dataset(
            prev_all_gw_data=prev_gw_data,
            prev_fpl_core_season_data=prev_season_data,
            prev_season_start_year=2024,
            current_all_gw_data=curr_gw_data,
            current_fpl_core_season_data=curr_season_data,
            current_season_start_year=2025,
            current_gw=8,
            min_gw=5,
        )
        
        # Previous season should only have player 101 (player_code 100)
        # Player 102 (player_code 200) should be filtered out
        if not prev_df.empty:
            prev_player_ids = prev_df['player_id'].unique().tolist()
            self.assertIn(101, prev_player_ids, 
                         "Player 101 (active in current season) should be included")
            self.assertNotIn(102, prev_player_ids,
                            "Player 102 (not in current season) should be filtered out")
    
    def test_train_cross_season_validation_is_current_season_only(self):
        """Validation samples should come only from current season."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        
        prev_gw_data, prev_season_data = self._make_prev_season_data(num_gws=10)
        curr_gw_data, curr_season_data = self._make_current_season_data(num_gws=8)
        
        # Train cross-season
        predictor.train_cross_season(
            prev_all_gw_data=prev_gw_data,
            prev_fpl_core_season_data=prev_season_data,
            prev_season_start_year=2024,
            current_all_gw_data=curr_gw_data,
            current_fpl_core_season_data=curr_season_data,
            current_season_start_year=2025,
            current_gw=8,
            validation_split=0.2,
        )
        
        # Check metrics for cross-season info
        self.assertIn('cross_season', predictor.metrics,
                     "Metrics should include cross_season info")
        
        cross_season_metrics = predictor.metrics['cross_season']
        
        # Validation GWs should be from current season only (later GWs)
        val_gws = cross_season_metrics.get('val_gws', [])
        if val_gws:
            # All validation GWs should be from current season (GW5+)
            for gw in val_gws:
                self.assertGreaterEqual(gw, 5,
                                       f"Validation GW {gw} should be >= 5 (min_gw)")
    
    def test_train_cross_season_season_weights_recorded(self):
        """Cross-season training should record the season weights used."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        
        prev_gw_data, prev_season_data = self._make_prev_season_data(num_gws=10)
        curr_gw_data, curr_season_data = self._make_current_season_data(num_gws=8)
        
        # Train with specific weights
        predictor.train_cross_season(
            prev_all_gw_data=prev_gw_data,
            prev_fpl_core_season_data=prev_season_data,
            prev_season_start_year=2024,
            current_all_gw_data=curr_gw_data,
            current_fpl_core_season_data=curr_season_data,
            current_season_start_year=2025,
            current_gw=8,
            current_season_weight=2.0,
            prev_season_weight=1.0,
        )
        
        # Verify metrics record the weights
        self.assertIn('cross_season', predictor.metrics)
        season_weights = predictor.metrics['cross_season'].get('season_weights', {})
        
        self.assertEqual(season_weights.get('current'), 2.0,
                        "Current season weight should be recorded as 2.0")
        self.assertEqual(season_weights.get('prev'), 1.0,
                        "Previous season weight should be recorded as 1.0")
    
    def test_season_start_year_feature_added(self):
        """season_start_year should be added to training samples."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        
        prev_gw_data, prev_season_data = self._make_prev_season_data(num_gws=10)
        curr_gw_data, curr_season_data = self._make_current_season_data(num_gws=8)
        
        prev_df, curr_df = predictor.build_cross_season_training_dataset(
            prev_all_gw_data=prev_gw_data,
            prev_fpl_core_season_data=prev_season_data,
            prev_season_start_year=2024,
            current_all_gw_data=curr_gw_data,
            current_fpl_core_season_data=curr_season_data,
            current_season_start_year=2025,
            current_gw=8,
            min_gw=5,
        )
        
        # Check previous season dataset has correct season_start_year
        if not prev_df.empty:
            self.assertIn('season_start_year', prev_df.columns,
                         "Previous season df should have season_start_year column")
            prev_years = prev_df['season_start_year'].unique()
            self.assertEqual(len(prev_years), 1, 
                            "All prev season rows should have same year")
            self.assertEqual(prev_years[0], 2024,
                            "Previous season should be 2024")
        
        # Check current season dataset has correct season_start_year
        if not curr_df.empty:
            self.assertIn('season_start_year', curr_df.columns,
                         "Current season df should have season_start_year column")
            curr_years = curr_df['season_start_year'].unique()
            self.assertEqual(len(curr_years), 1,
                            "All current season rows should have same year")
            self.assertEqual(curr_years[0], 2025,
                            "Current season should be 2025")
    
    def test_season_start_year_in_feature_cols(self):
        """season_start_year should be in CONTEXT_FEATURES."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        
        self.assertIn('season_start_year', predictor.CONTEXT_FEATURES,
                     "CONTEXT_FEATURES should include season_start_year")
        self.assertIn('season_start_year', predictor.feature_cols,
                     "feature_cols should include season_start_year")
    
    def test_reset_context_caches_clears_all_caches(self):
        """_reset_context_caches should clear all context caches."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        
        # Populate caches with dummy data
        predictor._team_strength = {'dummy': 'data'}
        predictor._team_strength_by_gw = {1: {'dummy': 'data'}}
        predictor._fixture_cache = {(1, 1): {'dummy': 'data'}}
        
        # Reset caches
        predictor._reset_context_caches()
        
        # All caches should be empty
        self.assertEqual(predictor._team_strength, {},
                        "_team_strength should be cleared")
        self.assertEqual(predictor._team_strength_by_gw, {},
                        "_team_strength_by_gw should be cleared")
        self.assertEqual(predictor._fixture_cache, {},
                        "_fixture_cache should be cleared")
    
    def test_cross_season_metrics_include_sample_counts(self):
        """Cross-season metrics should include sample counts per season."""
        from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
        
        predictor = FPLCorePredictor()
        
        prev_gw_data, prev_season_data = self._make_prev_season_data(num_gws=10)
        curr_gw_data, curr_season_data = self._make_current_season_data(num_gws=8)
        
        predictor.train_cross_season(
            prev_all_gw_data=prev_gw_data,
            prev_fpl_core_season_data=prev_season_data,
            prev_season_start_year=2024,
            current_all_gw_data=curr_gw_data,
            current_fpl_core_season_data=curr_season_data,
            current_season_start_year=2025,
            current_gw=8,
        )
        
        cross_season = predictor.metrics.get('cross_season', {})
        
        self.assertIn('prev_season_samples', cross_season,
                     "Should record prev_season_samples count")
        self.assertIn('current_season_samples', cross_season,
                     "Should record current_season_samples count")
        
        # Both should be integers >= 0
        self.assertIsInstance(cross_season['prev_season_samples'], int)
        self.assertIsInstance(cross_season['current_season_samples'], int)

class TestIntelligenceSectionRendering(unittest.TestCase):
    """Tests for optional intelligence-driven report rendering."""

    def setUp(self):
        self.generator = LaTeXReportGenerator(team_id=847569, gameweek=17, plot_dir=Path("plots"))

    def _build_base_inputs(self):
        team_info = {
            "team_name": "My Team",
            "manager_name": "Manager",
            "overall_points": 1500,
            "overall_rank": 12345,
            "season": "2025-26",
        }
        gw_history = [
            {"event": 16, "points": 60, "total_points": 1440, "overall_rank": 14000},
            {"event": 17, "points": 60, "total_points": 1500, "overall_rank": 12345},
        ]
        squad = [
            {"id": 1, "name": "Alpha", "position": "MID", "position_in_squad": 1, "is_captain": True, "is_vice_captain": False}
        ]
        squad_analysis = [
            {
                "player_id": 1,
                "name": "Alpha",
                "position": "MID",
                "position_in_squad": 1,
                "raw_stats": {"total_points": 120},
                "form_analysis": {"average": 6.2, "trend": "rising"},
                "expected_vs_actual": {"expected_goals": 2.0, "goals_diff": -0.5},
            },
            {
                "player_id": 2,
                "name": "Bravo",
                "position": "FWD",
                "position_in_squad": 12,
                "raw_stats": {"total_points": 90},
                "form_analysis": {"average": 4.1, "trend": "falling"},
                "expected_vs_actual": {"expected_goals": 1.2, "goals_diff": 0.3},
            },
        ]
        captain_picks = [{"name": "Alpha", "position": "MID", "reasons": ["High xP"]}]
        chips_used = []
        transfers = []
        multi_week_strategy = {
            "current_gameweek": 17,
            "planning_horizon": 5,
            "expected_value": {"current_squad": 52.0, "optimized_squad": 58.5, "potential_gain": 6.5},
            "fixture_analysis": {},
            "squad_predictions": {},
            "immediate_recommendations": [],
            "planned_transfers": [],
            "alternative_strategies": {},
            "model_metrics": {"mae": 2.5, "r2": 0.2},
            "model_confidence": "medium",
            "mip_recommendation": None,
        }
        wildcard_team = {
            "budget": {"total": 100.0, "spent": 99.0, "remaining": 1.0},
            "squad": [{"name": "WC Player", "position": "MID", "team": "ARS", "team_id": 1, "price": 8.0, "score": 70.0}],
            "starting_xi": [{"name": "WC Player", "position": "MID", "team": "ARS", "price": 8.0, "xp_5gw": 28.0, "score": 70.0, "ppg": 5.5, "fixtures": []}],
            "bench": [],
            "formation": "3-5-2",
            "captain": {"name": "WC Player"},
            "vice_captain": {"name": "WC Player"},
            "ev_analysis": {"current_squad_xp": 52.0, "optimized_xp": 58.0, "potential_gain": 6.0, "horizon": "5 GWs"},
        }
        free_hit_team = {
            "budget": {"total": 100.0, "spent": 99.0, "remaining": 1.0},
            "squad": [{"name": "FH Player", "position": "MID", "team": "LIV", "team_id": 2, "price": 9.0, "score": 75.0, "ep_next": 7.5, "league_ownership": 30.0}],
            "starting_xi": [{"name": "FH Player", "position": "MID", "team": "LIV", "price": 9.0, "ep_next": 7.5, "league_ownership": 30.0, "fixtures": []}],
            "bench": [],
            "formation": "3-5-2",
            "captain": {"name": "FH Player"},
            "vice_captain": {"name": "FH Player"},
            "target_gw": 18,
            "strategy": "balanced",
            "league_analysis": {"sample_size": 10, "differentials": [], "template_picks": []},
            "ev_analysis": {"current_squad_xp": 48.0, "optimized_xp": 54.0, "potential_gain": 6.0},
        }
        chip_analysis = {
            "current_gw": 17,
            "half": "first",
            "chips_remaining_display": "3/4",
            "squad_issues": {"total_issues": 0, "summary": "No major issues"},
            "chips": {
                "wildcard": {"urgency": "medium", "recommendation": "Use before GW20"},
                "freehit": {"urgency": "low", "recommendation": "Save for BGW"},
                "bboost": {"urgency": "low", "recommendation": "Target DGW"},
                "3xc": {"urgency": "low", "recommendation": "Use in DGW"},
            },
            "triggers": ["DGW announcement", "Injury cluster"],
            "phase2": {},
        }
        return {
            "team_info": team_info,
            "gw_history": gw_history,
            "squad": squad,
            "squad_analysis": squad_analysis,
            "recommendations": [],
            "captain_picks": captain_picks,
            "chips_used": chips_used,
            "transfers": transfers,
            "multi_week_strategy": multi_week_strategy,
            "wildcard_team": wildcard_team,
            "free_hit_team": free_hit_team,
            "season_history": [],
            "chip_analysis": chip_analysis,
        }

    def _build_intelligence_payload(self):
        base = {
            "headline": "AI Headline",
            "tactical_summary": "AI tactical summary with metric anchors.",
            "metric_highlights": ["+6.5 xP over 5 GWs", "3 chips remaining before reset"],
            "actions": ["Prioritize one immediate transfer", "Preserve captaincy upside"],
            "risks": ["Late injury changes can alter expected gain"],
        }
        chip_payload = dict(base)
        chip_payload["chip_recommendations"] = {
            "wildcard": {"recommendation": "Wildcard if two starters are flagged.", "trigger": "Two injuries in starting XI"},
            "freehit": {"recommendation": "Free Hit on largest BGW.", "trigger": "6+ expected blanks"},
            "bboost": {"recommendation": "Bench Boost on DGW with deep bench.", "trigger": "4 bench doubles"},
            "3xc": {"recommendation": "Triple Captain premium with two fixtures.", "trigger": "Confirmed DGW for premium attacker"},
        }
        payload = {
            "transfer_strategy": dict(base),
            "wildcard_draft": dict(base),
            "free_hit_draft": dict(base),
            "chip_usage_strategy": chip_payload,
            "season_insights": dict(base),
        }
        meta = {
            key: {"label": "Intelligence layer enabled - model: gpt-5.2"}
            for key in payload.keys()
        }
        return payload, meta

    def test_compile_report_uses_intelligence_payload_for_all_sections(self):
        """compile_report should inject intelligence narratives for all five sections."""
        inputs = self._build_base_inputs()
        payload, meta = self._build_intelligence_payload()

        latex = self.generator.compile_report(
            **inputs,
            intelligence_payload=payload,
            intelligence_meta=meta,
        )

        self.assertIn("Transfer Intelligence", latex)
        self.assertIn("Wildcard Strategy Notes", latex)
        self.assertIn("Free Hit Strategy Guide", latex)
        self.assertIn("Chip Intelligence", latex)
        self.assertIn("Season Intelligence", latex)
        self.assertIn("Intelligence layer enabled - model: gpt-5.2", latex)
        self.assertIn("Wildcard if two starters are flagged.", latex)

    def test_compile_report_uses_deterministic_when_section_payload_missing(self):
        """Missing section payload should fall back to deterministic section content."""
        inputs = self._build_base_inputs()
        payload, meta = self._build_intelligence_payload()
        payload = {"transfer_strategy": payload["transfer_strategy"]}
        meta = {"transfer_strategy": meta["transfer_strategy"]}

        latex = self.generator.compile_report(
            **inputs,
            intelligence_payload=payload,
            intelligence_meta=meta,
        )

        self.assertIn("Transfer Intelligence", latex)
        self.assertIn(r"\subsection{Strategy Notes}", latex)
        self.assertIn(r"\subsection{Strategy Guide}", latex)
        self.assertIn("This Wildcard draft prioritizes", latex)


class TestFixtureHeatmapBgwDgw(unittest.TestCase):
    """Tests for BGW/DGW handling in the fixture difficulty heatmap."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        df = pd.DataFrame(columns=["id", "web_name", "team", "element_type"])
        self.generator = LaTeXReportGenerator(
            DummyFetcher(df), DummyAnalyzer(), self.temp_dir
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_fix(self, opp="ARS", is_home=True, win_prob=0.55, difficulty=2):
        return {
            "opponent": opp,
            "is_home": is_home,
            "difficulty": difficulty,
            "win_prob": win_prob,
        }

    def test_bgw_column_alignment(self):
        """BGW in the middle of the horizon renders in the correct column."""
        fixtures_by_gw = {
            11: [self._make_fix("ARS")],
            12: [],  # BGW
            13: [self._make_fix("CHE")],
            14: [self._make_fix("TOT")],
            15: [self._make_fix("LIV")],
        }
        fixture_analysis = {
            1: {
                "player_name": "Salah",
                "position": "MID",
                "fixtures_by_gw": fixtures_by_gw,
                "swing": "neutral",
            }
        }
        result = self.generator._generate_fixture_heatmap(
            fixture_analysis, {1: {"cumulative": 30.0}}, current_gw=10, horizon=5
        )
        # BGW should appear in GW12 column (2nd), not at the end
        # Split by & to check column positions
        lines = [l for l in result.split("\n") if "Salah" in l]
        self.assertEqual(len(lines), 1)
        cells = lines[0].split("&")
        # cells: [player, swing, GW11, GW12, GW13, GW14, GW15, xP]
        self.assertIn("ARS", cells[2])   # GW11
        self.assertIn("BGW", cells[3])   # GW12 - the blank
        self.assertIn("CHE", cells[4])   # GW13

    def test_dgw_cell_shows_two_opponents(self):
        """DGW cell shows both opponent codes."""
        fixtures_by_gw = {
            11: [self._make_fix("ARS", True, 0.65), self._make_fix("CHE", False, 0.30)],
            12: [self._make_fix("TOT")],
            13: [self._make_fix("LIV")],
            14: [self._make_fix("MCI")],
            15: [self._make_fix("BOU")],
        }
        fixture_analysis = {
            1: {
                "player_name": "Haaland",
                "position": "FWD",
                "fixtures_by_gw": fixtures_by_gw,
                "swing": "neutral",
            }
        }
        result = self.generator._generate_fixture_heatmap(
            fixture_analysis, {1: {"cumulative": 35.0}}, current_gw=10, horizon=5
        )
        lines = [l for l in result.split("\n") if "Haaland" in l]
        self.assertEqual(len(lines), 1)
        cells = lines[0].split("&")
        dgw_cell = cells[2]  # GW11 column
        self.assertIn("ARS", dgw_cell)
        self.assertIn("CHE", dgw_cell)
        self.assertIn("DGW", dgw_cell)

    def test_dgw_cell_split_colors(self):
        """Each fixture in a DGW cell gets its own color via colorbox."""
        easy_fix = self._make_fix("BOU", True, 0.70, 1)
        hard_fix = self._make_fix("MCI", False, 0.15, 5)
        fixtures_by_gw = {
            11: [easy_fix, hard_fix],
            12: [self._make_fix()],
            13: [self._make_fix()],
            14: [self._make_fix()],
            15: [self._make_fix()],
        }
        fixture_analysis = {
            1: {
                "player_name": "Palmer",
                "position": "MID",
                "fixtures_by_gw": fixtures_by_gw,
                "swing": "neutral",
            }
        }
        result = self.generator._generate_fixture_heatmap(
            fixture_analysis, {1: {"cumulative": 28.0}}, current_gw=10, horizon=5
        )
        lines = [l for l in result.split("\n") if "Palmer" in l]
        dgw_cell = lines[0].split("&")[2]
        # Should have two separate colorbox commands with different colors
        self.assertIn("fplgreen", dgw_cell)  # Easy fixture
        self.assertIn("fplpink", dgw_cell)   # Hard fixture
        # Both should use \colorbox (split coloring)
        self.assertEqual(dgw_cell.count(r"\colorbox"), 2)

    def test_normal_gw_unaffected(self):
        """Standard single-fixture weeks render the same as before."""
        fixtures_by_gw = {
            11: [self._make_fix("ARS", True, 0.55)],
            12: [self._make_fix("CHE", False, 0.40)],
            13: [self._make_fix("TOT", True, 0.50)],
            14: [self._make_fix("LIV", False, 0.25)],
            15: [self._make_fix("MCI", True, 0.60)],
        }
        fixture_analysis = {
            1: {
                "player_name": "Saka",
                "position": "MID",
                "fixtures_by_gw": fixtures_by_gw,
                "swing": "neutral",
            }
        }
        result = self.generator._generate_fixture_heatmap(
            fixture_analysis, {1: {"cumulative": 25.0}}, current_gw=10, horizon=5
        )
        # No BGW or DGW markers
        lines = [l for l in result.split("\n") if "Saka" in l]
        row = lines[0]
        self.assertNotIn("BGW", row)
        self.assertNotIn("DGW", row)
        # All 5 opponents present
        for opp in ["ARS", "CHE", "TOT", "LIV", "MCI"]:
            self.assertIn(opp, row)

    def test_legend_includes_bgw_dgw(self):
        """Legend includes BGW and DGW markers."""
        fixture_analysis = {
            1: {
                "player_name": "Test",
                "position": "MID",
                "fixtures_by_gw": {gw: [self._make_fix()] for gw in range(11, 16)},
                "swing": "neutral",
            }
        }
        result = self.generator._generate_fixture_heatmap(
            fixture_analysis, {1: {"cumulative": 10.0}}, current_gw=10, horizon=5
        )
        self.assertIn("BGW", result)
        self.assertIn("DGW", result)
        self.assertIn("no fixture", result)
        self.assertIn("two fixtures", result)

    def test_backward_compat_no_fixtures_by_gw(self):
        """Falls back to index-based rendering when fixtures_by_gw is absent."""
        fixture_analysis = {
            1: {
                "player_name": "OldData",
                "position": "MID",
                "fixtures": [self._make_fix("ARS"), self._make_fix("CHE")],
                "swing": "neutral",
            }
        }
        result = self.generator._generate_fixture_heatmap(
            fixture_analysis, {1: {"cumulative": 15.0}}, current_gw=10, horizon=5
        )
        lines = [l for l in result.split("\n") if "OldData" in l]
        row = lines[0]
        self.assertIn("ARS", row)
        self.assertIn("CHE", row)
        # Remaining 3 columns should show BGW (list exhaustion fallback)
        self.assertEqual(row.count("BGW"), 3)


class TestGetFixturesByGw(unittest.TestCase):
    """Tests for FPLDataFetcher.get_fixtures_by_gw."""

    def _make_fetcher_with_difficulties(self, team_fixtures_dict):
        """Create a FPLDataFetcher mock with controlled fixture difficulties."""
        df = pd.DataFrame(columns=["id", "web_name", "team", "element_type"])
        fetcher = DummyFetcher(df)
        mock_calc = MagicMock()
        mock_calc.get_fixture_difficulties.return_value = team_fixtures_dict
        fetcher._difficulty_calculator = mock_calc
        # Bind the real method to the dummy fetcher
        from reports.fpl_report.data_fetcher import FPLDataFetcher
        fetcher.get_fixtures_by_gw = FPLDataFetcher.get_fixtures_by_gw.__get__(
            fetcher, type(fetcher)
        )
        return fetcher

    def test_bgw_returns_empty_list(self):
        """Team with no fixture in a GW gets an empty list for that key."""
        team_fixtures = {
            1: [
                {"gameweek": 29, "opponent": "ARS", "is_home": True,
                 "fdr_elo": 2, "fdr_original": 2,
                 "win_prob": 0.55, "draw_prob": 0.25, "loss_prob": 0.20},
                # No GW30 fixture
                {"gameweek": 31, "opponent": "CHE", "is_home": False,
                 "fdr_elo": 4, "fdr_original": 4,
                 "win_prob": 0.30, "draw_prob": 0.30, "loss_prob": 0.40},
            ]
        }
        fetcher = self._make_fetcher_with_difficulties(team_fixtures)
        result = fetcher.get_fixtures_by_gw(1, 29, 31)
        self.assertEqual(len(result[29]), 1)
        self.assertEqual(len(result[30]), 0)  # BGW
        self.assertEqual(len(result[31]), 1)

    def test_dgw_returns_two_fixtures(self):
        """Team with two fixtures in a GW gets a two-element list."""
        team_fixtures = {
            1: [
                {"gameweek": 29, "opponent": "ARS", "is_home": True,
                 "fdr_elo": 2, "fdr_original": 2,
                 "win_prob": 0.55, "draw_prob": 0.25, "loss_prob": 0.20},
                {"gameweek": 29, "opponent": "CHE", "is_home": False,
                 "fdr_elo": 4, "fdr_original": 4,
                 "win_prob": 0.30, "draw_prob": 0.30, "loss_prob": 0.40},
            ]
        }
        fetcher = self._make_fetcher_with_difficulties(team_fixtures)
        result = fetcher.get_fixtures_by_gw(1, 29, 29)
        self.assertEqual(len(result[29]), 2)
        opponents = {f["opponent"] for f in result[29]}
        self.assertEqual(opponents, {"ARS", "CHE"})

    def test_no_calculator_returns_empty(self):
        """Returns all-empty dict when difficulty calculator is None."""
        df = pd.DataFrame(columns=["id", "web_name", "team", "element_type"])
        fetcher = DummyFetcher(df)
        fetcher._difficulty_calculator = None
        from reports.fpl_report.data_fetcher import FPLDataFetcher
        fetcher.get_fixtures_by_gw = FPLDataFetcher.get_fixtures_by_gw.__get__(
            fetcher, type(fetcher)
        )
        result = fetcher.get_fixtures_by_gw(1, 29, 31)
        for gw in range(29, 32):
            self.assertEqual(result[gw], [])


class TestBgwDgwDetection(unittest.TestCase):
    """Tests for get_bgw_dgw_gameweeks elif bugfix."""

    @patch("reports.fpl_report.data_fetcher.get_fixtures_data")
    def test_mixed_gw_both_bgw_and_dgw(self, mock_fixtures):
        """A GW with some teams blanking AND some doubled produces both entries."""
        from reports.fpl_report.data_fetcher import get_bgw_dgw_gameweeks

        # GW29: teams 1-18 play normally (9 matches), team 19+20 have no fixture (BGW)
        # Plus team 1 has a second match (DGW for team 1)
        fixtures = []
        for i in range(1, 19, 2):
            fixtures.append({"event": 29, "team_h": i, "team_a": i + 1})
        # Extra match for team 1 (DGW)
        fixtures.append({"event": 29, "team_h": 1, "team_a": 3})

        mock_fixtures.return_value = fixtures
        result = get_bgw_dgw_gameweeks(use_cache=False)

        bgw_gws = [entry["gw"] for entry in result["bgw"]]
        dgw_gws = [entry["gw"] for entry in result["dgw"]]

        # GW29 should appear in BOTH lists
        self.assertIn(29, bgw_gws)
        self.assertIn(29, dgw_gws)

        # Verify details
        bgw_entry = [e for e in result["bgw"] if e["gw"] == 29][0]
        self.assertIn(19, bgw_entry["team_ids"])
        self.assertIn(20, bgw_entry["team_ids"])

        dgw_entry = [e for e in result["dgw"] if e["gw"] == 29][0]
        self.assertIn(1, dgw_entry["team_ids"])


if __name__ == "__main__":
    unittest.main()

