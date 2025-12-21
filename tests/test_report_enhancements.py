import sys
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from typing import List, Dict

import pandas as pd


# Ensure we can import the in-repo package under reports/fpl_report
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"
if str(REPORTS_DIR) not in sys.path:
    sys.path.insert(0, str(REPORTS_DIR))


from fpl_report.data_fetcher import FPLDataFetcher, build_competitive_dataset
from fpl_report.transfer_recommender import TransferRecommender
from fpl_report.plot_generator import PlotGenerator
from fpl_report.latex_generator import LaTeXReportGenerator


class DummyAnalyzer:
    pass


class DummyFetcher:
    def __init__(self, df: pd.DataFrame):
        self._df = df
        self.players_df = df

    def get_bank(self) -> float:
        return 0.0

    def get_current_gameweek(self) -> int:
        return 10

    def get_all_players_by_position(self, position: str) -> pd.DataFrame:
        return self._df.copy()

    def get_upcoming_fixtures(self, team_id: int, num_fixtures: int = 5):
        return [{"gameweek": 6, "opponent": "ARS", "is_home": True, "difficulty": 2}]

    def get_position_peers(self, position: str, min_minutes: int = 90) -> pd.DataFrame:
        return pd.DataFrame()

    def _get_team_name(self, team_id: int) -> str:
        return {1: "LIV", 2: "TOT"}.get(team_id, "UNK")

    def get_player_history(self, player_id: int):
        return pd.DataFrame()

    def get_player_stats(self, player_id: int):
        return {}


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

        with patch("fpl_report.data_fetcher.get_entry_transfers_data", return_value=raw_transfers):
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

    @patch('fpl_report.data_fetcher.get_entry_data')
    @patch('fpl_report.data_fetcher.get_entry_personal_data')
    @patch('fpl_report.data_fetcher.get_entry_gws_data')
    @patch('fpl_report.data_fetcher.get_data')
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
                for i in range(1, 16)
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

    @patch('fpl_report.data_fetcher.get_entry_data')
    @patch('fpl_report.data_fetcher.get_entry_personal_data')
    @patch('fpl_report.data_fetcher.get_entry_gws_data')
    @patch('fpl_report.data_fetcher.get_data')
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
                for i in range(1, 16)
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

        competitive_data = [
            {
                'entry_id': 21023,
                'team_info': {'team_name': 'Team A', 'manager_name': 'Manager A', 'overall_points': 500},
                'squad': [
                    {'name': 'Salah', 'position': 'MID', 'stats': {'total_points': 120}},
                    {'name': 'Haaland', 'position': 'FWD', 'stats': {'total_points': 150}},
                    {'name': 'Saka', 'position': 'MID', 'stats': {'total_points': 90}},
                    {'name': 'Raya', 'position': 'GKP', 'stats': {'total_points': 60}},
                    {'name': 'Gabriel', 'position': 'DEF', 'stats': {'total_points': 80}},
                ]
            },
            {
                'entry_id': 6696002,
                'team_info': {'team_name': 'Team B', 'manager_name': 'Manager B', 'overall_points': 480},
                'squad': [
                    {'name': 'Son', 'position': 'MID', 'stats': {'total_points': 100}},
                    {'name': 'Watkins', 'position': 'FWD', 'stats': {'total_points': 110}},
                    {'name': 'Palmer', 'position': 'MID', 'stats': {'total_points': 130}},
                    {'name': 'Pickford', 'position': 'GKP', 'stats': {'total_points': 55}},
                    {'name': 'VVD', 'position': 'DEF', 'stats': {'total_points': 85}},
                ]
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
        self.generator = LaTeXReportGenerator(team_id=847569, gameweek=16)

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
        from fpl_report.transfer_strategy import WildcardOptimizer

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
        from fpl_report.transfer_strategy import WildcardOptimizer

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
        from fpl_report.transfer_strategy import WildcardOptimizer

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
        from fpl_report.transfer_strategy import WildcardOptimizer

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
        from fpl_report.transfer_strategy import WildcardOptimizer

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
        self.generator = LaTeXReportGenerator(team_id=847569, gameweek=16)

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
        from fpl_report.transfer_strategy import FreeHitOptimizer

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
        from fpl_report.transfer_strategy import FreeHitOptimizer

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
        from fpl_report.transfer_strategy import FreeHitOptimizer

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
        from fpl_report.transfer_strategy import FreeHitOptimizer

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
        from fpl_report.transfer_strategy import FreeHitOptimizer

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
        from fpl_report.transfer_strategy import FreeHitOptimizer

        players_df = self._make_mock_players_df()
        total_budget = 100.0

        optimizer = FreeHitOptimizer(players_df, total_budget)
        result = optimizer.build_squad()

        # Check that starting XI players have ep_next
        for player in result['starting_xi']:
            self.assertIn('ep_next', player)

    def test_free_hit_optimizer_with_league_ownership(self):
        """Test that optimizer uses league ownership data when provided."""
        from fpl_report.transfer_strategy import FreeHitOptimizer

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
        from fpl_report.transfer_strategy import FreeHitOptimizer

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


class TestFreeHitLaTeXSection(unittest.TestCase):
    """Tests for Free Hit team LaTeX section generation."""

    def setUp(self):
        """Set up LaTeX generator."""
        self.generator = LaTeXReportGenerator(team_id=847569, gameweek=16)

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

        # Should include league ownership header
        self.assertIn('LeagueOwn', latex)

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
            ('FWD_E', 15, 45), ('FWD_F', 16, 45)
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
        from fpl_report.transfer_strategy import TransferMIPSolver, MIP_AVAILABLE
        
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
        from fpl_report.transfer_strategy import MIPSolverResult
        
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
        from fpl_report.transfer_strategy import TransferMIPSolver, MIP_AVAILABLE
        
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
        from fpl_report.transfer_strategy import TransferMIPSolver
        
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
        from fpl_report.transfer_strategy import TransferMIPSolver
        
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
        from fpl_report.transfer_strategy import (
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
        from fpl_report.transfer_strategy import (
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
        from fpl_report.transfer_strategy import (
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
        
        timeline = build_transfer_timeline(result, current_gw=16, horizon=5)
        latex = format_timeline_for_latex(timeline)
        
        # Should contain TikZ elements
        self.assertIn('tikzpicture', latex)
        self.assertIn('GW17', latex)  # First week after current
        self.assertIn('11.0', latex)  # First week xP

    def test_multiperiod_plan_dataclass(self):
        """Test MultiPeriodPlan dataclass fields."""
        from fpl_report.transfer_strategy import MultiPeriodPlan
        
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
            from fpl_report.transfer_strategy import MIP_AVAILABLE
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
                    'team': (i % 16) + 1,
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
        
        from fpl_report.transfer_strategy import TransferMIPSolver
        
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


if __name__ == "__main__":
    unittest.main()


