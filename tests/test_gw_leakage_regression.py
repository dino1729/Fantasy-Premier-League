import unittest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import sys

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reports.fpl_report.plot_generator import PlotGenerator
from reports.fpl_report.fpl_core_predictor import FPLCorePredictor
import generate_fpl_report as gfr

class TestGWLeakageRegression(unittest.TestCase):
    """Tests to prevent future GW leakage into report analysis windows."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.plot_gen = PlotGenerator(self.temp_dir)
        
        # Setup basic season data
        self.fpl_core_season_data = {
            'playerstats': pd.DataFrame([
                {'id': 1, 'web_name': 'Player1', 'gw': 1, 'minutes': 90},
                {'id': 1, 'web_name': 'Player1', 'gw': 2, 'minutes': 90}
            ]),
            'players': pd.DataFrame([
                {'player_id': 1, 'position': 'Midfielder'}
            ])
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_plot_generator_select_gameweeks_ignores_fixture_only_gws(self):
        """Test that _select_gameweeks ignores GWs that only have fixtures (future GWs)."""
        # Create GW data: GW1-5 have stats, GW6 only has fixtures
        all_gw_data = {}
        for gw in range(1, 6):
            all_gw_data[gw] = {
                'playermatchstats': pd.DataFrame({'dummy': [1]}),
                'fixtures': pd.DataFrame({'event': [gw]})
            }
        
        # GW6 is the "next" GW injected for predictions - has fixtures but no match stats
        all_gw_data[6] = {
            'playermatchstats': None,
            'fixtures': pd.DataFrame({'event': [6]})
        }
        
        # Test Last 5 GWs selection
        # EXPECTATION: Should be [1, 2, 3, 4, 5], NOT [2, 3, 4, 5, 6]
        selected = self.plot_gen._select_gameweeks(all_gw_data, last_n_gw=5)
        
        self.assertEqual(selected, [1, 2, 3, 4, 5], 
                        f"Should select GW1-5, got {selected}. Future GW6 leaked in!")
        
        # Verify range label
        label = self.plot_gen._format_gw_range_label(selected)
        self.assertEqual(label, "GW1-5")

    def test_predict_fixture_fallback_without_mutation(self):
        """Test FPLCorePredictor.predict uses fpl_api_fixtures fallback without needing all_gw_data mutation."""
        predictor = FPLCorePredictor()
        
        # Minimal training data
        all_gw_data = {
            1: {
                'playermatchstats': pd.DataFrame([{
                    'player_id': 1, 'minutes_played': 90, 'total_shots': 1, 'xg': 0.1, 'xa': 0.1,
                    'touches_opposition_box': 1, 'goals': 0, 'assists': 0
                }]),
                'player_gameweek_stats': pd.DataFrame([{
                    'id': 1, 'event_points': 2, 'minutes': 90, 'now_cost': 50
                }]),
                'matches': pd.DataFrame([{'team_h': 1, 'team_a': 2, 'team_h_score': 1, 'team_a_score': 0}]),
                'fixtures': pd.DataFrame([{'gameweek': 1, 'home_team': 1, 'away_team': 2}])
            }
        }
        
        fpl_core_season_data = {
            'players': pd.DataFrame([{'player_id': 1, 'position': 'Midfielder', 'team_code': 1}]),
            # Add fallback fixtures for GW2 (future)
            'fpl_api_fixtures': pd.DataFrame([{
                'event': 2, 'team_h': 1, 'team_a': 3, 
                'team_h_difficulty': 2, 'team_a_difficulty': 4
            }])
        }
        
        # Train (mocking validation split to avoid issues with tiny data)
        # We just need it to be "trained" so predict works
        predictor.is_trained = True
        predictor.fallback_models = {'is_trained': True} # Mock fallback model
        
        # Mock _predict_with_stack to avoid actual prediction logic failure on tiny data
        predictor._predict_with_stack = lambda model, X: [4.5] * len(X)
        
        # Call predict for GW1 (predicting GW2)
        # IMPORTANT: all_gw_data ONLY has GW1. GW2 info comes from fpl_api_fixtures in season_data
        predictions = predictor.predict(
            all_gw_data, 
            fpl_core_season_data, 
            player_ids=[1], 
            current_gw=1
        )
        
        # Should get a prediction
        self.assertIn(1, predictions)
        self.assertEqual(predictions[1], 4.5)
        
        # Verify cache was built for GW2 using fallback
        self.assertIn((1, 2), predictor._fixture_cache, "GW2 fixture for team 1 should be cached via fallback")
        self.assertIn((3, 2), predictor._fixture_cache, "GW2 fixture for team 3 should be cached via fallback")

    def test_latest_available_fplcore_gw_falls_back_from_empty_current_gw(self):
        """When current GW data is empty, should use latest non-empty GW."""
        all_gw_data = {
            26: {'player_gameweek_stats': pd.DataFrame({'id': [1], 'gw': [26]})},
            27: {'player_gameweek_stats': pd.DataFrame({'id': [1], 'gw': [27]})},
            28: {'player_gameweek_stats': pd.DataFrame(columns=['id', 'gw'])},  # In-progress/empty
        }

        effective_gw = gfr.get_latest_available_fplcore_gw(all_gw_data, requested_gw=28)
        self.assertEqual(effective_gw, 27)

    def test_latest_available_fplcore_gw_requires_fully_finished_fixtures(self):
        """Should skip in-progress GW even when stats file is non-empty."""
        all_gw_data = {
            27: {
                'player_gameweek_stats': pd.DataFrame({'id': [1], 'gw': [27]}),
                'fixtures': pd.DataFrame({'finished': [True, True, True]})
            },
            28: {
                'player_gameweek_stats': pd.DataFrame({'id': [1], 'gw': [28]}),  # partial updates landed
                'fixtures': pd.DataFrame({'finished': [True, False, False]})      # GW still in progress
            },
        }

        effective_gw = gfr.get_latest_available_fplcore_gw(
            all_gw_data,
            requested_gw=28,
            require_finished_fixtures=True
        )
        self.assertEqual(effective_gw, 27)

    def test_clinical_chart_deletes_stale_gw_plot_when_no_gw_data(self):
        """Stale GW image must be removed if GW chart is not regenerated."""
        stale_gw_plot = self.temp_dir / 'clinical_wasteful_gw.png'
        stale_gw_plot.write_bytes(b'stale')

        fpl_core_season_data = {
            'playerstats': pd.DataFrame([{
                'id': 1,
                'web_name': 'Player1',
                'first_name': 'P',
                'second_name': 'One',
                'gw': 27,
                'minutes': 900,
                'goals_scored': 10,
                'assists': 2,
                'expected_goals': 7.0,
                'expected_assists': 1.2
            }]),
            'players': pd.DataFrame([{
                'player_id': 1,
                'position': 'Forward'
            }])
        }
        # Empty current GW dataset (e.g., GW still in progress)
        fpl_core_gw_data = {
            'player_gameweek_stats': pd.DataFrame(columns=[
                'id', 'gw', 'goals_scored', 'assists', 'expected_goals', 'expected_assists', 'minutes'
            ])
        }

        season_file, gw_file = self.plot_gen.generate_clinical_wasteful_chart(
            fpl_core_season_data=fpl_core_season_data,
            fpl_core_gw_data=fpl_core_gw_data,
            squad_ids=[1],
            current_gw=28
        )

        self.assertEqual(season_file, 'clinical_wasteful_season.png')
        self.assertIsNone(gw_file)
        self.assertTrue((self.temp_dir / 'clinical_wasteful_season.png').exists())
        self.assertFalse(stale_gw_plot.exists(), "Stale GW plot should have been deleted")

if __name__ == '__main__':
    unittest.main()
