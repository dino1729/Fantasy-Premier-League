import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from simulation.data_adapter import HistoricalDataAdapter
from simulation.engine import SimulationEngine
from simulation.state import PlayerState


class TestSimulationLeakageGuards(unittest.TestCase):
    def _write_gw(self, gws_dir: Path, gw: int, rows):
        df = pd.DataFrame(rows)
        df.to_csv(gws_dir / f'gw{gw}.csv', index=False)

    def test_build_players_df_for_solver_uses_prior_gw_totals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            season_path = Path(tmpdir) / 'season'
            gws_dir = season_path / 'gws'
            gws_dir.mkdir(parents=True, exist_ok=True)

            # Teams mapping
            pd.DataFrame(
                [{'id': 1, 'name': 'TeamA'}, {'id': 2, 'name': 'TeamB'}]
            ).to_csv(season_path / 'teams.csv', index=False)

            # Fixtures (not used in this specific test, but expected to exist sometimes)
            pd.DataFrame(
                [
                    {'id': 1, 'event': 1, 'team_h': 1, 'team_a': 2, 'team_h_difficulty': 3, 'team_a_difficulty': 3, 'finished': True},
                    {'id': 2, 'event': 2, 'team_h': 2, 'team_a': 1, 'team_h_difficulty': 3, 'team_a_difficulty': 3, 'finished': False},
                ]
            ).to_csv(season_path / 'fixtures.csv', index=False)

            # GW1 and GW2 snapshots include GW-level outcomes (total_points/minutes).
            # For GW2 decisions, we must only use cumulative stats through GW1.
            self._write_gw(
                gws_dir,
                1,
                [
                    {'element': 1, 'name': 'P1', 'position': 'MID', 'team': 'TeamA', 'value': 100, 'total_points': 10, 'minutes': 90, 'xP': 5.0},
                    {'element': 2, 'name': 'P2', 'position': 'FWD', 'team': 'TeamB', 'value': 90, 'total_points': 2, 'minutes': 90, 'xP': 4.0},
                ],
            )
            self._write_gw(
                gws_dir,
                2,
                [
                    {'element': 1, 'name': 'P1', 'position': 'MID', 'team': 'TeamA', 'value': 100, 'total_points': 15, 'minutes': 90, 'xP': 5.0},
                    {'element': 2, 'name': 'P2', 'position': 'FWD', 'team': 'TeamB', 'value': 90, 'total_points': 0, 'minutes': 0, 'xP': 4.0},
                ],
            )

            adapter = HistoricalDataAdapter(season_path=season_path)
            players_df = adapter.build_players_df_for_solver(as_of_gw=2)

            p1 = players_df[players_df['id'] == 1].iloc[0]
            p2 = players_df[players_df['id'] == 2].iloc[0]

            self.assertEqual(int(p1['total_points']), 10)
            self.assertEqual(int(p1['minutes']), 90)
            self.assertEqual(int(p2['total_points']), 2)
            self.assertEqual(int(p2['minutes']), 90)

    def test_xp_matrix_does_not_read_future_gw_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            season_path = Path(tmpdir) / 'season'
            gws_dir = season_path / 'gws'
            gws_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [{'id': 1, 'name': 'TeamA'}, {'id': 2, 'name': 'TeamB'}]
            ).to_csv(season_path / 'teams.csv', index=False)

            # Ensure a fixture exists in GW2 and GW3 so forecasts are non-zero.
            pd.DataFrame(
                [
                    {'id': 1, 'event': 2, 'team_h': 1, 'team_a': 2, 'team_h_difficulty': 3, 'team_a_difficulty': 3, 'finished': False},
                    {'id': 2, 'event': 3, 'team_h': 1, 'team_a': 2, 'team_h_difficulty': 3, 'team_a_difficulty': 3, 'finished': False},
                ]
            ).to_csv(season_path / 'fixtures.csv', index=False)

            # GW2 baseline xP is 5.0; GW3 file contains an intentionally huge xP (leak bait).
            self._write_gw(
                gws_dir,
                2,
                [{'element': 1, 'name': 'P1', 'position': 'MID', 'team': 'TeamA', 'value': 100, 'total_points': 0, 'minutes': 0, 'xP': 5.0}],
            )
            self._write_gw(
                gws_dir,
                3,
                [{'element': 1, 'name': 'P1', 'position': 'MID', 'team': 'TeamA', 'value': 100, 'total_points': 99, 'minutes': 90, 'xP': 999.0}],
            )

            adapter = HistoricalDataAdapter(season_path=season_path)
            engine = SimulationEngine(adapter, use_future_gw_xp=False)

            squad = [PlayerState(id=1, name='P1', position='MID', team_id=1, team_name='TeamA', purchase_price=10.0, current_price=10.0)]
            state = SimpleNamespace(squad=squad)

            xp_matrix = engine._build_xp_matrix(state, current_gw=2, horizon=2, candidate_ids=[1])

            # GW3 forecast must not equal the GW3 snapshot's xP (999.0).
            self.assertIn(1, xp_matrix)
            self.assertLess(xp_matrix[1][1], 50.0)
            self.assertAlmostEqual(xp_matrix[1][0], 5.0, places=6)
            self.assertAlmostEqual(xp_matrix[1][1], 5.0, places=6)


if __name__ == '__main__':
    unittest.main()

