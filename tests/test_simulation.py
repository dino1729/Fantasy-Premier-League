"""Unit tests for FPL simulation components."""

import unittest
import tempfile
from pathlib import Path

from simulation.state import (
    PlayerState,
    GameweekState,
    GameweekDecisions,
    GameweekResults,
    TransferRecord,
    AutoSubRecord,
    INITIAL_CHIPS,
)
from simulation.auto_sub import AutoSubSimulator


class TestPlayerState(unittest.TestCase):
    """Tests for PlayerState dataclass."""

    def test_create_player(self):
        player = PlayerState(
            id=328,
            name="Mohamed Salah",
            position="MID",
            team_id=14,
            team_name="Liverpool",
            purchase_price=12.5,
            current_price=13.0,
        )
        self.assertEqual(player.id, 328)
        self.assertEqual(player.position, "MID")
        self.assertEqual(player.purchase_price, 12.5)

    def test_to_dict_roundtrip(self):
        player = PlayerState(
            id=328,
            name="Mohamed Salah",
            position="MID",
            team_id=14,
            team_name="Liverpool",
            purchase_price=12.5,
            current_price=13.0,
        )
        player_dict = player.to_dict()
        restored = PlayerState.from_dict(player_dict)

        self.assertEqual(player.id, restored.id)
        self.assertEqual(player.name, restored.name)
        self.assertEqual(player.purchase_price, restored.purchase_price)


class TestGameweekState(unittest.TestCase):
    """Tests for GameweekState serialization."""

    def setUp(self):
        self.squad = [
            PlayerState(i, f"Player{i}", "MID", 1, "Team", 5.0, 5.0)
            for i in range(15)
        ]
        self.state = GameweekState(
            gameweek=10,
            squad=self.squad,
            bank=2.5,
            free_transfers=2,
            total_points=500,
            chips_available=set(INITIAL_CHIPS),
            decisions=GameweekDecisions(
                transfers=[],
                lineup=list(range(11)),
                bench_order=list(range(11, 15)),
                captain_id=0,
                vice_captain_id=1,
            ),
            results=GameweekResults(
                gw_points=60,
                gw_points_before_hits=60,
                hit_cost=0,
            ),
        )

    def test_to_json_roundtrip(self):
        json_str = self.state.to_json()
        restored = GameweekState.from_json(json_str)

        self.assertEqual(self.state.gameweek, restored.gameweek)
        self.assertEqual(self.state.bank, restored.bank)
        self.assertEqual(self.state.total_points, restored.total_points)
        self.assertEqual(len(self.state.squad), len(restored.squad))

    def test_checkpoint_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = self.state.save_checkpoint(Path(tmpdir))
            self.assertTrue(checkpoint_path.exists())

            loaded = GameweekState.load_checkpoint(checkpoint_path)
            self.assertEqual(self.state.gameweek, loaded.gameweek)
            self.assertEqual(self.state.total_points, loaded.total_points)

    def test_get_squad_value(self):
        value = self.state.get_squad_value()
        self.assertEqual(value, 15 * 5.0)  # 15 players at 5.0 each

    def test_get_total_value(self):
        total = self.state.get_total_value()
        self.assertEqual(total, 15 * 5.0 + 2.5)  # Squad + bank


class TestAutoSubSimulator(unittest.TestCase):
    """Tests for auto-substitution logic."""

    def setUp(self):
        self.sim = AutoSubSimulator()

        # Create test players
        self.gkp1 = PlayerState(1, "GKP1", "GKP", 1, "T1", 5.0, 5.0)
        self.gkp2 = PlayerState(2, "GKP2", "GKP", 2, "T2", 4.0, 4.0)
        self.def1 = PlayerState(3, "DEF1", "DEF", 1, "T1", 5.5, 5.5)
        self.def2 = PlayerState(4, "DEF2", "DEF", 2, "T2", 5.0, 5.0)
        self.def3 = PlayerState(5, "DEF3", "DEF", 3, "T3", 4.5, 4.5)
        self.def4 = PlayerState(6, "DEF4", "DEF", 4, "T4", 4.0, 4.0)
        self.mid1 = PlayerState(7, "MID1", "MID", 1, "T1", 8.0, 8.0)
        self.mid2 = PlayerState(8, "MID2", "MID", 2, "T2", 7.0, 7.0)
        self.mid3 = PlayerState(9, "MID3", "MID", 3, "T3", 6.0, 6.0)
        self.mid4 = PlayerState(10, "MID4", "MID", 4, "T4", 5.0, 5.0)
        self.fwd1 = PlayerState(11, "FWD1", "FWD", 1, "T1", 9.0, 9.0)
        self.fwd2 = PlayerState(12, "FWD2", "FWD", 2, "T2", 6.0, 6.0)
        self.fwd3 = PlayerState(13, "FWD3", "FWD", 3, "T3", 5.0, 5.0)

    def test_no_subs_needed(self):
        """Test when all starters play."""
        lineup = [self.gkp1, self.def1, self.def2, self.def3,
                  self.mid1, self.mid2, self.mid3, self.mid4,
                  self.fwd1, self.fwd2, self.fwd3]
        bench = [self.gkp2, self.def4]

        actual_minutes = {p.id: 90 for p in lineup + bench}

        final_xi, auto_subs, captain, _ = self.sim.apply_auto_subs(
            lineup, bench, actual_minutes, self.mid1.id, self.mid2.id
        )

        self.assertEqual(len(final_xi), 11)
        self.assertEqual(len(auto_subs), 0)
        self.assertEqual(captain, self.mid1.id)

    def test_simple_sub(self):
        """Test simple substitution when one starter doesn't play."""
        lineup = [self.gkp1, self.def1, self.def2, self.def3,
                  self.mid1, self.mid2, self.mid3, self.mid4,
                  self.fwd1, self.fwd2, self.fwd3]
        bench = [self.gkp2, self.def4]

        # FWD3 doesn't play
        actual_minutes = {p.id: 90 for p in lineup + bench}
        actual_minutes[self.fwd3.id] = 0

        final_xi, auto_subs, captain, _ = self.sim.apply_auto_subs(
            lineup, bench, actual_minutes, self.mid1.id, self.mid2.id
        )

        self.assertEqual(len(final_xi), 11)
        self.assertEqual(len(auto_subs), 1)
        self.assertEqual(auto_subs[0].player_out_id, self.fwd3.id)
        self.assertEqual(auto_subs[0].player_in_id, self.def4.id)

    def test_formation_constraint(self):
        """Test that formation constraints are respected."""
        # 3-4-3 formation
        lineup = [self.gkp1, self.def1, self.def2, self.def3,
                  self.mid1, self.mid2, self.mid3, self.mid4,
                  self.fwd1, self.fwd2, self.fwd3]

        # Only FWD on bench - can't sub in for MID without breaking formation
        bench = [self.gkp2, self.fwd3]

        # MID4 doesn't play, but we only have FWD on bench
        actual_minutes = {p.id: 90 for p in lineup}
        actual_minutes[self.mid4.id] = 0
        actual_minutes[self.gkp2.id] = 90
        actual_minutes[self.fwd3.id] = 90

        final_xi, auto_subs, _, _ = self.sim.apply_auto_subs(
            lineup, bench, actual_minutes, self.mid1.id, self.mid2.id
        )

        # FWD can't replace MID in a 3-4-3 (would become 3-3-4, invalid)
        # So no sub should happen
        self.assertEqual(len(auto_subs), 0)

    def test_captain_transfer(self):
        """Test captain transfers to vice when captain doesn't play."""
        lineup = [self.gkp1, self.def1, self.def2, self.def3,
                  self.mid1, self.mid2, self.mid3, self.mid4,
                  self.fwd1, self.fwd2, self.fwd3]
        bench = [self.gkp2, self.def4]

        actual_minutes = {p.id: 90 for p in lineup + bench}
        actual_minutes[self.mid1.id] = 0  # Captain doesn't play

        _, _, effective_captain, _ = self.sim.apply_auto_subs(
            lineup, bench, actual_minutes, self.mid1.id, self.mid2.id
        )

        # Vice captain should become effective captain
        self.assertEqual(effective_captain, self.mid2.id)

    def test_no_captain_multiplier_if_captain_and_vice_do_not_play(self):
        lineup = [self.gkp1, self.def1, self.def2, self.def3,
                  self.mid1, self.mid2, self.mid3, self.mid4,
                  self.fwd1, self.fwd2, self.fwd3]
        bench = [self.gkp2, self.def4]

        actual_minutes = {p.id: 90 for p in lineup + bench}
        actual_minutes[self.mid1.id] = 0  # captain doesn't play
        actual_minutes[self.mid2.id] = 0  # vice doesn't play

        # Ensure bench can cover at least one slot so XI still has 11
        final_xi, auto_subs, effective_captain, _ = self.sim.apply_auto_subs(
            lineup, bench, actual_minutes, self.mid1.id, self.mid2.id
        )

        self.assertEqual(effective_captain, 0)
        self.assertTrue(len(final_xi) <= 11)

        actual_points = {p.id: 2 for p in lineup + bench}
        total, _, _ = self.sim.calculate_gw_points(
            final_xi, actual_points, effective_captain_id=effective_captain, chip_used=None, hit_cost=0
        )

        # No captain multiplier; just sum of XI points.
        self.assertEqual(total, sum(actual_points[p.id] for p in final_xi))

    def test_calculate_gw_points(self):
        """Test points calculation with captain bonus."""
        lineup = [self.gkp1, self.def1, self.mid1]
        actual_points = {
            self.gkp1.id: 6,
            self.def1.id: 8,
            self.mid1.id: 10,
        }

        total, before_hits, captain_pts = self.sim.calculate_gw_points(
            lineup, actual_points, self.mid1.id, chip_used=None, hit_cost=0
        )

        # GKP: 6, DEF: 8, MID (captain): 10*2 = 20
        self.assertEqual(total, 34)
        self.assertEqual(captain_pts, 10)

    def test_triple_captain_bonus(self):
        """Test triple captain chip multiplier."""
        lineup = [self.gkp1, self.mid1]
        actual_points = {
            self.gkp1.id: 6,
            self.mid1.id: 10,
        }

        total, _, _ = self.sim.calculate_gw_points(
            lineup, actual_points, self.mid1.id,
            chip_used='triple_captain', hit_cost=0
        )

        # GKP: 6, MID (TC): 10*3 = 30
        self.assertEqual(total, 36)

    def test_hit_cost_deduction(self):
        """Test hit cost is deducted from points."""
        lineup = [self.gkp1, self.mid1]
        actual_points = {
            self.gkp1.id: 6,
            self.mid1.id: 10,
        }

        total, before_hits, _ = self.sim.calculate_gw_points(
            lineup, actual_points, self.mid1.id,
            chip_used=None, hit_cost=8
        )

        # Before hits: 6 + 20 = 26
        # After hits: 26 - 8 = 18
        self.assertEqual(before_hits, 26)
        self.assertEqual(total, 18)

    def test_remaining_bench_points_excludes_auto_subbed_players(self):
        """Bench points should exclude players subbed into the XI (no double counting)."""
        # Lineup has 1 non-player MID4; DEF4 is first bench and will be subbed in.
        lineup = [self.gkp1, self.def1, self.def2, self.def3,
                  self.mid1, self.mid2, self.mid3, self.mid4,
                  self.fwd1, self.fwd2, self.fwd3]
        bench = [self.def4, self.gkp2]  # Keep it small for the test

        actual_minutes = {p.id: 90 for p in lineup + bench}
        actual_minutes[self.mid4.id] = 0  # non-player triggers sub

        actual_points = {p.id: 1 for p in lineup + bench}
        actual_points[self.def4.id] = 6  # bench player that comes in
        actual_points[self.gkp2.id] = 5  # bench player that stays on bench

        final_xi, auto_subs, effective_captain, _ = self.sim.apply_auto_subs(
            lineup, bench, actual_minutes, self.mid1.id, self.mid2.id
        )
        gw_points, _, _ = self.sim.calculate_gw_points(
            final_xi, actual_points, effective_captain, chip_used=None, hit_cost=0
        )

        remaining_bench_points = self.sim.calculate_remaining_bench_points(
            bench, actual_points, auto_subs=auto_subs
        )

        # DEF4 was auto-subbed into the XI so must not be counted as remaining bench.
        self.assertEqual(remaining_bench_points, 5)
        # Without bench boost, only XI counts (no bench points added).
        expected_xi_points = sum(actual_points[p.id] for p in final_xi)
        expected_xi_points += actual_points.get(effective_captain, 0)  # captain double
        self.assertEqual(gw_points, expected_xi_points)


class TestTransferRecord(unittest.TestCase):
    """Tests for TransferRecord."""

    def test_create_transfer(self):
        transfer = TransferRecord(
            player_out_id=100,
            player_out_name="Player Out",
            player_in_id=200,
            player_in_name="Player In",
            price_out=6.0,
            price_in=7.5,
            is_hit=True,
        )
        self.assertEqual(transfer.player_in_id, 200)
        self.assertTrue(transfer.is_hit)

    def test_to_dict_roundtrip(self):
        transfer = TransferRecord(
            player_out_id=100,
            player_out_name="Player Out",
            player_in_id=200,
            player_in_name="Player In",
            price_out=6.0,
            price_in=7.5,
            is_hit=False,
        )
        restored = TransferRecord.from_dict(transfer.to_dict())
        self.assertEqual(transfer.player_out_id, restored.player_out_id)
        self.assertEqual(transfer.price_in, restored.price_in)


if __name__ == '__main__':
    unittest.main()
