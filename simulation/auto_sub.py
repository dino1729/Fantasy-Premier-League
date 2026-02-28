"""FPL auto-substitution simulator.

This module implements the official FPL automatic substitution rules
for when starting players don't play.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from simulation.state import PlayerState, AutoSubRecord


@dataclass
class FormationConstraints:
    """FPL formation constraints."""
    MIN_GKP: int = 1
    MAX_GKP: int = 1
    MIN_DEF: int = 3
    MAX_DEF: int = 5
    MIN_MID: int = 2
    MAX_MID: int = 5
    MIN_FWD: int = 1
    MAX_FWD: int = 3


class AutoSubSimulator:
    """Simulates FPL automatic substitution rules.

    FPL auto-sub rules:
    1. Substitutes come on in bench order (position 1-4)
    2. The resulting formation must be valid (3-5 DEF, 2-5 MID, 1-3 FWD)
    3. Exactly 1 GKP must be on the field
    4. If captain doesn't play, vice-captain gets 2x points
    """

    def __init__(self):
        self.constraints = FormationConstraints()

    def apply_auto_subs(
        self,
        lineup: List[PlayerState],
        bench: List[PlayerState],
        actual_minutes: Dict[int, int],
        captain_id: int,
        vice_captain_id: int,
    ) -> Tuple[List[PlayerState], List[AutoSubRecord], int, int]:
        """Apply automatic substitutions based on minutes played.

        Args:
            lineup: List of 11 starting players (in playing order)
            bench: List of 4 bench players (in priority order)
            actual_minutes: Dict mapping player_id to minutes played
            captain_id: Selected captain's player ID
            vice_captain_id: Selected vice-captain's player ID

        Returns:
            Tuple of:
            - final_xi: List of 11 players after auto-subs
            - auto_subs: List of AutoSubRecord documenting changes
            - effective_captain_id: Actual captain (may change if C didn't play)
            - effective_vice_captain_id: Vice captain ID
        """
        # Identify starters who didn't play
        non_players = [
            p for p in lineup
            if actual_minutes.get(p.id, 0) == 0
        ]
        players_in = [
            p for p in lineup
            if actual_minutes.get(p.id, 0) > 0
        ]

        auto_subs = []
        bench_used = []
        remaining_bench = list(bench)

        # Track current formation
        formation = self._count_formation(players_in)

        # Process each non-player in original lineup order
        for non_player in non_players:
            # Try each bench player in priority order
            for bench_player in remaining_bench:
                # Skip bench players who also didn't play
                if actual_minutes.get(bench_player.id, 0) == 0:
                    continue

                # Check if this substitution maintains valid formation
                if self._can_substitute(formation, non_player.position, bench_player.position):
                    # Make the substitution
                    auto_subs.append(AutoSubRecord(
                        player_out_id=non_player.id,
                        player_out_name=non_player.name,
                        player_in_id=bench_player.id,
                        player_in_name=bench_player.name,
                        reason='non_player',
                    ))

                    # Update formation
                    formation = self._update_formation(
                        formation, non_player.position, bench_player.position
                    )

                    bench_used.append(bench_player)
                    remaining_bench.remove(bench_player)
                    break

        # Build final XI
        final_xi = players_in + bench_used

        # Handle captain substitution
        effective_captain = captain_id
        effective_vice = vice_captain_id

        # If captain didn't play, check if they're in final XI
        captain_in_final = any(p.id == captain_id for p in final_xi)
        captain_played = actual_minutes.get(captain_id, 0) > 0

        if not captain_played:
            # Captain didn't play - vice captain gets double points
            effective_captain = vice_captain_id

            # Check if vice captain is in final XI and played
            vice_played = actual_minutes.get(vice_captain_id, 0) > 0
            if not vice_played:
                # Both didn't play - no double points awarded
                effective_captain = 0
                effective_vice = 0

        return final_xi, auto_subs, effective_captain, effective_vice

    def _count_formation(self, players: List[PlayerState]) -> Dict[str, int]:
        """Count players by position.

        Args:
            players: List of players

        Returns:
            Dict with position counts
        """
        formation = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        for player in players:
            pos = player.position
            if pos in formation:
                formation[pos] += 1
        return formation

    def _can_substitute(
        self, current_formation: Dict[str, int], pos_out: str, pos_in: str
    ) -> bool:
        """Check if a substitution maintains valid formation.

        Args:
            current_formation: Current position counts
            pos_out: Position of player going out
            pos_in: Position of player coming in

        Returns:
            True if resulting formation is valid
        """
        # Simulate the change
        new_formation = current_formation.copy()
        new_formation[pos_out] = new_formation.get(pos_out, 0) - 1
        new_formation[pos_in] = new_formation.get(pos_in, 0) + 1

        # Check constraints
        if new_formation.get('GKP', 0) != self.constraints.MIN_GKP:
            # Must have exactly 1 GKP
            # Exception: if GKP going out and GKP coming in, that's fine
            if pos_out == 'GKP' and pos_in == 'GKP':
                pass  # OK
            elif pos_out == 'GKP' or pos_in == 'GKP':
                # GKP can only be subbed for GKP
                return False

        if not (self.constraints.MIN_DEF <= new_formation.get('DEF', 0) <= self.constraints.MAX_DEF):
            return False

        if not (self.constraints.MIN_MID <= new_formation.get('MID', 0) <= self.constraints.MAX_MID):
            return False

        if not (self.constraints.MIN_FWD <= new_formation.get('FWD', 0) <= self.constraints.MAX_FWD):
            return False

        return True

    def _update_formation(
        self, formation: Dict[str, int], pos_out: str, pos_in: str
    ) -> Dict[str, int]:
        """Update formation counts after a substitution.

        Args:
            formation: Current formation counts
            pos_out: Position going out
            pos_in: Position coming in

        Returns:
            Updated formation counts
        """
        new_formation = formation.copy()
        new_formation[pos_out] = new_formation.get(pos_out, 0) - 1
        new_formation[pos_in] = new_formation.get(pos_in, 0) + 1
        return new_formation

    def order_bench(
        self, bench_players: List[PlayerState], priorities: Optional[Dict[int, int]] = None
    ) -> List[PlayerState]:
        """Order bench players by priority.

        Default order: GKP last (position 4), outfield by some heuristic.

        Args:
            bench_players: List of 4 bench players
            priorities: Optional dict mapping player_id to priority (1-4)

        Returns:
            Ordered bench list
        """
        if priorities:
            # Use explicit priorities
            return sorted(
                bench_players,
                key=lambda p: priorities.get(p.id, 99)
            )

        # Default ordering: outfield players first (by position), GKP last
        gkp = [p for p in bench_players if p.position == 'GKP']
        outfield = [p for p in bench_players if p.position != 'GKP']

        # Sort outfield by position priority (DEF > MID > FWD for auto-sub flexibility)
        position_priority = {'DEF': 1, 'MID': 2, 'FWD': 3}
        outfield.sort(key=lambda p: position_priority.get(p.position, 99))

        return outfield + gkp

    def calculate_bench_points(
        self,
        bench: List[PlayerState],
        actual_points: Dict[int, int],
        bench_boost_active: bool = False,
    ) -> int:
        """Calculate points from bench players.

        Args:
            bench: List of bench players
            actual_points: Dict mapping player_id to points
            bench_boost_active: If True, all bench points count

        Returns:
            Total bench points (0 unless bench boost)
        """
        if not bench_boost_active:
            return 0

        return sum(actual_points.get(p.id, 0) for p in bench)

    def calculate_remaining_bench_points(
        self,
        bench: List[PlayerState],
        actual_points: Dict[int, int],
        auto_subs: Optional[List[AutoSubRecord]] = None,
    ) -> int:
        """Calculate points for the bench players that remained on the bench.

        This excludes any bench players who were auto-subbed into the XI, to avoid
        double-counting (especially important when Bench Boost is active).
        """
        subbed_in_ids = {s.player_in_id for s in (auto_subs or [])}
        remaining = [p for p in bench if p.id not in subbed_in_ids]
        return sum(actual_points.get(p.id, 0) for p in remaining)

    def calculate_gw_points(
        self,
        final_xi: List[PlayerState],
        actual_points: Dict[int, int],
        effective_captain_id: int,
        chip_used: Optional[str] = None,
        hit_cost: int = 0,
    ) -> Tuple[int, int, int]:
        """Calculate total gameweek points.

        Args:
            final_xi: Final 11 players (after auto-subs)
            actual_points: Dict mapping player_id to points
            effective_captain_id: Captain whose points are doubled
            chip_used: Active chip (if any)
            hit_cost: Points deducted for transfer hits

        Returns:
            Tuple of (total_points, points_before_hits, captain_points)
        """
        captain_multiplier = 3 if chip_used == 'triple_captain' else 2

        total = 0
        captain_pts = 0

        for player in final_xi:
            pts = actual_points.get(player.id, 0)

            if effective_captain_id and player.id == effective_captain_id:
                captain_pts = pts
                pts *= captain_multiplier

            total += pts

        points_before_hits = total
        total -= hit_cost

        return total, points_before_hits, captain_pts
