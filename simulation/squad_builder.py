"""Initial squad builder for FPL simulation.

This module builds the starting GW1 squad using a hybrid template approach:
2 premium players + enablers, with the model filling positions optimally.
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from simulation.state import PlayerState, STARTING_BUDGET, SQUAD_SIZE
from simulation.data_adapter import HistoricalDataAdapter


class InitialSquadBuilder:
    """Builds the initial GW1 squad using hybrid template approach.

    Strategy: 2 premiums (1 MID, 1 FWD) + enablers
    - Pick highest xP players within each archetype
    - Respect budget, position quotas, and max 3 per team
    """

    # Position quotas
    POSITION_QUOTAS = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}

    # Archetype definitions for hybrid template
    # Note: Premiums are picked in Phase 1, enablers in Phase 2, rest filled in Phase 3
    ARCHETYPES = {
        # Premium slots - highest value, best performers (picked first)
        'premium_mid': {
            'position': 'MID',
            'min_price': 10.0,
            'count': 1,
            'priority': 1,
        },
        'premium_fwd': {
            'position': 'FWD',
            'min_price': 9.0,  # Lowered to ensure we get a premium FWD
            'count': 1,
            'priority': 2,
        },
        # Enablers - cheapest playing options (picked in Phase 2)
        'enabler_gkp_1': {
            'position': 'GKP',
            'max_price': 5.0,
            'min_price': 4.0,
            'count': 1,
            'priority': 10,
        },
        'enabler_gkp_2': {
            'position': 'GKP',
            'max_price': 5.0,
            'min_price': 4.0,
            'count': 1,
            'priority': 11,
        },
        'enabler_fwd_1': {
            'position': 'FWD',
            'max_price': 6.0,
            'min_price': 4.5,
            'count': 1,
            'priority': 9,
        },
    }

    MAX_PER_TEAM = 3

    def __init__(self, data_adapter: HistoricalDataAdapter):
        """Initialize squad builder.

        Args:
            data_adapter: Historical data adapter for player data
        """
        self.data = data_adapter

    def build_starting_squad(self, gw: int = 1) -> Tuple[List[PlayerState], float]:
        """Build initial squad for GW1.

        Uses hybrid template: 2 premiums + enablers, fill rest with best value.

        Args:
            gw: Gameweek to use for player data (usually 1)

        Returns:
            Tuple of (squad list, remaining bank)
        """
        all_players = self.data.get_all_players_in_gw(gw)

        # IMPORTANT (no leakage): for GW1 initialization we must not filter on
        # realized minutes/points. Use only information available pre-deadline.
        available = [p for p in all_players if p.get('price', 0) > 0]

        # Sort by xP for selection
        available.sort(key=lambda p: p.get('xp', 0), reverse=True)

        squad = []
        remaining_budget = STARTING_BUDGET
        team_counts: Dict[str, int] = defaultdict(int)
        position_counts: Dict[str, int] = defaultdict(int)
        used_ids = set()

        # Phase 1: Pick premiums (highest priority)
        for archetype_name, spec in sorted(
            self.ARCHETYPES.items(),
            key=lambda x: x[1].get('priority', 99)
        ):
            if spec['position'] not in ['MID', 'FWD']:
                continue  # Skip non-premium archetypes in phase 1
            if 'min_price' not in spec or spec.get('min_price', 0) < 10.0:
                continue

            candidates = [
                p for p in available
                if (
                    p['id'] not in used_ids and
                    p['position'] == spec['position'] and
                    p['price'] >= spec.get('min_price', 0) and
                    (spec.get('max_price') is None or p['price'] <= spec['max_price']) and
                    team_counts[p['team']] < self.MAX_PER_TEAM and
                    p['price'] <= remaining_budget
                )
            ]

            if not candidates:
                continue

            # Pick highest xP within archetype
            best = max(candidates, key=lambda p: p.get('xp', 0))
            squad.append(self._create_player_state(best, gw))
            remaining_budget -= best['price']
            team_counts[best['team']] += 1
            position_counts[best['position']] += 1
            used_ids.add(best['id'])

        # Phase 2: Pick enablers
        for archetype_name, spec in sorted(
            self.ARCHETYPES.items(),
            key=lambda x: x[1].get('priority', 99)
        ):
            if spec.get('min_price', 0) >= 10.0:
                continue  # Skip premiums

            for _ in range(spec.get('count', 1)):
                if position_counts[spec['position']] >= self.POSITION_QUOTAS[spec['position']]:
                    continue

                candidates = [
                    p for p in available
                    if (
                        p['id'] not in used_ids and
                        p['position'] == spec['position'] and
                        p['price'] >= spec.get('min_price', 0) and
                        (spec.get('max_price') is None or p['price'] <= spec['max_price']) and
                        team_counts[p['team']] < self.MAX_PER_TEAM and
                        p['price'] <= remaining_budget
                    )
                ]

                if not candidates:
                    continue

                # For enablers, pick cheapest that played (or highest xP if same price)
                candidates.sort(key=lambda p: (p['price'], -p.get('xp', 0)))
                best = candidates[0]

                squad.append(self._create_player_state(best, gw))
                remaining_budget -= best['price']
                team_counts[best['team']] += 1
                position_counts[best['position']] += 1
                used_ids.add(best['id'])

        # Phase 3: Fill remaining positions with best value picks
        for position, quota in self.POSITION_QUOTAS.items():
            while position_counts[position] < quota:
                needed = quota - position_counts[position]

                candidates = [
                    p for p in available
                    if (
                        p['id'] not in used_ids and
                        p['position'] == position and
                        team_counts[p['team']] < self.MAX_PER_TEAM and
                        p['price'] <= remaining_budget
                    )
                ]

                if not candidates:
                    # Try with lower price threshold if budget tight
                    break

                # Pick best value (xP per cost), but prioritize actually playing
                def value_score(p):
                    xp = p.get('xp', 0)
                    price = p['price'] if p['price'] > 0 else 5.0
                    return (xp / price, xp)

                candidates.sort(key=value_score, reverse=True)
                best = candidates[0]

                squad.append(self._create_player_state(best, gw))
                remaining_budget -= best['price']
                team_counts[best['team']] += 1
                position_counts[position] += 1
                used_ids.add(best['id'])

        # Verify squad is valid
        if len(squad) != SQUAD_SIZE:
            print(f"Warning: Squad has {len(squad)} players, expected {SQUAD_SIZE}")

        # Sort squad by position for consistent ordering
        position_order = {'GKP': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
        squad.sort(key=lambda p: (position_order.get(p.position, 99), -p.current_price))

        return squad, remaining_budget

    def _create_player_state(self, player_dict: Dict, gw: int) -> PlayerState:
        """Create PlayerState from player dict.

        Args:
            player_dict: Dict from data adapter
            gw: Gameweek for price lookup

        Returns:
            PlayerState object
        """
        # Get team ID from name lookup
        team_name = player_dict.get('team', '')
        team_id = 0
        for tid, tname in self.data.team_lookup.items():
            if tname == team_name:
                team_id = tid
                break

        price = player_dict.get('price', 5.0)

        return PlayerState(
            id=player_dict['id'],
            name=player_dict.get('name', f"Player_{player_dict['id']}"),
            position=player_dict.get('position', 'MID'),
            team_id=team_id,
            team_name=team_name,
            purchase_price=price,
            current_price=price,
        )

    def validate_squad(self, squad: List[PlayerState]) -> Tuple[bool, List[str]]:
        """Validate a squad meets all FPL constraints.

        Args:
            squad: List of PlayerState objects

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check size
        if len(squad) != SQUAD_SIZE:
            errors.append(f"Squad has {len(squad)} players, expected {SQUAD_SIZE}")

        # Check position quotas
        position_counts = defaultdict(int)
        for player in squad:
            position_counts[player.position] += 1

        for pos, quota in self.POSITION_QUOTAS.items():
            if position_counts[pos] != quota:
                errors.append(
                    f"Position {pos}: {position_counts[pos]} players, expected {quota}"
                )

        # Check max per team
        team_counts = defaultdict(int)
        for player in squad:
            team_counts[player.team_name] += 1

        for team, count in team_counts.items():
            if count > self.MAX_PER_TEAM:
                errors.append(f"Team {team}: {count} players, max {self.MAX_PER_TEAM}")

        # Check for duplicates
        ids = [p.id for p in squad]
        if len(ids) != len(set(ids)):
            errors.append("Squad contains duplicate players")

        return len(errors) == 0, errors

    def get_squad_summary(self, squad: List[PlayerState], bank: float) -> Dict:
        """Get summary statistics for a squad.

        Args:
            squad: List of PlayerState objects
            bank: Money in bank

        Returns:
            Dict with squad statistics
        """
        total_value = sum(p.current_price for p in squad)

        position_counts = defaultdict(int)
        position_values = defaultdict(float)
        for player in squad:
            position_counts[player.position] += 1
            position_values[player.position] += player.current_price

        team_counts = defaultdict(int)
        for player in squad:
            team_counts[player.team_name] += 1

        return {
            'total_value': total_value,
            'bank': bank,
            'total_budget_used': total_value,
            'position_counts': dict(position_counts),
            'position_values': dict(position_values),
            'team_counts': dict(team_counts),
            'player_count': len(squad),
        }
