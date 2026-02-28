"""Risk-adjusted captain selection for FPL simulation.

This module selects captain and vice-captain with consideration
for both expected points and consistency.
"""

from typing import Dict, List, Tuple, Optional
from simulation.state import PlayerState
from simulation.data_adapter import HistoricalDataAdapter


class RiskAdjustedCaptainSelector:
    """Selects captain with risk/consistency adjustment.

    When two players have similar xP (within threshold), prefers
    the more consistent performer (lower variance in recent points).
    """

    # xP threshold for considering players "tied"
    XP_TIEBREAK_THRESHOLD = 0.5

    # Consistency window (GWs to look back)
    CONSISTENCY_WINDOW = 5

    def __init__(self, data_adapter: HistoricalDataAdapter):
        """Initialize captain selector.

        Args:
            data_adapter: Historical data adapter for consistency calculation
        """
        self.data = data_adapter

    def select_captain(
        self,
        candidates: List[Dict],
        gw: int,
        threshold: float = None,
    ) -> Tuple[int, int]:
        """Select captain and vice-captain with risk adjustment.

        Args:
            candidates: List of candidate dicts with 'id', 'xp', 'name', etc.
            gw: Current gameweek (for consistency calculation)
            threshold: xP threshold for tiebreaking (default: 0.5)

        Returns:
            Tuple of (captain_id, vice_captain_id)
        """
        if not candidates:
            raise ValueError("No captain candidates provided")

        if threshold is None:
            threshold = self.XP_TIEBREAK_THRESHOLD

        # Enrich candidates with consistency scores
        enriched = []
        for c in candidates:
            consistency = self.data.calculate_player_consistency(
                c['id'], gw, self.CONSISTENCY_WINDOW
            )
            enriched.append({
                **c,
                'consistency': consistency,
            })

        # Sort by xP descending
        sorted_candidates = sorted(enriched, key=lambda x: x['xp'], reverse=True)

        if len(sorted_candidates) == 1:
            return sorted_candidates[0]['id'], sorted_candidates[0]['id']

        top = sorted_candidates[0]
        second = sorted_candidates[1]

        # Check if within threshold - apply tiebreaker
        if abs(top['xp'] - second['xp']) < threshold:
            # Pick more consistent player
            if top['consistency'] < second['consistency']:
                captain = second
                vice = top
            else:
                captain = top
                vice = second
        else:
            captain = top
            vice = second

        return captain['id'], vice['id']

    def select_captain_from_squad(
        self,
        squad: List[PlayerState],
        xp_predictions: Dict[int, float],
        gw: int,
        lineup_ids: Optional[List[int]] = None,
    ) -> Tuple[int, int]:
        """Select captain from squad based on xP predictions.

        Args:
            squad: Full squad
            xp_predictions: Dict mapping player_id to predicted xP
            gw: Current gameweek
            lineup_ids: Optional list of starting XI IDs (only pick from these)

        Returns:
            Tuple of (captain_id, vice_captain_id)
        """
        # Filter to lineup if provided
        if lineup_ids:
            eligible = [p for p in squad if p.id in lineup_ids]
        else:
            eligible = squad

        # Build candidate list
        candidates = []
        for player in eligible:
            xp = xp_predictions.get(player.id, 0.0)
            candidates.append({
                'id': player.id,
                'name': player.name,
                'xp': xp,
                'position': player.position,
                'team_id': player.team_id,
                'price': player.current_price,
            })

        return self.select_captain(candidates, gw)

    def get_captain_candidates(
        self,
        squad: List[PlayerState],
        xp_predictions: Dict[int, float],
        gw: int,
        top_n: int = 5,
    ) -> List[Dict]:
        """Get top captain candidates with full analysis.

        Args:
            squad: Full squad
            xp_predictions: Dict mapping player_id to predicted xP
            gw: Current gameweek
            top_n: Number of candidates to return

        Returns:
            List of candidate dicts sorted by adjusted score
        """
        candidates = []

        for player in squad:
            xp = xp_predictions.get(player.id, 0.0)
            consistency = self.data.calculate_player_consistency(
                player.id, gw, self.CONSISTENCY_WINDOW
            )

            # Calculate risk-adjusted score
            # Higher consistency slightly boosts score
            adjusted_xp = xp * (1 + 0.1 * consistency)

            candidates.append({
                'id': player.id,
                'name': player.name,
                'position': player.position,
                'xp': xp,
                'consistency': consistency,
                'adjusted_xp': adjusted_xp,
                'team_id': player.team_id,
                'price': player.current_price,
            })

        # Sort by adjusted xP
        candidates.sort(key=lambda x: x['adjusted_xp'], reverse=True)

        return candidates[:top_n]

    def analyze_captain_decision(
        self,
        captain_id: int,
        vice_id: int,
        candidates: List[Dict],
    ) -> Dict:
        """Analyze and explain a captain decision.

        Args:
            captain_id: Selected captain ID
            vice_id: Selected vice-captain ID
            candidates: Full candidate list

        Returns:
            Dict with decision analysis
        """
        captain = next((c for c in candidates if c['id'] == captain_id), None)
        vice = next((c for c in candidates if c['id'] == vice_id), None)

        if not captain:
            return {'error': 'Captain not found in candidates'}

        analysis = {
            'captain': {
                'id': captain_id,
                'name': captain.get('name', ''),
                'xp': captain.get('xp', 0),
                'consistency': captain.get('consistency', 0),
            },
            'vice_captain': {
                'id': vice_id,
                'name': vice.get('name', '') if vice else '',
                'xp': vice.get('xp', 0) if vice else 0,
            },
            'tiebreaker_applied': False,
            'reason': 'Highest xP selection',
        }

        # Check if tiebreaker was applied
        if len(candidates) >= 2:
            sorted_by_xp = sorted(candidates, key=lambda x: x['xp'], reverse=True)
            top_xp = sorted_by_xp[0]
            second_xp = sorted_by_xp[1]

            if abs(top_xp['xp'] - second_xp['xp']) < self.XP_TIEBREAK_THRESHOLD:
                analysis['tiebreaker_applied'] = True
                if captain_id != top_xp['id']:
                    analysis['reason'] = (
                        f"Tiebreaker: {captain.get('name', '')} selected over "
                        f"{top_xp.get('name', '')} due to higher consistency "
                        f"({captain.get('consistency', 0):.2f} vs "
                        f"{top_xp.get('consistency', 0):.2f})"
                    )
                else:
                    analysis['reason'] = (
                        f"Highest xP ({captain.get('xp', 0):.1f}) with good consistency"
                    )

        return analysis
