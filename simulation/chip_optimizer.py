"""Dynamic chip optimization for FPL simulation.

This module determines optimal chip timing based on fixture analysis,
squad state, and strategic criteria.
"""

from typing import Dict, List, Optional, Set, Tuple
from simulation.state import GameweekState, PlayerState, ChipType
from simulation.data_adapter import HistoricalDataAdapter


class ChipOptimizer:
    """Determines optimal chip timing based on configurable triggers.

    Chip trigger criteria (user-configured):
    - Wildcard: Fixture swing detection (4+ players with worsening fixtures)
    - Triple Captain: DGW + premium asset (>=10m)
    - Bench Boost: DGW with 4+ bench players doubling
    - Free Hit: Blank gameweek with 6+ teams missing
    """

    # Configuration constants
    WILDCARD_WORSENING_THRESHOLD = 4  # Min players with worsening fixtures
    WILDCARD_MIN_GWS_REMAINING = 5  # Don't use WC with fewer GWs left
    WILDCARD_FDR_INCREASE_THRESHOLD = 1.0  # Avg FDR increase to count as "worsening"
    WILDCARD_LOOKBACK = 3  # GWs to look back for fixture comparison
    WILDCARD_LOOKAHEAD = 5  # GWs to look ahead for fixture comparison

    TC_MIN_PRICE = 8.0  # Premium threshold for TC (lowered)
    TC_MAX_FDR = 4.0  # Both fixtures must have FDR <= this (relaxed)

    BB_MIN_BENCH_DOUBLING = 2  # Min bench players doubling for BB (lowered from 4)

    FH_MIN_TEAMS_MISSING = 3  # Min teams blanking for FH (lowered from 6)
    FH_MIN_SQUAD_AFFECTED = 2  # Min squad players affected by blank (lowered from 4)

    # Force chip usage thresholds (use remaining chips near season end)
    # 2025-26 season: Two sets of chips, so force usage in each half
    FORCE_BB_BY_GW = 17  # Force BB by GW17 in first half (fresh set at GW20)
    FORCE_FH_BY_GW = 16  # Force FH by GW16 in first half (fresh set at GW20)
    FORCE_TC_BY_GW = 18  # Force TC by GW18 in first half (fresh set at GW20)

    def __init__(self, data_adapter: HistoricalDataAdapter):
        """Initialize chip optimizer.

        Args:
            data_adapter: Historical data adapter for fixture/DGW info
        """
        self.data = data_adapter
        self._dgw_bgw = data_adapter.get_dgw_bgw_info()

    def decide_chip(
        self,
        state: GameweekState,
        gw: int,
        captain_candidate: Optional[Dict] = None,
    ) -> Optional[str]:
        """Decide which chip (if any) to use this GW.

        Priority order:
        1. Free Hit (BGW coverage - highest priority)
        2. Wildcard (fixture swing)
        3. Bench Boost (DGW + strong bench)
        4. Triple Captain (DGW + premium)
        5. Forced chip usage near season end

        Args:
            state: Current simulation state
            gw: Gameweek number
            captain_candidate: Best captain candidate (for TC decision)

        Returns:
            Chip name to use, or None
        """
        # Free Hit has highest priority (BGW coverage)
        if self.should_use_free_hit(state, gw):
            return ChipType.FREE_HIT.value

        # Wildcard for fixture swings
        if self.should_use_wildcard(state, gw):
            # Use the available wildcard
            if ChipType.WILDCARD_1.value in state.chips_available:
                return ChipType.WILDCARD_1.value

        # Bench Boost on DGW with strong bench
        if self.should_use_bench_boost(state, gw):
            return ChipType.BENCH_BOOST.value

        # Triple Captain on DGW with premium
        if captain_candidate and self.should_use_triple_captain(
            state, captain_candidate, gw
        ):
            return ChipType.TRIPLE_CAPTAIN.value

        # Force chip usage near season end to avoid wasting chips
        forced = self._check_forced_chip_usage(state, gw)
        if forced:
            return forced

        return None

    def _check_forced_chip_usage(self, state: GameweekState, gw: int) -> Optional[str]:
        """Force use of remaining chips near season end.

        2025-26 season: Force usage in both halves since chips reset at GW20.

        Args:
            state: Current simulation state
            gw: Gameweek number

        Returns:
            Chip to force use, or None
        """
        from simulation.state import CHIP_RESET_GW

        # Force chip usage in first half (GW1-19)
        if gw < CHIP_RESET_GW:
            # Force Free Hit
            if gw >= self.FORCE_FH_BY_GW:
                if ChipType.FREE_HIT.value in state.chips_available:
                    return ChipType.FREE_HIT.value

            # Force Bench Boost
            if gw >= self.FORCE_BB_BY_GW:
                if ChipType.BENCH_BOOST.value in state.chips_available:
                    return ChipType.BENCH_BOOST.value

            # Force Triple Captain
            if gw >= self.FORCE_TC_BY_GW:
                if ChipType.TRIPLE_CAPTAIN.value in state.chips_available:
                    return ChipType.TRIPLE_CAPTAIN.value

        # Force chip usage in second half (GW20-38) - use same thresholds relative to GW20
        else:
            # Force Free Hit by GW36 (16 GWs after reset)
            if gw >= CHIP_RESET_GW + 16:
                if ChipType.FREE_HIT.value in state.chips_available:
                    return ChipType.FREE_HIT.value

            # Force Bench Boost by GW37 (17 GWs after reset)
            if gw >= CHIP_RESET_GW + 17:
                if ChipType.BENCH_BOOST.value in state.chips_available:
                    return ChipType.BENCH_BOOST.value

            # Force Triple Captain by GW38 (18 GWs after reset)
            if gw >= CHIP_RESET_GW + 18:
                if ChipType.TRIPLE_CAPTAIN.value in state.chips_available:
                    return ChipType.TRIPLE_CAPTAIN.value

        return None

    def should_use_wildcard(self, state: GameweekState, gw: int) -> bool:
        """Determine if wildcard should be used.

        Trigger: Fixture swing detection
        - 4+ squad players have significantly worsening fixtures
        - At least 5 GWs remaining (don't waste WC at season end)

        Args:
            state: Current simulation state
            gw: Gameweek number

        Returns:
            True if wildcard should be used
        """
        # Check if wildcard is available
        wc_available = ChipType.WILDCARD_1.value in state.chips_available
        if not wc_available:
            return False

        # Don't use WC too late in season
        gws_remaining = 38 - gw + 1
        if gws_remaining < self.WILDCARD_MIN_GWS_REMAINING:
            return False

        # Analyze fixture swings
        worsening_count = 0

        for player in state.squad:
            team_id = self._get_player_team_id(player, gw)
            if team_id is None:
                continue

            # Compare past vs future fixture difficulty
            past_fdr = self._get_avg_fdr(team_id, gw - self.WILDCARD_LOOKBACK, gw - 1)
            future_fdr = self._get_avg_fdr(
                team_id, gw, gw + self.WILDCARD_LOOKAHEAD - 1
            )

            # Check if fixtures are significantly worsening
            fdr_increase = future_fdr - past_fdr
            if fdr_increase >= self.WILDCARD_FDR_INCREASE_THRESHOLD:
                worsening_count += 1

        return worsening_count >= self.WILDCARD_WORSENING_THRESHOLD

    def should_use_triple_captain(
        self,
        state: GameweekState,
        captain_candidate: Dict,
        gw: int,
    ) -> bool:
        """Determine if triple captain should be used.

        Trigger: DGW + premium asset
        - Captain is playing in DGW (2 fixtures)
        - Captain price >= 10.0m (premium)
        - Both fixtures have favorable difficulty

        Args:
            state: Current simulation state
            captain_candidate: Best captain candidate dict
            gw: Gameweek number

        Returns:
            True if triple captain should be used
        """
        if ChipType.TRIPLE_CAPTAIN.value not in state.chips_available:
            return False

        # Check if this is a DGW
        dgw_info = self._get_dgw_for_gw(gw)
        if dgw_info is None:
            return False

        # Get captain's team
        captain_team_id = captain_candidate.get("team_id")
        if captain_team_id is None:
            return False

        # Check if captain's team is doubling
        if captain_team_id not in dgw_info["team_ids"]:
            return False

        # Check captain is premium
        captain_price = captain_candidate.get("price", 0)
        if captain_price < self.TC_MIN_PRICE:
            return False

        # Check both fixtures are favorable
        fixtures = self.data.get_fixtures_for_gw(gw)
        captain_fixtures = [
            f
            for f in fixtures
            if f["team_h"] == captain_team_id or f["team_a"] == captain_team_id
        ]

        for fixture in captain_fixtures:
            if fixture["team_h"] == captain_team_id:
                fdr = fixture.get("team_h_difficulty", 3)
            else:
                fdr = fixture.get("team_a_difficulty", 3)

            if fdr > self.TC_MAX_FDR:
                return False

        return True

    def should_use_bench_boost(self, state: GameweekState, gw: int) -> bool:
        """Determine if bench boost should be used.

        Trigger: DGW with strong bench
        - At least 4 bench players are from teams with double fixtures

        Args:
            state: Current simulation state
            gw: Gameweek number

        Returns:
            True if bench boost should be used
        """
        if ChipType.BENCH_BOOST.value not in state.chips_available:
            return False

        # Check if this is a DGW
        dgw_info = self._get_dgw_for_gw(gw)
        if dgw_info is None:
            return False

        # Get doubling teams
        doubling_teams = set(dgw_info["team_ids"])

        # Count bench players from doubling teams
        # Assume bench is last 4 players in squad (positions 12-15)
        bench = state.squad[11:15] if len(state.squad) >= 15 else []

        bench_doubling = 0
        for player in bench:
            team_id = self._get_player_team_id(player, gw)
            if team_id and team_id in doubling_teams:
                bench_doubling += 1

        return bench_doubling >= self.BB_MIN_BENCH_DOUBLING

    def should_use_free_hit(self, state: GameweekState, gw: int) -> bool:
        """Determine if free hit should be used.

        Trigger: Blank gameweek coverage
        - BGW with 6+ teams missing fixtures
        - Squad has 4+ players from blanking teams

        Args:
            state: Current simulation state
            gw: Gameweek number

        Returns:
            True if free hit should be used
        """
        if ChipType.FREE_HIT.value not in state.chips_available:
            return False

        # Check if this is a significant BGW
        bgw_info = self._get_bgw_for_gw(gw)
        if bgw_info is None:
            return False

        if bgw_info["teams_missing"] < self.FH_MIN_TEAMS_MISSING:
            return False

        # Get blanking teams
        blanking_teams = set(bgw_info["team_ids"])

        # Count squad players affected
        affected = 0
        for player in state.squad:
            team_id = self._get_player_team_id(player, gw)
            if team_id and team_id in blanking_teams:
                affected += 1

        return affected >= self.FH_MIN_SQUAD_AFFECTED

    def _get_player_team_id(self, player: PlayerState, gw: int) -> Optional[int]:
        """Get team ID for a player.

        Args:
            player: PlayerState object
            gw: Gameweek number (for lookup)

        Returns:
            Team ID or None
        """
        # Try from data adapter first
        team_id = self.data.get_player_team_id(player.id, gw)
        if team_id:
            return team_id

        # Fall back to player state
        return player.team_id

    def _get_avg_fdr(self, team_id: int, start_gw: int, end_gw: int) -> float:
        """Get average fixture difficulty rating for a team.

        Args:
            team_id: FPL team ID
            start_gw: Start gameweek
            end_gw: End gameweek

        Returns:
            Average FDR (default 3.0 if no data)
        """
        total_fdr = 0.0
        fixture_count = 0

        for gw in range(max(1, start_gw), min(39, end_gw + 1)):
            fdr = self.data.get_fixture_difficulty(team_id, gw)
            # Only count if team has fixtures
            if fdr != 3.0 or not self.data.is_bgw(team_id, gw):
                total_fdr += fdr
                fixture_count += 1

        if fixture_count == 0:
            return 3.0  # Default medium

        return total_fdr / fixture_count

    def _get_dgw_for_gw(self, gw: int) -> Optional[Dict]:
        """Get DGW info for a specific gameweek.

        Args:
            gw: Gameweek number

        Returns:
            DGW info dict or None
        """
        for dgw in self._dgw_bgw["dgw"]:
            if dgw["gw"] == gw:
                return dgw
        return None

    def _get_bgw_for_gw(self, gw: int) -> Optional[Dict]:
        """Get BGW info for a specific gameweek.

        Args:
            gw: Gameweek number

        Returns:
            BGW info dict or None
        """
        for bgw in self._dgw_bgw["bgw"]:
            if bgw["gw"] == gw:
                return bgw
        return None

    def get_dgw_gws(self) -> List[int]:
        """Get list of all DGW gameweek numbers."""
        return [d["gw"] for d in self._dgw_bgw["dgw"]]

    def get_bgw_gws(self) -> List[int]:
        """Get list of all BGW gameweek numbers."""
        return [b["gw"] for b in self._dgw_bgw["bgw"]]

    def analyze_chip_opportunities(
        self, state: GameweekState, start_gw: int, end_gw: int = 38
    ) -> List[Dict]:
        """Analyze chip opportunities across remaining season.

        Useful for planning chip usage strategy.

        Args:
            state: Current simulation state
            start_gw: Starting gameweek
            end_gw: Ending gameweek

        Returns:
            List of opportunity dicts with GW and chip recommendations
        """
        opportunities = []

        for gw in range(start_gw, end_gw + 1):
            opp = {
                "gw": gw,
                "is_dgw": self._get_dgw_for_gw(gw) is not None,
                "is_bgw": self._get_bgw_for_gw(gw) is not None,
                "dgw_teams": [],
                "bgw_teams": [],
                "recommended_chip": None,
            }

            dgw_info = self._get_dgw_for_gw(gw)
            if dgw_info:
                opp["dgw_teams"] = dgw_info.get("team_names", [])

            bgw_info = self._get_bgw_for_gw(gw)
            if bgw_info:
                opp["bgw_teams"] = bgw_info.get("team_names", [])

            opportunities.append(opp)

        return opportunities
