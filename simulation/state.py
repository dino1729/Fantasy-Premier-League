"""State management for FPL season simulation.

This module defines the core data structures for tracking simulation state
across gameweeks, including squad composition, transfers, and results.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Optional, Any
from enum import Enum
import json
from pathlib import Path


class ChipType(str, Enum):
    """FPL chip types."""

    WILDCARD_1 = "wildcard1"
    WILDCARD_2 = "wildcard2"
    TRIPLE_CAPTAIN = "triple_captain"
    BENCH_BOOST = "bench_boost"
    FREE_HIT = "free_hit"


@dataclass
class PlayerState:
    """Snapshot of a player in the squad.

    Attributes:
        id: FPL element ID
        name: Player's display name
        position: Position code (GKP, DEF, MID, FWD)
        team_id: FPL team ID
        team_name: Team display name
        purchase_price: Price when acquired (millions)
        current_price: Current selling price (millions)
    """

    id: int
    name: str
    position: str
    team_id: int
    team_name: str
    purchase_price: float
    current_price: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlayerState":
        return cls(**data)


@dataclass
class TransferRecord:
    """Record of a single transfer.

    Attributes:
        player_out_id: ID of player sold
        player_out_name: Name of player sold
        player_in_id: ID of player bought
        player_in_name: Name of player bought
        price_out: Sale price (millions)
        price_in: Purchase price (millions)
        is_hit: Whether this transfer cost -4 points
    """

    player_out_id: int
    player_out_name: str
    player_in_id: int
    player_in_name: str
    price_out: float
    price_in: float
    is_hit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransferRecord":
        return cls(**data)


@dataclass
class AutoSubRecord:
    """Record of an automatic substitution.

    Attributes:
        player_out_id: ID of starter who didn't play
        player_out_name: Name of starter
        player_in_id: ID of bench player who came on
        player_in_name: Name of bench player
        reason: Why the sub occurred (e.g., 'non_player', 'formation_valid')
    """

    player_out_id: int
    player_out_name: str
    player_in_id: int
    player_in_name: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoSubRecord":
        return cls(**data)


@dataclass
class GameweekDecisions:
    """Decisions made for a specific gameweek.

    Attributes:
        transfers: List of transfers made
        lineup: List of 11 player IDs in starting XI
        bench_order: List of 4 player IDs in bench priority order
        captain_id: Captain player ID
        vice_captain_id: Vice captain player ID
        chip_used: Chip activated this GW (if any)
        formation: Formation string (e.g., '3-4-3')
    """

    transfers: List[TransferRecord] = field(default_factory=list)
    lineup: List[int] = field(default_factory=list)
    bench_order: List[int] = field(default_factory=list)
    captain_id: int = 0
    vice_captain_id: int = 0
    chip_used: Optional[str] = None
    formation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transfers": [t.to_dict() for t in self.transfers],
            "lineup": self.lineup,
            "bench_order": self.bench_order,
            "captain_id": self.captain_id,
            "vice_captain_id": self.vice_captain_id,
            "chip_used": self.chip_used,
            "formation": self.formation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameweekDecisions":
        return cls(
            transfers=[TransferRecord.from_dict(t) for t in data.get("transfers", [])],
            lineup=data.get("lineup", []),
            bench_order=data.get("bench_order", []),
            captain_id=data.get("captain_id", 0),
            vice_captain_id=data.get("vice_captain_id", 0),
            chip_used=data.get("chip_used"),
            formation=data.get("formation", ""),
        )


@dataclass
class GameweekResults:
    """Results after a gameweek is scored.

    Attributes:
        gw_points: Points scored this GW (after hits)
        gw_points_before_hits: Points before deducting hits
        hit_cost: Total hit cost this GW
        auto_subs: List of auto-substitutions applied
        effective_captain_id: Actual captain (may differ if original didn't play)
        captain_points: Points scored by captain (before 2x)
        bench_points: Points scored by bench players
    """

    gw_points: int = 0
    gw_points_before_hits: int = 0
    hit_cost: int = 0
    auto_subs: List[AutoSubRecord] = field(default_factory=list)
    effective_captain_id: int = 0
    captain_points: int = 0
    bench_points: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gw_points": self.gw_points,
            "gw_points_before_hits": self.gw_points_before_hits,
            "hit_cost": self.hit_cost,
            "auto_subs": [s.to_dict() for s in self.auto_subs],
            "effective_captain_id": self.effective_captain_id,
            "captain_points": self.captain_points,
            "bench_points": self.bench_points,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameweekResults":
        return cls(
            gw_points=data.get("gw_points", 0),
            gw_points_before_hits=data.get("gw_points_before_hits", 0),
            hit_cost=data.get("hit_cost", 0),
            auto_subs=[AutoSubRecord.from_dict(s) for s in data.get("auto_subs", [])],
            effective_captain_id=data.get("effective_captain_id", 0),
            captain_points=data.get("captain_points", 0),
            bench_points=data.get("bench_points", 0),
        )


@dataclass
class GameweekState:
    """Complete state snapshot after a gameweek.

    This is the primary checkpoint object saved after each GW.
    Contains everything needed to resume simulation from this point.

    Attributes:
        gameweek: The gameweek number (1-38)
        squad: List of 15 PlayerState objects
        bank: Money in bank (millions)
        free_transfers: Free transfers available for next GW
        total_points: Cumulative points through this GW
        chips_available: Set of chip names still available
        decisions: Decisions made this GW
        results: Results after scoring this GW
    """

    gameweek: int
    squad: List[PlayerState]
    bank: float
    free_transfers: int
    total_points: int
    chips_available: Set[str]
    decisions: GameweekDecisions
    results: GameweekResults

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "gameweek": self.gameweek,
            "squad": [p.to_dict() for p in self.squad],
            "bank": self.bank,
            "free_transfers": self.free_transfers,
            "total_points": self.total_points,
            "chips_available": list(self.chips_available),
            "decisions": self.decisions.to_dict(),
            "results": self.results.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameweekState":
        """Create from dictionary (JSON deserialization)."""
        return cls(
            gameweek=data["gameweek"],
            squad=[PlayerState.from_dict(p) for p in data["squad"]],
            bank=data["bank"],
            free_transfers=data["free_transfers"],
            total_points=data["total_points"],
            chips_available=set(data["chips_available"]),
            decisions=GameweekDecisions.from_dict(data["decisions"]),
            results=GameweekResults.from_dict(data["results"]),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "GameweekState":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save_checkpoint(self, output_dir: Path) -> Path:
        """Save state to checkpoint file.

        Args:
            output_dir: Directory to save checkpoint

        Returns:
            Path to the saved checkpoint file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = output_dir / f"gw{self.gameweek}.json"
        with open(checkpoint_path, "w") as f:
            f.write(self.to_json())

        return checkpoint_path

    @classmethod
    def load_checkpoint(cls, checkpoint_path: Path) -> "GameweekState":
        """Load state from checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            GameweekState loaded from file
        """
        with open(checkpoint_path, "r") as f:
            return cls.from_json(f.read())

    def get_player_by_id(self, player_id: int) -> Optional[PlayerState]:
        """Get a player from squad by ID."""
        for player in self.squad:
            if player.id == player_id:
                return player
        return None

    def get_squad_value(self) -> float:
        """Get total squad value (selling prices)."""
        return sum(p.current_price for p in self.squad)

    def get_total_value(self) -> float:
        """Get total value (squad + bank)."""
        return self.get_squad_value() + self.bank


@dataclass
class SimulationResult:
    """Final results of a complete simulation run.

    Attributes:
        season: Season identifier (e.g., '2024-25')
        states: List of all GameweekState snapshots
        total_points: Final total points
        total_hits: Total hit points taken
        chips_used: Dict mapping chip name to GW used
        transfers_made: Total number of transfers
    """

    season: str
    states: List[GameweekState]
    total_points: int
    total_hits: int
    chips_used: Dict[str, int]
    transfers_made: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "season": self.season,
            "total_points": self.total_points,
            "total_hits": self.total_hits,
            "chips_used": self.chips_used,
            "transfers_made": self.transfers_made,
            "states": [s.to_dict() for s in self.states],
        }

    def get_best_gw(self) -> tuple:
        """Get best gameweek (gw_number, points)."""
        if not self.states:
            return (0, 0)
        best = max(self.states, key=lambda s: s.results.gw_points)
        return (best.gameweek, best.results.gw_points)

    def get_worst_gw(self) -> tuple:
        """Get worst gameweek (gw_number, points)."""
        if not self.states:
            return (0, 0)
        worst = min(self.states, key=lambda s: s.results.gw_points)
        return (worst.gameweek, worst.results.gw_points)

    def get_average_gw_points(self) -> float:
        """Get average points per gameweek."""
        if not self.states:
            return 0.0
        return sum(s.results.gw_points for s in self.states) / len(self.states)


# Initial chips available at season start
INITIAL_CHIPS = {
    ChipType.WILDCARD_1.value,
    ChipType.TRIPLE_CAPTAIN.value,
    ChipType.BENCH_BOOST.value,
    ChipType.FREE_HIT.value,
}

# 2025-26 season: Two sets of chips (GW1-19 and GW20-38)
CHIP_RESET_GW = 20  # GW20 starts second set of chips
CHIPS_PER_HALF = 4  # 1 WC + 1 TC + 1 BB + 1 FH per half

# FPL constants
STARTING_BUDGET = 100.0
STARTING_FREE_TRANSFERS = 1
MAX_FREE_TRANSFERS = 5
HIT_COST = 4
SQUAD_SIZE = 15
XI_SIZE = 11
