"""Transfer MIP Solver implementation."""

import os
import time
import logging
import pandas as pd
from typing import List, Dict, Optional, Any
from .definitions import MIPSolverResult, POSITION_MAP, POSITION_QUOTAS, MAX_PER_TEAM

logger = logging.getLogger(__name__)

# Optional MIP solver imports - graceful fallback if not installed
try:
    import sasoptpy as so
    import highspy
    MIP_AVAILABLE = True
except ImportError:
    MIP_AVAILABLE = False
    logger.warning("MIP solver dependencies (sasoptpy, highspy) not found. Solver will be unavailable.")

class TransferMIPSolver:
    """Mixed-Integer Programming solver for FPL transfer optimization.
    
    Uses sasoptpy for model building and HiGHS for solving to find the
    mathematically optimal transfer strategy over a multi-gameweek horizon.
    
    The solver maximizes expected points across the planning horizon while
    respecting FPL constraints (budget, position quotas, max 3 per team).
    """
    
    HIT_COST = 4  # Points deducted per extra transfer
    
    # Valid formation constraints (min/max in starting XI)
    XI_MIN = {'GKP': 1, 'DEF': 3, 'MID': 2, 'FWD': 1}
    XI_MAX = {'GKP': 1, 'DEF': 5, 'MID': 5, 'FWD': 3}
    
    def __init__(
        self,
        current_squad: List[Dict],
        bank: float,
        players_df: pd.DataFrame,
        xp_matrix: Dict[int, List[float]],
        free_transfers: int = 1,
        horizon: int = 5,
        candidate_pool_size: int = 30,
        discount_factor: float = 0.95,
        time_limit: float = 60.0,
        teams_data: Optional[List[Dict]] = None,
        candidate_ids: Optional[List[int]] = None,
    ):
        """Initialize the MIP solver.
        
        Args:
            current_squad: List of current squad player dicts (from get_current_squad).
                           Must include 'id', 'position', 'selling_price_m', 'team_id'.
            bank: Current money in bank (millions).
            players_df: DataFrame with all FPL players from bootstrap.
            xp_matrix: Dict mapping player_id -> list of xP for each GW in horizon.
            free_transfers: Number of free transfers available (1 or 2).
            horizon: Number of gameweeks to optimize over.
            candidate_pool_size: Max candidates per position to consider.
            discount_factor: Discount for future GWs (gamma in objective).
            time_limit: Solver time limit in seconds.
            teams_data: List of team dicts for name lookup.
            candidate_ids: Optional explicit candidate pool restriction (ids).
                           When provided, candidate selection will prioritize xP
                           rather than using observed minutes/points heuristics.
        """
        self.current_squad = current_squad
        self.bank = bank
        self.players_df = players_df.copy()
        self.xp_matrix = xp_matrix
        self.free_transfers = max(1, min(5, free_transfers))
        self.horizon = horizon
        self.candidate_pool_size = candidate_pool_size
        self.discount_factor = discount_factor
        self.time_limit = time_limit
        self.candidate_ids = set(candidate_ids) if candidate_ids else None
        
        # Build team name lookup
        self.team_names = {}
        if teams_data:
            for t in teams_data:
                self.team_names[t['id']] = t.get('short_name', 'UNK')
        
        # Add position labels to players_df
        # Use POSITION_MAP from definitions
        self.players_df['position'] = self.players_df['element_type'].map(POSITION_MAP)
        self.players_df['price'] = self.players_df['now_cost'] / 10.0
        
        # Current squad IDs
        self.current_squad_ids = set(p['id'] for p in current_squad)
        
        # Build candidate pool
        self.candidates = self._build_candidate_pool()
        
    def _build_candidate_pool(self) -> List[Dict]:
        """Build pool of candidate players for optimization.
        
        Includes all current squad players plus top N per position
        by total_points, filtered by availability.
        """
        candidates = []
        candidate_ids = set()
        
        # Add all current squad players first
        for p in self.current_squad:
            pid = p['id']
            candidate_ids.add(pid)
            
            # Get xP from matrix or default to 0
            xp_list = self.xp_matrix.get(pid, [0.0] * self.horizon)
            if len(xp_list) < self.horizon:
                xp_list = xp_list + [0.0] * (self.horizon - len(xp_list))
            
            candidates.append({
                'id': pid,
                'name': p.get('name', 'Unknown'),
                'position': p['position'],
                'team_id': p.get('team_id', 0),
                'team': p.get('team', self.team_names.get(p.get('team_id', 0), 'UNK')),
                'buy_price': p['stats'].get('now_cost', 0) / 10.0,
                'sell_price': p.get('selling_price_m') or (p['stats'].get('now_cost', 0) / 10.0),
                'is_current': True,
                'xp': xp_list,
                'total_xp': sum(xp_list),
                'status': p['stats'].get('status', 'a'),
                'chance': p['stats'].get('chance_of_playing_next_round', 100)
            })
        
        # Add top candidates per position (not in current squad)
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            pos_df = self.players_df[self.players_df['position'] == pos].copy()
            
            # Filter available players
            pos_df = pos_df[pos_df['status'] == 'a']
            if self.candidate_ids is None:
                # Heuristic to remove non-playing players (live mode).
                # Note: for pre-season / early season this can be too strict.
                pos_df = pos_df[pos_df['minutes'] >= 90]  # At least ~1 game
            else:
                # Explicit candidate set (simulation/backtest-safe mode).
                # Avoid ranking/filtering based on observed GW outcomes.
                pos_df = pos_df[pos_df['id'].isin(self.candidate_ids)]
            
            # Exclude current squad
            pos_df = pos_df[~pos_df['id'].isin(candidate_ids)]
            
            if self.candidate_ids is None:
                # Sort by season-to-date total_points (proxy for quality)
                pos_df = pos_df.nlargest(self.candidate_pool_size, 'total_points')
            else:
                # Prioritize by total xP over the horizon (consistent with objective).
                def xp_score(pid: int) -> float:
                    xp_list = self.xp_matrix.get(int(pid), [])
                    if not xp_list:
                        return 0.0
                    return float(sum(xp_list[: self.horizon]))

                pos_df = pos_df.assign(_xp_score=pos_df['id'].map(xp_score))
                # Stable fallback tie-breaker: prefer higher total_points if present.
                if 'total_points' in pos_df.columns:
                    pos_df = pos_df.sort_values(['_xp_score', 'total_points'], ascending=False)
                else:
                    pos_df = pos_df.sort_values(['_xp_score'], ascending=False)
                pos_df = pos_df.head(self.candidate_pool_size)
            
            for _, row in pos_df.iterrows():
                pid = int(row['id'])
                if pid in candidate_ids:
                    continue
                candidate_ids.add(pid)
                
                xp_list = self.xp_matrix.get(pid, [0.0] * self.horizon)
                if len(xp_list) < self.horizon:
                    xp_list = xp_list + [0.0] * (self.horizon - len(xp_list))
                
                candidates.append({
                    'id': pid,
                    'name': row.get('web_name', 'Unknown'),
                    'position': pos,
                    'team_id': int(row.get('team', 0)),
                    'team': self.team_names.get(int(row.get('team', 0)), 'UNK'),
                    'buy_price': row['price'],
                    'sell_price': row['price'],  # Same for non-owned players
                    'is_current': False,
                    'xp': xp_list,
                    'total_xp': sum(xp_list),
                    'status': row.get('status', 'a'),
                    'chance': row.get('chance_of_playing_next_round', 100)
                })
        
        return candidates
    
    def solve(self) -> MIPSolverResult:
        """Solve the transfer optimization problem.
        
        Returns:
            MIPSolverResult with optimal transfers and squad.
        """
        if not MIP_AVAILABLE:
            return MIPSolverResult(
                status='unavailable',
                message='MIP solver not available. Install sasoptpy and highspy.'
            )
        
        start_time = time.time()
        
        try:
            result = self._solve_mip()
            result.solver_time = time.time() - start_time
            return result
        except Exception as e:
            return MIPSolverResult(
                status='error',
                message=f'Solver error: {str(e)}',
                solver_time=time.time() - start_time
            )
    
    def _solve_mip(self) -> MIPSolverResult:
        """Build and solve the MIP model."""
        # Create model
        model = so.Model(name='FPL_Transfer_Optimizer')
        
        # Index sets
        players = list(range(len(self.candidates)))
        weeks = list(range(self.horizon))
        positions = ['GKP', 'DEF', 'MID', 'FWD']
        
        # Get candidate data
        cand = self.candidates
        
        # Decision Variables
        # x[p] = 1 if player p is in the new squad
        x = model.add_variables(players, vartype=so.BIN, name='squad')
        
        # lineup[p,w] = 1 if player p is in starting XI for week w
        lineup = model.add_variables(players, weeks, vartype=so.BIN, name='lineup')
        
        # captain[p,w] = 1 if player p is captain for week w
        captain = model.add_variables(players, weeks, vartype=so.BIN, name='captain')
        
        # Derived: transfer in/out
        # transfer_in[p] = 1 if player not in current squad but in new squad
        # transfer_out[p] = 1 if player in current squad but not in new squad
        transfer_in = {}
        transfer_out = {}
        for p in players:
            if cand[p]['is_current']:
                # Current player: transfer_out = 1 - x[p]
                transfer_out[p] = 1 - x[p]
                transfer_in[p] = 0
            else:
                # New player: transfer_in = x[p]
                transfer_in[p] = x[p]
                transfer_out[p] = 0
        
        # Number of transfers
        num_transfers = so.quick_sum(transfer_in[p] for p in players if not cand[p]['is_current'])
        
        # Hits: max(0, num_transfers - free_transfers)
        hits = model.add_variable(lb=0, vartype=so.CONT, name='hits')
        model.add_constraint(hits >= num_transfers - self.free_transfers, name='hits_lower')
        
        # CONSTRAINTS
        
        # 1. Squad size = 15
        model.add_constraint(
            so.quick_sum(x[p] for p in players) == 15,
            name='squad_size'
        )
        
        # 2. Position quotas
        for pos in positions:
            quota = POSITION_QUOTAS[pos]
            model.add_constraint(
                so.quick_sum(x[p] for p in players if cand[p]['position'] == pos) == quota,
                name=f'position_{pos}'
            )
        
        # 3. Max 3 players per team
        team_ids = set(c['team_id'] for c in cand)
        for tid in team_ids:
            model.add_constraint(
                so.quick_sum(x[p] for p in players if cand[p]['team_id'] == tid) <= MAX_PER_TEAM,
                name=f'team_{tid}'
            )
        
        # 4. Budget constraint
        # Money available = bank + sum of sell prices for players sold
        # Money spent = sum of buy prices for new players
        sell_revenue = so.quick_sum(
            cand[p]['sell_price'] * transfer_out[p] 
            for p in players if cand[p]['is_current']
        )
        buy_cost = so.quick_sum(
            cand[p]['buy_price'] * transfer_in[p]
            for p in players if not cand[p]['is_current']
        )
        model.add_constraint(
            buy_cost <= self.bank + sell_revenue,
            name='budget'
        )
        
        # 5. Lineup constraints per week
        for w in weeks:
            # Exactly 11 in lineup
            model.add_constraint(
                so.quick_sum(lineup[p, w] for p in players) == 11,
                name=f'lineup_size_w{w}'
            )
            
            # Can only be in lineup if in squad
            for p in players:
                model.add_constraint(lineup[p, w] <= x[p], name=f'lineup_squad_{p}_w{w}')
            
            # Formation constraints
            for pos in positions:
                pos_lineup = so.quick_sum(
                    lineup[p, w] for p in players if cand[p]['position'] == pos
                )
                model.add_constraint(
                    pos_lineup >= self.XI_MIN[pos],
                    name=f'xi_min_{pos}_w{w}'
                )
                model.add_constraint(
                    pos_lineup <= self.XI_MAX[pos],
                    name=f'xi_max_{pos}_w{w}'
                )
            
            # Exactly one captain
            model.add_constraint(
                so.quick_sum(captain[p, w] for p in players) == 1,
                name=f'one_captain_w{w}'
            )
            
            # Captain must be in lineup
            for p in players:
                model.add_constraint(captain[p, w] <= lineup[p, w], name=f'cap_lineup_{p}_w{w}')
        
        # OBJECTIVE
        # Maximize discounted expected points minus hit cost
        objective = 0
        
        for w in weeks:
            discount = self.discount_factor ** w
            
            # Points from lineup (including captain bonus)
            week_pts = so.quick_sum(
                lineup[p, w] * cand[p]['xp'][w] + captain[p, w] * cand[p]['xp'][w]
                for p in players
            )
            objective += discount * week_pts
        
        # Subtract hit cost
        objective -= self.HIT_COST * hits
        
        model.set_objective(objective, sense=so.MAX, name='maximize_xp')
        
        # Solve using HiGHS
        result = self._solve_with_highs(model)
        
        if result['status'] != 'optimal':
            return MIPSolverResult(
                status=result['status'],
                message=result.get('message', 'Solver did not find optimal solution')
            )
        
        # Extract solution
        solution = result['solution']
        
        # Build result
        new_squad = []
        transfers_out = []
        transfers_in = []
        
        for p in players:
            x_val = solution.get(f'squad[{p}]', 0)
            if x_val > 0.5:  # Selected in squad
                player_info = cand[p].copy()
                
                # Calculate average xP across lineup weeks
                lineup_weeks = []
                captain_weeks = []
                for w in weeks:
                    if solution.get(f'lineup[{p},{w}]', 0) > 0.5:
                        lineup_weeks.append(w)
                    if solution.get(f'captain[{p},{w}]', 0) > 0.5:
                        captain_weeks.append(w)
                
                player_info['lineup_weeks'] = lineup_weeks
                player_info['captain_weeks'] = captain_weeks
                player_info['is_starter'] = len(lineup_weeks) > 0
                new_squad.append(player_info)
                
                if not cand[p]['is_current']:
                    transfers_in.append(player_info)
        
        # Players transferred out
        for p in players:
            if cand[p]['is_current']:
                x_val = solution.get(f'squad[{p}]', 0)
                if x_val < 0.5:  # Not in new squad
                    transfers_out.append(cand[p])
        
        # Calculate hit cost
        num_xfers = len(transfers_in)
        hit_cost = max(0, num_xfers - self.free_transfers) * self.HIT_COST
        
        # Build starting XI and bench for first week
        starting_xi = []
        bench = []
        captain_player = None
        vc_player = None
        
        for player in new_squad:
            if 0 in player.get('lineup_weeks', []):
                starting_xi.append(player)
                if 0 in player.get('captain_weeks', []):
                    captain_player = player
            else:
                bench.append(player)
        
        # Sort XI by position
        pos_order = {'GKP': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
        starting_xi.sort(key=lambda x: (pos_order.get(x['position'], 4), -x['total_xp']))
        bench.sort(key=lambda x: (0 if x['position'] == 'GKP' else 1, -x['total_xp']))
        
        # Vice captain is second highest xP in XI
        xi_by_xp = sorted(starting_xi, key=lambda x: x['total_xp'], reverse=True)
        if len(xi_by_xp) > 1:
            vc_player = xi_by_xp[1] if xi_by_xp[0] == captain_player else xi_by_xp[0]
        
        # Determine formation
        formation = self._get_formation(starting_xi)
        
        # Calculate per-GW xP
        per_gw_xp = []
        for w in weeks:
            week_xp = sum(
                player['xp'][w] * (2 if w in player.get('captain_weeks', []) else 1)
                for player in starting_xi
            )
            per_gw_xp.append(round(week_xp, 1))
        
        # Total expected points
        total_xp = sum(per_gw_xp) - hit_cost
        
        # Budget remaining
        sell_total = sum(p['sell_price'] for p in transfers_out)
        buy_total = sum(p['buy_price'] for p in transfers_in)
        budget_remaining = round(self.bank + sell_total - buy_total, 1)
        
        return MIPSolverResult(
            status='optimal',
            transfers_out=transfers_out,
            transfers_in=transfers_in,
            new_squad=new_squad,
            starting_xi=starting_xi,
            bench=bench,
            formation=formation,
            captain=captain_player,
            vice_captain=vc_player,
            hit_cost=hit_cost,
            num_transfers=num_xfers,
            free_transfers_used=min(num_xfers, self.free_transfers),
            budget_remaining=budget_remaining,
            expected_points=round(total_xp, 1),
            per_gw_xp=per_gw_xp,
            message=f'Optimal solution found'
        )
    
    def _solve_with_highs(self, model: Any) -> Dict:
        """Solve sasoptpy model using HiGHS solver."""
        # Create temp file for MPS export - close it first so sasoptpy can write
        import tempfile as tmp
        fd, mps_file = tmp.mkstemp(suffix='.mps')
        os.close(fd)
        
        try:
            # Export model to MPS format
            model.export_mps(filename=mps_file)
            
            # Fix MPS format for HiGHS compatibility
            self._fix_mps_for_highs(mps_file)
            
            # Create HiGHS solver instance
            h = highspy.Highs()
            h.setOptionValue('time_limit', self.time_limit)
            h.setOptionValue('output_flag', False)  # Suppress output
            
            # Read the model
            h.readModel(mps_file)
            
            # Set optimization sense (HiGHS defaults to minimize)
            if getattr(self, '_is_maximizing', True):
                h.changeObjectiveSense(highspy.ObjSense.kMaximize)
            
            # Solve
            h.run()
            
            # Get solution status
            status = h.getModelStatus()
            
            if status == highspy.HighsModelStatus.kOptimal:
                # Extract solution with proper variable name mapping
                info = h.getInfo()
                sol = h.getSolution()
                col_values = sol.col_value
                
                try:
                    lp = h.getLp()
                    col_names = lp.col_names_
                except AttributeError:
                    # Fallback for older highspy versions - use sasoptpy variable order
                    col_names = [var.get_name() for var in model.get_variables()]
                
                # Build solution dictionary by zipping names with values
                if len(col_names) == len(col_values):
                    solution = dict(zip(col_names, col_values))
                else:
                    # Fallback
                    solution = {}
                    sasoptpy_vars = list(model.get_variables())
                    for i, val in enumerate(col_values):
                        if i < len(sasoptpy_vars):
                            solution[sasoptpy_vars[i].get_name()] = val
                        else:
                            solution[f'x_{i}'] = val
                
                return {
                    'status': 'optimal',
                    'solution': solution,
                    'objective': info.objective_function_value,
                    'time': 0.0
                }
            elif status == highspy.HighsModelStatus.kInfeasible:
                return {'status': 'infeasible', 'message': 'Model is infeasible'}
            elif status == highspy.HighsModelStatus.kUnbounded:
                return {'status': 'unbounded', 'message': 'Model is unbounded'}
            else:
                return {'status': 'timeout', 'message': f'Solver status: {status}'}
                
        except Exception as e:
            logger.error(f"HiGHS solver error: {e}")
            raise e
        finally:
            # Clean up temp file
            if os.path.exists(mps_file):
                try:
                    os.unlink(mps_file)
                except:
                    pass
    
    def _fix_mps_for_highs(self, mps_file: str):
        """Fix sasoptpy MPS format for HiGHS compatibility.
        
        sasoptpy writes 'MAX obj' but HiGHS expects 'N obj'.
        This method patches the MPS file to use standard format.
        """
        with open(mps_file, 'r') as f:
            content = f.read()
        
        # Track if we're maximizing (sasoptpy uses MAX/MIN in ROWS section)
        is_max = ' MAX ' in content or '\tMAX\t' in content
        
        # Replace MAX/MIN with N (free row indicator)
        import re
        content = re.sub(r'(\s)(MAX|MIN)(\s+)', r'\1N  \3', content)
        
        with open(mps_file, 'w') as f:
            f.write(content)
        
        # Store sense for later (HiGHS defaults to minimize)
        self._is_maximizing = is_max
    
    def _get_formation(self, starting_xi: List[Dict]) -> str:
        """Determine formation from starting XI."""
        counts = {'DEF': 0, 'MID': 0, 'FWD': 0}
        for p in starting_xi:
            pos = p['position']
            if pos in counts:
                counts[pos] += 1
        return f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"


class MultiPeriodMIPSolver:
    """Multi-period MIP solver with week-by-week transfer sequencing.

    Unlike TransferMIPSolver which optimizes a single final squad, this solver
    models transfers as decisions that can happen in any week of the horizon.
    This enables:
    - Strategic timing of transfers based on fixture swings
    - Free transfer banking (hold this week, double transfer next)
    - Week-by-week lineup optimization with evolving squad

    The solver produces three scenario plans: conservative, balanced, aggressive.
    """

    HIT_COST = 4
    MAX_FT_BANK = 5

    # Formation constraints
    XI_MIN = {'GKP': 1, 'DEF': 3, 'MID': 2, 'FWD': 1}
    XI_MAX = {'GKP': 1, 'DEF': 5, 'MID': 5, 'FWD': 3}

    def __init__(
        self,
        current_squad: List[Dict],
        bank: float,
        players_df: pd.DataFrame,
        xp_matrix: Dict[int, List[float]],
        free_transfers: int = 1,
        horizon: int = 5,
        candidate_pool_size: int = 40,
        discount_factor: float = 0.95,
        time_limit: float = 300.0,
        teams_data: Optional[List[Dict]] = None,
        price_predictions: Optional[Dict[int, List[float]]] = None,
        rival_squad_ids: Optional[set] = None,
        current_gw: int = 1,
    ):
        """Initialize the multi-period solver.

        Args:
            current_squad: List of current squad player dicts.
            bank: Current money in bank (millions).
            players_df: DataFrame with all FPL players.
            xp_matrix: Dict mapping player_id -> list of xP for each GW.
            free_transfers: Starting free transfers (1-5).
            horizon: Number of gameweeks to optimize over.
            candidate_pool_size: Max candidates per position.
            discount_factor: Discount for future GWs.
            time_limit: Solver time limit per scenario in seconds.
            teams_data: List of team dicts for name lookup.
            price_predictions: Dict {player_id: [price_gw0, price_gw1, ...]}.
            rival_squad_ids: Set of player IDs owned by mini-league rivals.
            current_gw: Current gameweek number.
        """
        self.current_squad = current_squad
        self.bank = bank
        self.players_df = players_df.copy()
        self.xp_matrix = xp_matrix
        self.free_transfers = max(1, min(self.MAX_FT_BANK, free_transfers))
        self.horizon = horizon
        self.candidate_pool_size = candidate_pool_size
        self.discount_factor = discount_factor
        self.time_limit = time_limit
        self.price_predictions = price_predictions or {}
        self.rival_squad_ids = rival_squad_ids or set()
        self.current_gw = current_gw

        # Build team name lookup
        self.team_names = {}
        if teams_data:
            for t in teams_data:
                self.team_names[t['id']] = t.get('short_name', 'UNK')

        # Add position labels
        self.players_df['position'] = self.players_df['element_type'].map(POSITION_MAP)
        self.players_df['price'] = self.players_df['now_cost'] / 10.0

        # Current squad IDs
        self.current_squad_ids = set(p['id'] for p in current_squad)

        # Build candidate pool
        self.candidates = self._build_candidate_pool()

        # Tracking for solution extraction
        self._solution = None

    def _build_candidate_pool(self) -> List[Dict]:
        """Build pool of candidate players for optimization."""
        candidates = []
        candidate_ids = set()

        # Add all current squad players first
        for p in self.current_squad:
            pid = p['id']
            candidate_ids.add(pid)

            xp_list = self.xp_matrix.get(pid, [0.0] * self.horizon)
            if len(xp_list) < self.horizon:
                xp_list = xp_list + [0.0] * (self.horizon - len(xp_list))

            # Get price trajectory if available
            prices = self.price_predictions.get(pid, [p.get('selling_price_m', 0)] * self.horizon)

            candidates.append({
                'id': pid,
                'name': p.get('name', 'Unknown'),
                'position': p['position'],
                'team_id': p.get('team_id', 0),
                'team': p.get('team', self.team_names.get(p.get('team_id', 0), 'UNK')),
                'buy_price': p['stats'].get('now_cost', 0) / 10.0,
                'sell_price': p.get('selling_price_m') or (p['stats'].get('now_cost', 0) / 10.0),
                'price_trajectory': prices,
                'is_current': True,
                'xp': xp_list,
                'total_xp': sum(xp_list),
                'status': p['stats'].get('status', 'a'),
                'chance': p['stats'].get('chance_of_playing_next_round', 100),
                'is_rival_owned': pid in self.rival_squad_ids,
            })

        # Add top candidates per position
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            pos_df = self.players_df[self.players_df['position'] == pos].copy()
            pos_df = pos_df[pos_df['status'] == 'a']
            pos_df = pos_df[pos_df['minutes'] >= 90]
            pos_df = pos_df[~pos_df['id'].isin(candidate_ids)]
            pos_df = pos_df.nlargest(self.candidate_pool_size, 'total_points')

            for _, row in pos_df.iterrows():
                pid = int(row['id'])
                if pid in candidate_ids:
                    continue
                candidate_ids.add(pid)

                xp_list = self.xp_matrix.get(pid, [0.0] * self.horizon)
                if len(xp_list) < self.horizon:
                    xp_list = xp_list + [0.0] * (self.horizon - len(xp_list))

                prices = self.price_predictions.get(pid, [row['price']] * self.horizon)

                candidates.append({
                    'id': pid,
                    'name': row.get('web_name', 'Unknown'),
                    'position': pos,
                    'team_id': int(row.get('team', 0)),
                    'team': self.team_names.get(int(row.get('team', 0)), 'UNK'),
                    'buy_price': row['price'],
                    'sell_price': row['price'],
                    'price_trajectory': prices,
                    'is_current': False,
                    'xp': xp_list,
                    'total_xp': sum(xp_list),
                    'status': row.get('status', 'a'),
                    'chance': row.get('chance_of_playing_next_round', 100),
                    'is_rival_owned': pid in self.rival_squad_ids,
                })

        return candidates

    def solve(self, scenarios: List[str] = None) -> 'MultiPeriodResult':
        """Solve for multiple scenarios.

        Args:
            scenarios: List of scenario names to solve.

        Returns:
            MultiPeriodResult with results for each scenario.
        """
        from .definitions import MultiPeriodResult, SCENARIO_CONFIGS

        if not MIP_AVAILABLE:
            return MultiPeriodResult(
                balanced=MIPSolverResult(
                    status='unavailable',
                    message='MIP solver not available. Install sasoptpy and highspy.'
                )
            )

        if scenarios is None:
            scenarios = ['conservative', 'balanced', 'aggressive']

        # Calculate baseline xP (no transfers)
        baseline_xp = self._calculate_baseline_xp()

        results = {}
        for scenario in scenarios:
            logger.info(f"Solving {scenario} scenario...")
            config = SCENARIO_CONFIGS.get(scenario, SCENARIO_CONFIGS['balanced'])
            result = self._solve_scenario(scenario, config)
            result.baseline_xp = baseline_xp
            results[scenario] = result

        # Determine recommendation based on net gain
        best_scenario = 'balanced'
        best_net_gain = results['balanced'].expected_points - baseline_xp if results.get('balanced') else 0

        for name, result in results.items():
            if result and result.status == 'optimal':
                net_gain = result.expected_points - baseline_xp
                # Conservative needs significantly better to be recommended
                # Aggressive needs to justify the risk
                if name == 'conservative' and net_gain > best_net_gain * 0.9:
                    best_scenario = name
                    best_net_gain = net_gain
                elif name == 'aggressive' and net_gain > best_net_gain * 1.1:
                    best_scenario = name
                    best_net_gain = net_gain

        # Generate watchlist if balanced scenario recommends few/no transfers
        watchlist = []
        if results.get('balanced') and results['balanced'].num_transfers == 0:
            watchlist = self._generate_watchlist()

        return MultiPeriodResult(
            conservative=results.get('conservative'),
            balanced=results.get('balanced'),
            aggressive=results.get('aggressive'),
            recommended=best_scenario,
            watchlist=watchlist,
            baseline_xp=baseline_xp,
        )

    def _calculate_baseline_xp(self) -> float:
        """Calculate expected points with current squad, no transfers."""
        total_xp = 0.0
        for w in range(self.horizon):
            discount = self.discount_factor ** w
            # Find best XI from current squad for this week
            week_xp = self._calculate_week_xp_for_squad(
                [c for c in self.candidates if c['is_current']], w
            )
            total_xp += discount * week_xp
        return round(total_xp, 1)

    def _calculate_week_xp_for_squad(self, squad: List[Dict], week: int) -> float:
        """Calculate best possible xP for a squad in a given week."""
        # Sort by xP for the week
        sorted_squad = sorted(squad, key=lambda p: p['xp'][week], reverse=True)

        # Greedy XI selection respecting position constraints
        xi = []
        pos_counts = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}

        for p in sorted_squad:
            pos = p['position']
            if len(xi) >= 11:
                break
            if pos_counts[pos] < self.XI_MAX[pos]:
                xi.append(p)
                pos_counts[pos] += 1

        # Validate minimum requirements
        for pos, min_count in self.XI_MIN.items():
            while pos_counts[pos] < min_count:
                for p in sorted_squad:
                    if p not in xi and p['position'] == pos:
                        xi.append(p)
                        pos_counts[pos] += 1
                        break
                else:
                    break

        # Calculate xP with captain bonus (best player gets 2x)
        if not xi:
            return 0.0

        xi_xp = [p['xp'][week] for p in xi[:11]]
        captain_xp = max(xi_xp) if xi_xp else 0
        return sum(xi_xp) + captain_xp  # Captain counted twice

    def _solve_scenario(self, scenario: str, config: Dict) -> MIPSolverResult:
        """Solve for a specific scenario configuration."""
        from .definitions import MIPSolverResult, WeeklyPlan, TransferAction

        start_time = time.time()

        try:
            result = self._solve_mip(config)
            result.solver_time = time.time() - start_time
            result.scenario = scenario
            return result
        except Exception as e:
            logger.error(f"Solver error for {scenario}: {e}")
            return MIPSolverResult(
                status='error',
                message=f'Solver error: {str(e)}',
                solver_time=time.time() - start_time,
                scenario=scenario,
            )

    def _solve_mip(self, config: Dict) -> MIPSolverResult:
        """Build and solve the multi-period MIP model."""
        from .definitions import MIPSolverResult, WeeklyPlan, TransferAction

        model = so.Model(name='FPL_MultiPeriod_Optimizer')

        players = list(range(len(self.candidates)))
        weeks = list(range(self.horizon))
        positions = ['GKP', 'DEF', 'MID', 'FWD']
        cand = self.candidates

        max_hits = config.get('max_hits', 8)
        max_transfers_per_week = config.get('max_transfers_per_week', 2)

        # Decision Variables
        # squad[p,w] = 1 if player p is in squad during week w
        squad = model.add_variables(players, weeks, vartype=so.BIN, name='squad')

        # transfer_in[p,w] = 1 if player p joins squad at start of week w
        transfer_in = model.add_variables(players, weeks, vartype=so.BIN, name='tin')

        # transfer_out[p,w] = 1 if player p leaves squad at start of week w
        transfer_out = model.add_variables(players, weeks, vartype=so.BIN, name='tout')

        # lineup[p,w] = 1 if player p starts in week w
        lineup = model.add_variables(players, weeks, vartype=so.BIN, name='lineup')

        # captain[p,w] = 1 if player p is captain in week w
        captain = model.add_variables(players, weeks, vartype=so.BIN, name='cap')

        # hits[w] = hit cost for week w
        hits = model.add_variables(weeks, lb=0, vartype=so.CONT, name='hits')

        # ft_used[w] = free transfers used in week w
        ft_used = model.add_variables(weeks, lb=0, ub=self.MAX_FT_BANK, vartype=so.INT, name='ft_used')

        # CONSTRAINTS

        # 1. Initial squad state (week 0 transfers from current to new)
        for p in players:
            if cand[p]['is_current']:
                # Current player: either stays or transferred out in week 0
                model.add_constraint(
                    squad[p, 0] == 1 - transfer_out[p, 0],
                    name=f'init_current_{p}'
                )
                model.add_constraint(transfer_in[p, 0] == 0, name=f'no_buy_current_{p}')
            else:
                # Non-current: either bought in week 0 or not in squad
                model.add_constraint(
                    squad[p, 0] == transfer_in[p, 0],
                    name=f'init_new_{p}'
                )
                model.add_constraint(transfer_out[p, 0] == 0, name=f'no_sell_new_{p}_w0')

        # 2. Squad continuity for weeks > 0
        for w in range(1, self.horizon):
            for p in players:
                model.add_constraint(
                    squad[p, w] == squad[p, w-1] + transfer_in[p, w] - transfer_out[p, w],
                    name=f'continuity_{p}_w{w}'
                )
                # Can only sell if in squad previous week
                model.add_constraint(
                    transfer_out[p, w] <= squad[p, w-1],
                    name=f'sell_if_owned_{p}_w{w}'
                )
                # Can only buy if not in squad previous week
                model.add_constraint(
                    transfer_in[p, w] <= 1 - squad[p, w-1],
                    name=f'buy_if_not_owned_{p}_w{w}'
                )

        # 3. Squad size = 15 each week
        for w in weeks:
            model.add_constraint(
                so.quick_sum(squad[p, w] for p in players) == 15,
                name=f'squad_size_w{w}'
            )

        # 4. Position quotas each week
        for w in weeks:
            for pos in positions:
                quota = POSITION_QUOTAS[pos]
                model.add_constraint(
                    so.quick_sum(squad[p, w] for p in players if cand[p]['position'] == pos) == quota,
                    name=f'position_{pos}_w{w}'
                )

        # 5. Max 3 per team each week
        team_ids = set(c['team_id'] for c in cand)
        for w in weeks:
            for tid in team_ids:
                model.add_constraint(
                    so.quick_sum(squad[p, w] for p in players if cand[p]['team_id'] == tid) <= MAX_PER_TEAM,
                    name=f'team_{tid}_w{w}'
                )

        # 6. Budget constraint each week (simplified - use initial budget)
        # More accurate would track sell/buy prices across weeks
        for w in weeks:
            sell_revenue = so.quick_sum(
                cand[p]['sell_price'] * transfer_out[p, w]
                for p in players if cand[p]['is_current'] or w > 0
            )
            buy_cost = so.quick_sum(
                cand[p]['buy_price'] * transfer_in[p, w]
                for p in players
            )
            model.add_constraint(
                buy_cost <= self.bank + sell_revenue,
                name=f'budget_w{w}'
            )

        # 7. Lineup constraints
        for w in weeks:
            model.add_constraint(
                so.quick_sum(lineup[p, w] for p in players) == 11,
                name=f'lineup_size_w{w}'
            )
            for p in players:
                model.add_constraint(lineup[p, w] <= squad[p, w], name=f'lineup_squad_{p}_w{w}')

            for pos in positions:
                pos_lineup = so.quick_sum(lineup[p, w] for p in players if cand[p]['position'] == pos)
                model.add_constraint(pos_lineup >= self.XI_MIN[pos], name=f'xi_min_{pos}_w{w}')
                model.add_constraint(pos_lineup <= self.XI_MAX[pos], name=f'xi_max_{pos}_w{w}')

            model.add_constraint(
                so.quick_sum(captain[p, w] for p in players) == 1,
                name=f'one_captain_w{w}'
            )
            for p in players:
                model.add_constraint(captain[p, w] <= lineup[p, w], name=f'cap_lineup_{p}_w{w}')

        # 8. Transfer limits per week
        for w in weeks:
            num_transfers_w = so.quick_sum(transfer_in[p, w] for p in players)
            model.add_constraint(
                num_transfers_w <= max_transfers_per_week,
                name=f'max_transfers_w{w}'
            )
            model.add_constraint(
                ft_used[w] <= num_transfers_w,
                name=f'ft_used_limit_w{w}'
            )

        # 9. Hit cost calculation
        # hits[w] >= 4 * (transfers[w] - ft_available[w])
        # FT accumulation: start with self.free_transfers, gain 1 per week (cap 5)
        for w in weeks:
            num_transfers_w = so.quick_sum(transfer_in[p, w] for p in players)
            # Simplified: assume we have at least 1 FT per week after week 0
            ft_avail = self.free_transfers if w == 0 else 1
            model.add_constraint(
                hits[w] >= self.HIT_COST * (num_transfers_w - ft_avail),
                name=f'hits_lower_w{w}'
            )

        # 10. Total hit limit for scenario
        model.add_constraint(
            so.quick_sum(hits[w] for w in weeks) <= max_hits,
            name='total_hits_limit'
        )

        # OBJECTIVE
        objective = 0
        for w in weeks:
            discount = self.discount_factor ** w
            week_pts = so.quick_sum(
                lineup[p, w] * cand[p]['xp'][w] + captain[p, w] * cand[p]['xp'][w]
                for p in players
            )
            objective += discount * week_pts - hits[w]

        model.set_objective(objective, sense=so.MAX, name='maximize_xp')

        # Solve
        solution = self._solve_with_highs(model)

        if solution['status'] != 'optimal':
            return MIPSolverResult(
                status=solution['status'],
                message=solution.get('message', 'Solver did not find optimal solution')
            )

        # Extract solution
        return self._extract_solution(solution['solution'], cand, weeks)

    def _solve_with_highs(self, model) -> Dict:
        """Solve sasoptpy model using HiGHS solver."""
        import tempfile as tmp

        fd, mps_file = tmp.mkstemp(suffix='.mps')
        os.close(fd)

        try:
            model.export_mps(filename=mps_file)
            self._fix_mps_for_highs(mps_file)

            h = highspy.Highs()
            h.setOptionValue('time_limit', self.time_limit)
            h.setOptionValue('output_flag', False)
            h.readModel(mps_file)

            if getattr(self, '_is_maximizing', True):
                h.changeObjectiveSense(highspy.ObjSense.kMaximize)

            h.run()
            status = h.getModelStatus()

            if status == highspy.HighsModelStatus.kOptimal:
                info = h.getInfo()
                sol = h.getSolution()
                col_values = sol.col_value

                try:
                    lp = h.getLp()
                    col_names = lp.col_names_
                except AttributeError:
                    col_names = [var.get_name() for var in model.get_variables()]

                if len(col_names) == len(col_values):
                    solution = dict(zip(col_names, col_values))
                else:
                    solution = {}
                    for i, val in enumerate(col_values):
                        solution[f'x_{i}'] = val

                return {
                    'status': 'optimal',
                    'solution': solution,
                    'objective': info.objective_function_value,
                    'gap': getattr(info, 'mip_gap', 0.0),
                }
            elif status == highspy.HighsModelStatus.kInfeasible:
                return {'status': 'infeasible', 'message': 'Model is infeasible'}
            else:
                return {'status': 'timeout', 'message': f'Solver status: {status}'}

        except Exception as e:
            logger.error(f"HiGHS solver error: {e}")
            raise e
        finally:
            if os.path.exists(mps_file):
                try:
                    os.unlink(mps_file)
                except:
                    pass

    def _fix_mps_for_highs(self, mps_file: str):
        """Fix sasoptpy MPS format for HiGHS compatibility."""
        with open(mps_file, 'r') as f:
            content = f.read()

        is_max = ' MAX ' in content or '\tMAX\t' in content

        import re
        content = re.sub(r'(\s)(MAX|MIN)(\s+)', r'\1N  \3', content)

        with open(mps_file, 'w') as f:
            f.write(content)

        self._is_maximizing = is_max

    def _extract_solution(self, solution: Dict, cand: List[Dict], weeks: List[int]) -> MIPSolverResult:
        """Extract structured result from solver solution."""
        from .definitions import MIPSolverResult, WeeklyPlan, TransferAction

        weekly_plans = []
        transfer_sequence = []
        all_transfers_in = []
        all_transfers_out = []
        total_hits = 0

        players = list(range(len(cand)))

        for w in weeks:
            # Extract transfers for this week
            week_in = []
            week_out = []

            for p in players:
                tin_val = solution.get(f'tin[{p},{w}]', 0)
                tout_val = solution.get(f'tout[{p},{w}]', 0)

                if tin_val > 0.5:
                    week_in.append(cand[p])
                if tout_val > 0.5:
                    week_out.append(cand[p])

            # Extract lineup
            week_lineup = []
            week_bench = []
            week_captain = None

            for p in players:
                squad_val = solution.get(f'squad[{p},{w}]', 0)
                lineup_val = solution.get(f'lineup[{p},{w}]', 0)
                cap_val = solution.get(f'cap[{p},{w}]', 0)

                if squad_val > 0.5:
                    if lineup_val > 0.5:
                        week_lineup.append(cand[p])
                        if cap_val > 0.5:
                            week_captain = cand[p]
                    else:
                        week_bench.append(cand[p])

            # Calculate FT and hits
            num_transfers = len(week_in)
            ft_avail = self.free_transfers if w == 0 else 1
            week_hits = max(0, num_transfers - ft_avail) * self.HIT_COST
            total_hits += week_hits

            # Calculate week xP
            week_xp = sum(p['xp'][w] for p in week_lineup)
            if week_captain:
                week_xp += week_captain['xp'][w]

            # Confidence based on week distance
            confidence = 'high' if w < 2 else 'moderate' if w < 4 else 'low'

            # Generate reasoning
            reasoning = self._generate_week_reasoning(week_in, week_out, w)

            weekly_plans.append(WeeklyPlan(
                gameweek=self.current_gw + w + 1,
                transfers_in=week_in,
                transfers_out=week_out,
                ft_available=ft_avail,
                ft_used=min(num_transfers, ft_avail),
                ft_remaining=max(0, ft_avail - num_transfers),
                hit_cost=week_hits,
                lineup=week_lineup,
                bench=week_bench,
                captain=week_captain,
                formation=self._get_formation(week_lineup),
                expected_xp=round(week_xp, 1),
                confidence=confidence,
                reasoning=reasoning,
                is_hold=(num_transfers == 0),
            ))

            # Build transfer sequence - match by position
            # FPL requires transfers to be within the same position
            matched_pairs = self._match_transfers_by_position(week_out, week_in)
            for i, (p_out, p_in) in enumerate(matched_pairs):
                cost = 'free' if i < ft_avail else f'-{self.HIT_COST} hit'
                gain = p_in['xp'][w] - p_out['xp'][w]

                transfer_sequence.append(TransferAction(
                    gameweek=self.current_gw + w + 1,
                    player_out=p_out,
                    player_in=p_in,
                    expected_gain=round(gain, 1),
                    cost=cost,
                    reasoning=f"{p_out['name']} -> {p_in['name']}: {reasoning}",
                ))

            all_transfers_in.extend(week_in)
            all_transfers_out.extend(week_out)

        # Build final result
        total_xp = sum(wp.expected_xp * (self.discount_factor ** i) for i, wp in enumerate(weekly_plans))
        total_xp -= total_hits

        # Get first week lineup for backward compatibility
        first_plan = weekly_plans[0] if weekly_plans else None

        return MIPSolverResult(
            status='optimal',
            transfers_out=all_transfers_out,
            transfers_in=all_transfers_in,
            new_squad=first_plan.lineup + first_plan.bench if first_plan else [],
            starting_xi=first_plan.lineup if first_plan else [],
            bench=first_plan.bench if first_plan else [],
            formation=first_plan.formation if first_plan else '',
            captain=first_plan.captain if first_plan else None,
            hit_cost=total_hits,
            num_transfers=len(all_transfers_in),
            budget_remaining=self.bank,
            expected_points=round(total_xp, 1),
            per_gw_xp=[wp.expected_xp for wp in weekly_plans],
            weekly_plans=weekly_plans,
            transfer_sequence=transfer_sequence,
            confidence_per_gw=[wp.confidence for wp in weekly_plans],
            message='Multi-period optimal solution found',
        )

    def _generate_week_reasoning(self, transfers_in: List[Dict], transfers_out: List[Dict], week: int) -> str:
        """Generate reasoning for a week's transfers."""
        if not transfers_in:
            if week == 0:
                return "Hold - current squad is well-positioned"
            return "Hold - bank free transfer for future flexibility"

        reasons = []
        # Match by position before generating reasoning
        matched_pairs = self._match_transfers_by_position(transfers_out, transfers_in)
        for p_out, p_in in matched_pairs:
            xp_diff = p_in['xp'][week] - p_out['xp'][week]
            reasons.append(f"{p_out['name']}({p_out['xp'][week]:.1f}xP) -> {p_in['name']}({p_in['xp'][week]:.1f}xP)")

        return "; ".join(reasons)

    def _get_formation(self, lineup: List[Dict]) -> str:
        """Determine formation from lineup."""
        counts = {'DEF': 0, 'MID': 0, 'FWD': 0}
        for p in lineup:
            pos = p.get('position', '')
            if pos in counts:
                counts[pos] += 1
        return f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"

    def _match_transfers_by_position(
        self, transfers_out: List[Dict], transfers_in: List[Dict]
    ) -> List[tuple]:
        """Match transfer-out and transfer-in players by position.

        FPL requires transfers to be within the same position - you can only
        replace a MID with another MID, a FWD with another FWD, etc.

        Args:
            transfers_out: List of players being sold.
            transfers_in: List of players being bought.

        Returns:
            List of (player_out, player_in) tuples matched by position.
        """
        from collections import defaultdict

        # Group by position
        out_by_pos = defaultdict(list)
        in_by_pos = defaultdict(list)

        for p in transfers_out:
            pos = p.get('position', 'UNK')
            out_by_pos[pos].append(p)

        for p in transfers_in:
            pos = p.get('position', 'UNK')
            in_by_pos[pos].append(p)

        # Match within each position
        matched = []
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            outs = out_by_pos.get(pos, [])
            ins = in_by_pos.get(pos, [])

            # Pair them up (zip will stop at shorter list)
            for p_out, p_in in zip(outs, ins):
                matched.append((p_out, p_in))

        return matched

    def _generate_watchlist(self) -> List[Dict]:
        """Generate watchlist of players to monitor."""
        watchlist = []

        # Find non-owned players with good xP
        for c in self.candidates:
            if not c['is_current'] and c['total_xp'] > 15:
                watchlist.append({
                    'name': c['name'],
                    'position': c['position'],
                    'team': c['team'],
                    'total_xp': c['total_xp'],
                    'reason': 'High expected points over horizon',
                })

        # Sort by total xP and return top 5
        watchlist.sort(key=lambda x: x['total_xp'], reverse=True)
        return watchlist[:5]
