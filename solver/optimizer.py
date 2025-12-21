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
        teams_data: Optional[List[Dict]] = None
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
            pos_df = pos_df[pos_df['minutes'] >= 90]  # At least ~1 game
            
            # Exclude current squad
            pos_df = pos_df[~pos_df['id'].isin(candidate_ids)]
            
            # Sort by total_points (proxy for quality)
            pos_df = pos_df.nlargest(self.candidate_pool_size, 'total_points')
            
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
