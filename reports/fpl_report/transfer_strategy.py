"""Transfer Strategy Planner Module

Builds comprehensive multi-week transfer plans with expected value analysis,
fixture swing detection, and coordinated transfer recommendations.

Includes MIP-based transfer optimization using sasoptpy + HiGHS for
mathematically optimal transfer decisions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import tempfile
import os

from solver.optimizer import TransferMIPSolver, MIP_AVAILABLE
from solver.definitions import MIPSolverResult


@dataclass
class TransferPlan:
    """Represents a single transfer plan."""
    gameweek: int
    player_out: Dict
    player_in: Dict
    expected_gain: float
    reasoning: str
    priority: str  # 'high', 'medium', 'low'
    cumulative_gain: float = 0.0


class TransferStrategyPlanner:
    """Plans comprehensive multi-week transfer strategies.
    
    Analyzes squad over a planning horizon to recommend optimal
    transfer timing, sequences, and expected value.
    """

    def __init__(self, data_fetcher, player_analyzer, transfer_recommender):
        """Initialize the strategy planner.
        
        Args:
            data_fetcher: FPLDataFetcher instance.
            player_analyzer: PlayerAnalyzer instance.
            transfer_recommender: TransferRecommender instance.
        """
        self.fetcher = data_fetcher
        self.analyzer = player_analyzer
        self.recommender = transfer_recommender
        self.predictor = transfer_recommender.predictor
        
        # Store FPL Core data references for the new predictor
        self.all_gw_data = getattr(transfer_recommender, 'all_gw_data', None)
        self.fpl_core_season_data = getattr(transfer_recommender, 'fpl_core_season_data', None)
        self.current_gw = getattr(transfer_recommender, 'current_gw', None)
        self._use_fpl_core_predictor = getattr(transfer_recommender, '_fpl_core_predictor_ready', False)
        
        # Cross-season training data (from recommender)
        self.prev_all_gw_data = getattr(transfer_recommender, 'prev_all_gw_data', None)
        self.prev_fpl_core_season_data = getattr(transfer_recommender, 'prev_fpl_core_season_data', None)
        self.prev_season_start_year = getattr(transfer_recommender, 'prev_season_start_year', None)
        self.current_season_start_year = getattr(transfer_recommender, 'current_season_start_year', None)
        self._cross_season_available = getattr(transfer_recommender, '_cross_season_available', False)
        
        # Access new inference pipeline if available
        self.inference_pipeline = getattr(transfer_recommender, 'inference_pipeline', None)
        if self.inference_pipeline:
            print("TransferStrategyPlanner: Using position-specific ML models")
    
    def _ensure_predictor_trained(self):
        """Ensure the prediction model is trained.
        
        First tries to load pre-trained models from disk (faster).
        Falls back to training from scratch if no saved models found.
        """
        from .fpl_core_predictor import FPLCorePredictor
        
        if self.predictor.is_trained:
            return
            
        if self._use_fpl_core_predictor and isinstance(self.predictor, FPLCorePredictor):
            # Try loading pre-trained models from disk first
            if self.predictor.load_from_disk():
                print("TransferStrategyPlanner: Using pre-trained FPL Core Predictor models")
                return
            
            # Fall back to training if no saved models
            print("TransferStrategyPlanner: No pre-trained models found, training from scratch...")
            
            # Use cross-season training if previous season data is available
            if self._cross_season_available:
                self.predictor.train_cross_season(
                    prev_all_gw_data=self.prev_all_gw_data,
                    prev_fpl_core_season_data=self.prev_fpl_core_season_data,
                    prev_season_start_year=self.prev_season_start_year,
                    current_all_gw_data=self.all_gw_data,
                    current_fpl_core_season_data=self.fpl_core_season_data,
                    current_season_start_year=self.current_season_start_year,
                    current_gw=self.current_gw,
                )
            else:
                self.predictor.train(self.all_gw_data, self.fpl_core_season_data, self.current_gw)
        else:
            all_players = self.fetcher.players_df
            train_ids = all_players[all_players['minutes'] >= 450]['id'].tolist()[:200]
            self.predictor.train(train_ids)
    
    def _predict_multiple_gws(self, player_ids: List[int], num_gws: int = 5) -> Dict:
        """Wrapper for predict_multiple_gws that handles both predictor types."""
        from .fpl_core_predictor import FPLCorePredictor
        
        if self._use_fpl_core_predictor and isinstance(self.predictor, FPLCorePredictor):
            return self.predictor.predict_multiple_gws(
                self.all_gw_data, self.fpl_core_season_data,
                player_ids, self.current_gw, num_gws
            )
        else:
            return self.predictor.predict_multiple_gws(player_ids, num_gws)
        
    def generate_strategy(self, squad_analysis: List[Dict], 
                          num_weeks: int = 5,
                          use_mip: bool = True,
                          mip_time_limit: float = 60.0,
                          mip_candidate_pool: int = 30,
                          free_transfers: int = 1,
                          current_gw: int = None) -> Dict:
        """Generate comprehensive multi-week transfer strategy.
        
        Args:
            squad_analysis: List of player analysis dicts for current squad.
            num_weeks: Planning horizon in gameweeks (default 5).
            use_mip: Whether to run the MIP solver for optimal transfers.
            mip_time_limit: Time limit for MIP solver in seconds.
            mip_candidate_pool: Number of candidates per position for MIP.
            free_transfers: Number of free transfers available (1 or 2).
            
        Returns:
            Strategy dict containing:
            - squad_predictions: Multi-GW predictions for current squad
            - fixture_analysis: Fixture difficulty analysis per player
            - immediate_recommendations: Transfers to make now (heuristic)
            - planned_transfers: Sequence of recommended transfers
            - alternative_strategies: Aggressive/Conservative options
            - expected_value: EV analysis
            - model_metrics: Model performance metrics
            - mip_recommendation: MIP solver result (if use_mip=True)
        """
        if current_gw is None:
            current_gw = self.fetcher.get_current_gameweek()
        
        # Get player IDs from squad
        squad_ids = [p['player_id'] for p in squad_analysis if p.get('player_id')]
        
        # Ensure model is trained
        if not self.predictor.is_trained:
            self._ensure_predictor_trained()
        
        # Get multi-GW predictions for squad
        squad_predictions = self._predict_multiple_gws(squad_ids, num_weeks)
        
        # Analyze fixtures for each player
        fixture_analysis = self._analyze_squad_fixtures(squad_analysis, num_weeks)
        
        # Calculate expected value for current squad
        current_squad_ev = self._calculate_squad_ev(squad_predictions)
        
        # Get underperformers
        underperformers = self.recommender.identify_underperformers(squad_analysis)
        
        # Get immediate recommendations (enhanced with multi-GW data)
        immediate_recommendations = self._get_immediate_recommendations(
            underperformers, squad_predictions, fixture_analysis, num_weeks
        )
        
        # Generate transfer sequence plan
        planned_transfers = self._plan_transfer_sequence(
            squad_analysis, immediate_recommendations, num_weeks,
            current_gw, free_transfers
        )
        
        # Generate alternative strategies
        alternative_strategies = self._generate_alternatives(
            immediate_recommendations, planned_transfers
        )
        
        # Calculate optimized squad EV
        optimized_ev = self._calculate_optimized_ev(
            current_squad_ev, immediate_recommendations
        )
        
        result = {
            'current_gameweek': current_gw,
            'planning_horizon': num_weeks,
            'squad_predictions': squad_predictions,
            'fixture_analysis': fixture_analysis,
            'immediate_recommendations': immediate_recommendations,
            'planned_transfers': planned_transfers,
            'alternative_strategies': alternative_strategies,
            'expected_value': {
                'current_squad': current_squad_ev,
                'optimized_squad': optimized_ev,
                'potential_gain': round(optimized_ev - current_squad_ev, 1)
            },
            'model_metrics': self.predictor.get_model_metrics(),
            'model_confidence': self.predictor.get_confidence_level(),
            'mip_recommendation': None
        }
        
        # Run MIP solver if requested (PRIORITY MODEL)
        if use_mip and MIP_AVAILABLE:
            mip_result = self._run_mip_solver(
                squad_analysis,
                num_weeks,
                mip_time_limit,
                mip_candidate_pool,
                free_transfers,
                current_gw
            )
            result['mip_recommendation'] = mip_result
            
            # If MIP found a solution, use it to EXCLUSIVELY drive the strategy
            if mip_result.get('status') == 'optimal' and mip_result.get('transfers_in'):
                try:
                    # Convert MIP plan to standard format for compatibility (though we might hide it in report)
                    result['planned_transfers'] = self._convert_mip_to_plan(
                        mip_result, current_gw
                    )
                    
                    # UPDATE EV based on MIP
                    if mip_result.get('expected_points'):
                        result['expected_value']['optimized_squad'] = mip_result['expected_points']
                        result['expected_value']['potential_gain'] = round(
                            mip_result['expected_points'] - current_squad_ev, 1
                        )
                    
                    # CRITICAL: Remove heuristic recommendations to satisfy "Only 1 Optimal Strategy" rule
                    result['immediate_recommendations'] = [] 
                    result['alternative_strategies'] = {}
                    
                except Exception as e:
                    print(f"Failed to convert MIP plan: {e}")
                    import traceback
                    traceback.print_exc()
            
            # If MIP is optimal but suggests NO transfers
            elif mip_result.get('status') == 'optimal':
                 result['planned_transfers'] = []
                 result['immediate_recommendations'] = []
                 result['alternative_strategies'] = {}
                 result['expected_value']['optimized_squad'] = current_squad_ev
                 result['expected_value']['potential_gain'] = 0.0

        elif use_mip and not MIP_AVAILABLE:
            result['mip_recommendation'] = {
                'status': 'unavailable',
                'message': 'MIP solver not available. Install sasoptpy and highspy.'
            }
        
        return result
    
    def _run_mip_solver(self, squad_analysis: List[Dict], 
                        num_weeks: int,
                        time_limit: float,
                        candidate_pool: int,
                        free_transfers: int,
                        current_gw: int = None) -> Dict:
        """Run the MIP solver and return results as a dict.
        
        Args:
            squad_analysis: Current squad analysis.
            num_weeks: Planning horizon.
            time_limit: Solver time limit.
            candidate_pool: Candidates per position.
            free_transfers: Available free transfers.
            
        Returns:
            Dict with solver results suitable for LaTeX rendering.
        """
        try:
            # Get current squad from fetcher
            if current_gw is None:
                current_gw = self.fetcher.get_current_gameweek()
                
            current_squad = self.fetcher.get_current_squad(current_gw)
            bank = self.fetcher.get_bank()
            
            # Build xP matrix for all candidates
            # First, get all player IDs we'll need predictions for
            squad_ids = [p['id'] for p in current_squad]
            
            # Get top players from each position
            top_players = self.fetcher.players_df.nlargest(
                candidate_pool * 4, 'total_points'
            )['id'].tolist()
            
            all_ids = list(set(squad_ids + top_players))
            
            # Get predictions
            predictions = self._predict_multiple_gws(all_ids, num_weeks)
            
            # Convert to xP matrix format {id: [xp1, xp2, ...]}
            xp_matrix = {}
            for pid, pred in predictions.items():
                xp_matrix[pid] = pred.get('predictions', [0.0] * num_weeks)
            
            # Get teams data for name lookup
            teams_data = self.fetcher.bootstrap_data.get('teams', [])
            
            # Create and run solver
            solver = TransferMIPSolver(
                current_squad=current_squad,
                bank=bank,
                players_df=self.fetcher.players_df,
                xp_matrix=xp_matrix,
                free_transfers=free_transfers,
                horizon=num_weeks,
                candidate_pool_size=candidate_pool,
                time_limit=time_limit,
                teams_data=teams_data
            )
            
            result = solver.solve()
            
            # Convert MIPSolverResult to dict for JSON/LaTeX compatibility
            return {
                'status': result.status,
                'transfers_out': result.transfers_out,
                'transfers_in': result.transfers_in,
                'starting_xi': result.starting_xi,
                'bench': result.bench,
                'formation': result.formation,
                'captain': result.captain,
                'vice_captain': result.vice_captain,
                'hit_cost': result.hit_cost,
                'num_transfers': result.num_transfers,
                'free_transfers_used': result.free_transfers_used,
                'budget_remaining': result.budget_remaining,
                'expected_points': result.expected_points,
                'per_gw_xp': result.per_gw_xp,
                'solver_time': result.solver_time,
                'message': result.message
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'MIP solver error: {str(e)}',
                'transfers_out': [],
                'transfers_in': [],
                'hit_cost': 0,
                'num_transfers': 0,
                'expected_points': 0
            }

    def _analyze_squad_fixtures(self, squad_analysis: List[Dict], 
                                 num_weeks: int) -> Dict[int, Dict]:
        """Analyze fixture difficulty for each squad player.
        
        Returns dict mapping player_id to fixture analysis.
        """
        fixture_analysis = {}
        
        for player in squad_analysis:
            pid = player.get('player_id')
            if not pid:
                continue
            
            # Get player info
            player_name = player.get('name', 'Unknown')
            position = player.get('position', 'UNK')
            
            # Get team_id - try multiple locations
            team_id = player.get('stats', {}).get('team', 0)
            if not team_id:
                team_id = player.get('raw_stats', {}).get('team', 0)
            
            fixtures = self.fetcher.get_upcoming_fixtures(team_id, num_weeks)
            
            if not fixtures:
                fixture_analysis[pid] = {
                    'player_name': player_name,
                    'position': position,
                    'fixtures': [],
                    'avg_difficulty': 3.0,
                    'swing': 'neutral',
                    'swing_gw': None,
                    'difficulty_trend': [3.0] * num_weeks
                }
                continue
            
            difficulties = [f.get('difficulty', 3) for f in fixtures]
            
            # Calculate fixture swing - compare first 2 vs next 3
            early_avg = np.mean(difficulties[:2]) if len(difficulties) >= 2 else 3.0
            late_avg = np.mean(difficulties[2:]) if len(difficulties) > 2 else 3.0
            
            swing_diff = late_avg - early_avg
            if swing_diff < -0.5:
                swing = 'improving'
                swing_gw = self._find_swing_point(difficulties)
            elif swing_diff > 0.5:
                swing = 'worsening'
                swing_gw = self._find_swing_point(difficulties, improving=False)
            else:
                swing = 'neutral'
                swing_gw = None
            
            fixture_analysis[pid] = {
                'player_name': player_name,
                'position': position,
                'fixtures': fixtures,
                'avg_difficulty': round(np.mean(difficulties), 2),
                'swing': swing,
                'swing_gw': swing_gw,
                'difficulty_trend': difficulties,
                'early_avg': round(early_avg, 2),
                'late_avg': round(late_avg, 2)
            }
        
        return fixture_analysis

    def _find_swing_point(self, difficulties: List[float], 
                          improving: bool = True) -> Optional[int]:
        """Find the gameweek where fixture swing occurs."""
        if len(difficulties) < 2:
            return None
            
        threshold = 0.5
        for i in range(1, len(difficulties)):
            diff = difficulties[i-1] - difficulties[i]
            if improving and diff > threshold:
                return i
            elif not improving and diff < -threshold:
                return i
        return None

    def _get_immediate_recommendations(self, underperformers: List[Dict],
                                        squad_predictions: Dict[int, Dict],
                                        fixture_analysis: Dict[int, Dict],
                                        num_weeks: int) -> List[Dict]:
        """Get enhanced transfer recommendations with multi-GW data.
        
        Adds 5-GW expected points and priority indicators to recommendations.
        """
        if not underperformers:
            return []
        
        budget = self.fetcher.get_bank()
        recommendations = []
        
        for player in underperformers[:5]:
            position = player['position']
            current_price = player.get('price', 4.0) or 4.0
            total_budget = budget + current_price
            pid = player.get('player_id')
            
            # Get current player's multi-GW prediction
            current_pred = squad_predictions.get(pid, {})
            current_5gw = current_pred.get('cumulative', 0)
            
            # Get current player's fixture analysis
            current_fixtures = fixture_analysis.get(pid, {})
            current_swing = current_fixtures.get('swing', 'neutral')
            
            # Get candidates
            candidates = self.fetcher.get_all_players_by_position(position)
            candidates = candidates[
                (candidates['now_cost'] / 10 <= total_budget) &
                (candidates['minutes'] >= 90) &
                (candidates['id'] != pid)
            ].copy()
            
            # Filter available players
            available_mask = candidates.apply(
                lambda row: self.recommender._is_player_available(row.to_dict()), 
                axis=1
            )
            candidates = candidates[available_mask]
            
            if candidates.empty:
                continue
            
            # Get multi-GW predictions for candidates
            candidate_ids = candidates['id'].tolist()[:30]
            candidate_predictions = self._predict_multiple_gws(candidate_ids, num_weeks)
            
            # Score and rank candidates
            scored_candidates = []
            for _, cand in candidates.iterrows():
                cand_id = int(cand['id'])
                if cand_id not in candidate_predictions:
                    continue
                
                cand_pred = candidate_predictions[cand_id]
                cand_5gw = cand_pred.get('cumulative', 0)
                expected_gain = cand_5gw - current_5gw
                
                # Get candidate fixture analysis
                cand_team = int(cand.get('team', 0))
                cand_fixtures = self.fetcher.get_upcoming_fixtures(cand_team, num_weeks)
                cand_difficulties = [f.get('difficulty', 3) for f in cand_fixtures]
                cand_avg_fdr = np.mean(cand_difficulties) if cand_difficulties else 3.0
                
                # Determine candidate fixture swing
                if len(cand_difficulties) >= 3:
                    early = np.mean(cand_difficulties[:2])
                    late = np.mean(cand_difficulties[2:])
                    if late - early < -0.5:
                        cand_swing = 'improving'
                    elif late - early > 0.5:
                        cand_swing = 'worsening'
                    else:
                        cand_swing = 'neutral'
                else:
                    cand_swing = 'neutral'
                
                # Calculate priority based on expected gain and urgency
                priority = self._calculate_priority(
                    expected_gain, player.get('severity', 0), 
                    current_swing, cand_swing
                )
                
                if expected_gain > -5:  # Allow slightly negative if fixture swing
                    scored_candidates.append({
                        'player_id': cand_id,
                        'name': cand.get('web_name', 'Unknown'),
                        'team': self.fetcher._get_team_name(cand_team),
                        'price': round(float(cand.get('now_cost', 0)) / 10, 1),
                        'form': round(float(cand.get('form', 0) or 0), 1),
                        'total_points': int(cand.get('total_points', 0) or 0),
                        'ownership': round(float(cand.get('selected_by_percent', 0) or 0), 1),
                        'predicted_points': cand_pred.get('predictions', [0])[0],
                        '5gw_expected': round(cand_5gw, 1),
                        'expected_gain': round(expected_gain, 1),
                        'fixtures': cand_fixtures[:3],
                        'avg_fdr': round(cand_avg_fdr, 2),
                        'fixture_swing': cand_swing,
                        'confidence': cand_pred.get('confidence', 'medium'),
                        'priority': priority
                    })
            
            # Sort by expected gain
            scored_candidates.sort(key=lambda x: x['expected_gain'], reverse=True)
            
            if scored_candidates:
                recommendations.append({
                    'out': {
                        **player,
                        '5gw_expected': round(current_5gw, 1),
                        'fixture_swing': current_swing
                    },
                    'in_options': scored_candidates[:3],
                    'budget_after': round(total_budget - scored_candidates[0]['price'], 1),
                    'best_gain': scored_candidates[0]['expected_gain'],
                    'priority': scored_candidates[0]['priority']
                })
        
        # Sort recommendations by priority and expected gain
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(
            key=lambda x: (priority_order.get(x['priority'], 2), -x.get('best_gain', 0))
        )
        
        return recommendations

    def _calculate_priority(self, expected_gain: float, severity: int,
                            current_swing: str, new_swing: str) -> str:
        """Calculate transfer priority based on multiple factors."""
        score = 0
        
        # Expected gain contribution
        if expected_gain > 10:
            score += 3
        elif expected_gain > 5:
            score += 2
        elif expected_gain > 2:
            score += 1
        
        # Severity contribution (from underperformer analysis)
        if severity >= 6:
            score += 3
        elif severity >= 4:
            score += 2
        elif severity >= 2:
            score += 1
        
        # Fixture swing bonus
        if current_swing == 'worsening' and new_swing == 'improving':
            score += 2
        elif new_swing == 'improving':
            score += 1
        elif current_swing == 'worsening':
            score += 1
        
        # Determine priority
        if score >= 5:
            return 'high'
        elif score >= 3:
            return 'medium'
        else:
            return 'low'

    def _convert_mip_to_plan(self, mip_result: Dict, current_gw: int) -> List[Dict]:
        """Convert MIP solver result to standard transfer plan format.
        
        Args:
            mip_result: Dict returned by _run_mip_solver.
            current_gw: Current gameweek number.
            
        Returns:
            List of transfer dicts compatible with report generator.
        """
        planned = []
        # MIP Returns list of player dicts, not tuples
        transfers_out_list = mip_result.get('transfers_out', [])
        transfers_in_list = mip_result.get('transfers_in', [])
        
        # In the static MIP model, all transfers happen at start (GW+1)
        transfer_gw = current_gw + 1
        
        # Collect IDs for batch prediction
        t_out_ids = [p['id'] for p in transfers_out_list if p.get('id')]
        t_in_ids = [p['id'] for p in transfers_in_list if p.get('id')]
        all_ids = list(set(t_out_ids + t_in_ids))
        
        # Batch predict (5 weeks horizon)
        predictions = {}
        if all_ids:
            predictions = self._predict_multiple_gws(all_ids, num_gws=5)
        
        # Match transfers (N out, M in - we try to pair them for display)
        # Since logic replaces X with Y, we just list them.
        
        # We need to pair them arbitrarily if counts are equal, or just list
        # For the visualization "Out -> In", we need pairs.
        # If lengths differ, we'll use placeholder or "None"
        
        max_len = max(len(transfers_out_list), len(transfers_in_list))
        
        for i in range(max_len):
            p_out = transfers_out_list[i] if i < len(transfers_out_list) else None
            p_in = transfers_in_list[i] if i < len(transfers_in_list) else None
            
            name_out = p_out.get('web_name', p_out.get('name', 'Unknown')) if p_out else '-'
            pos_out = p_out.get('position', 'UNK') if p_out else ''
            id_out = p_out.get('id') if p_out else None
            
            name_in = p_in.get('web_name', p_in.get('name', 'Unknown')) if p_in else '-'
            pos_in = p_in.get('position', 'UNK') if p_in else ''
            id_in = p_in.get('id') if p_in else None
            
            # Gain stats
            gain = 0
            if id_out and id_in:
                pred_out = predictions.get(id_out, {}).get('cumulative', 0)
                pred_in = predictions.get(id_in, {}).get('cumulative', 0)
                gain = round(pred_in - pred_out, 1)
            elif id_in:
                gain = round(predictions.get(id_in, {}).get('cumulative', 0), 1)
                
            reason = f"MIP optimal move"
            if gain > 0:
                reason += f" (+{gain} xP)"
            
            planned.append({
                'gameweek': transfer_gw,
                'action': 'transfer',
                'out': name_out,
                'out_position': pos_out,
                'in': name_in,
                'in_position': pos_in,
                'expected_gain': max(0.1, gain),
                'reasoning': reason,
                'priority': 'high',
                'take_hit': False # Solver handled cost
            })
            
        return planned

    def _plan_transfer_sequence(self, squad_analysis: List[Dict],
                                 recommendations: List[Dict],
                                 num_weeks: int,
                                 current_gw: int,
                                 free_transfers: int) -> List[Dict]:
        """Plan a sequence of transfers over the planning horizon."""
        if not recommendations:
            return []
        
        # Use passed current_gw instead of fetching live
        
        planned = []
        
        # High priority transfers for this week
        high_priority = [r for r in recommendations if r.get('priority') == 'high']
        medium_priority = [r for r in recommendations if r.get('priority') == 'medium']
        
        # Week 1: Handle high priority transfers
        # We allow up to 'free_transfers' to be free, then hits
        num_free_used = 0
        
        for i, rec in enumerate(high_priority[:max(free_transfers, 2)]): 
            best_in = rec['in_options'][0] if rec['in_options'] else None
            if best_in:
                # Is this a free transfer or hit?
                is_hit = num_free_used >= free_transfers
                
                planned.append({
                    'gameweek': current_gw + 1,
                    'action': 'transfer',
                    'out': rec['out']['name'],
                    'out_position': rec['out']['position'],
                    'in': best_in['name'],
                    'in_position': rec['out']['position'],
                    'expected_gain': best_in['expected_gain'],
                    'reasoning': self._generate_reasoning(rec, best_in),
                    'priority': rec['priority'],
                    'take_hit': is_hit
                })
                num_free_used += 1
        
        # Week 2+: Handle medium priority
        for i, rec in enumerate(medium_priority[:2]):
            best_in = rec['in_options'][0] if rec['in_options'] else None
            if best_in:
                planned.append({
                    'gameweek': current_gw + 2 + (i // 2),
                    'action': 'consider',
                    'out': rec['out']['name'],
                    'out_position': rec['out']['position'],
                    'in': best_in['name'],
                    'in_position': rec['out']['position'],
                    'expected_gain': best_in['expected_gain'],
                    'reasoning': self._generate_reasoning(rec, best_in),
                    'priority': rec['priority'],
                    'take_hit': False
                })
        
        return planned

    def _generate_reasoning(self, recommendation: Dict, 
                             candidate: Dict) -> str:
        """Generate human-readable reasoning for a transfer."""
        reasons = []
        
        out_player = recommendation['out']
        
        # Form-based reasoning
        if out_player.get('current_form', 5) < 3:
            reasons.append(f"poor form ({out_player.get('current_form', 0):.1f})")
        
        # Fixture swing reasoning
        out_swing = out_player.get('fixture_swing', 'neutral')
        in_swing = candidate.get('fixture_swing', 'neutral')
        
        if out_swing == 'worsening':
            reasons.append("tough fixtures ahead")
        if in_swing == 'improving':
            reasons.append("favorable fixture run")
        
        # Expected points reasoning
        gain = candidate.get('expected_gain', 0)
        if gain > 5:
            reasons.append(f"+{gain:.1f} xP over 5 GWs")
        
        # Underperformer reasons
        for reason in out_player.get('reasons', [])[:2]:
            if reason not in reasons:
                reasons.append(reason.lower())
        
        if not reasons:
            reasons.append("optimization opportunity")
        
        return "; ".join(reasons[:3])

    def _generate_alternatives(self, recommendations: List[Dict],
                                planned_transfers: List[Dict]) -> Dict:
        """Generate alternative transfer strategies."""
        if not recommendations:
            return {
                'aggressive': {'description': 'No transfers needed', 'transfers': 0, 'hits': 0, 'net_gain': 0},
                'conservative': {'description': 'No transfers needed', 'transfers': 0, 'hits': 0, 'net_gain': 0},
                'wildcard_consideration': False
            }
        
        total_gain = sum(r.get('best_gain', 0) for r in recommendations[:3])
        
        # Aggressive: Multiple transfers with hits
        aggressive_transfers = min(len(recommendations), 3)
        aggressive_hits = max(0, aggressive_transfers - 1)
        aggressive_gain = total_gain - (aggressive_hits * 4)  # 4 point hit cost
        
        # Conservative: Just 1 free transfer
        conservative_gain = recommendations[0].get('best_gain', 0) if recommendations else 0
        
        # Wildcard consideration threshold
        wildcard_threshold = len([r for r in recommendations if r.get('priority') in ['high', 'medium']]) >= 4
        
        return {
            'aggressive': {
                'description': f"{aggressive_transfers} transfers" + (f" (-{aggressive_hits * 4} hit)" if aggressive_hits else ""),
                'transfers': aggressive_transfers,
                'hits': aggressive_hits,
                'net_gain': round(aggressive_gain, 1),
                'recommended': aggressive_gain > conservative_gain + 4  # Worth if net gain > 4
            },
            'conservative': {
                'description': "1 transfer (free)",
                'transfers': 1,
                'hits': 0,
                'net_gain': round(conservative_gain, 1),
                'recommended': not (aggressive_gain > conservative_gain + 4)
            },
            'wildcard_consideration': wildcard_threshold,
            'wildcard_reason': "4+ beneficial transfers identified" if wildcard_threshold else None
        }

    def _calculate_squad_ev(self, squad_predictions: Dict[int, Dict]) -> float:
        """Calculate total expected value for current squad."""
        return round(sum(
            pred.get('cumulative', 0) 
            for pred in squad_predictions.values()
        ), 1)

    def _calculate_optimized_ev(self, current_ev: float, 
                                 recommendations: List[Dict]) -> float:
        """Calculate expected value if all recommended transfers are made."""
        total_gain = sum(
            rec.get('best_gain', 0) 
            for rec in recommendations[:3]  # Max 3 transfers
        )
        return round(current_ev + total_gain, 1)

    def get_fixture_heatmap_data(self, squad_analysis: List[Dict], 
                                  num_weeks: int = 5) -> List[Dict]:
        """Get fixture difficulty data formatted for heatmap visualization.
        
        Returns list of dicts with player names and fixture difficulties.
        """
        heatmap_data = []
        
        for player in squad_analysis:
            pid = player.get('player_id')
            name = player.get('name', 'Unknown')
            position = player.get('position', 'UNK')
            team_id = player.get('stats', {}).get('team', 0)
            if not team_id:
                team_id = player.get('raw_stats', {}).get('team', 0)
            
            fixtures = self.fetcher.get_upcoming_fixtures(team_id, num_weeks)
            
            row = {
                'player': name,
                'position': position,
                'fixtures': []
            }
            
            for i in range(num_weeks):
                if i < len(fixtures):
                    fix = fixtures[i]
                    row['fixtures'].append({
                        'opponent': fix.get('opponent', '?'),
                        'is_home': fix.get('is_home', False),
                        'difficulty': fix.get('difficulty', 3),
                        'gameweek': fix.get('gameweek', 0)
                    })
                else:
                    row['fixtures'].append({
                        'opponent': 'BGW',
                        'is_home': False,
                        'difficulty': 0,
                        'gameweek': 0
                    })
            
            heatmap_data.append(row)
        
        # Sort by position
        position_order = {'GKP': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
        heatmap_data.sort(key=lambda x: position_order.get(x['position'], 4))
        
        return heatmap_data


class WildcardOptimizer:
    """Builds an optimal 15-player Wildcard squad within budget constraints.
    
    Constraints:
    - 2 GKP, 5 DEF, 5 MID, 3 FWD
    - Max 3 players per real club
    - Total cost <= total_budget
    - Only available players (status='a', chance_of_playing >= 75)
    
    Scoring approach: "Season/balanced" - prioritizes consistent long-term
    performers over short-term fixture runs. Uses cheap bench strategy.
    """
    
    POSITION_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    POSITION_QUOTAS = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    MAX_PER_TEAM = 3
    
    # Minimum players needed in XI by position
    XI_MIN = {'GKP': 1, 'DEF': 3, 'MID': 2, 'FWD': 1}
    XI_MAX = {'GKP': 1, 'DEF': 5, 'MID': 5, 'FWD': 3}
    
    def __init__(self, players_df, total_budget: float, predictions: Dict[int, Dict] = None,
                 teams_data: Optional[List[Dict]] = None, fixtures_data: Optional[List[Dict]] = None,
                 current_gw: int = 1):
        """Initialize the optimizer.
        
        Args:
            players_df: DataFrame with all FPL players from bootstrap data.
            total_budget: Total budget in millions (squad_value + bank).
            predictions: Optional dict of player_id -> prediction data from FPLPointsPredictor.
            teams_data: List of team dicts from bootstrap data (for team names).
            fixtures_data: List of fixture dicts for FDR lookup.
            current_gw: Current gameweek number for fixture calculation.
        """
        self.players_df = players_df.copy()
        self.total_budget = total_budget
        self.predictions = predictions or {}
        self.current_gw = current_gw
        
        # Build team ID -> short name mapping
        self.team_names = {}
        if teams_data:
            for team in teams_data:
                self.team_names[team['id']] = team.get('short_name', 'UNK')
        
        # Build team ID -> upcoming fixtures mapping
        self.team_fixtures = {}
        if fixtures_data:
            self._build_fixture_map(fixtures_data)
        
        # Add position labels
        self.players_df['position'] = self.players_df['element_type'].map(self.POSITION_MAP)
        # Convert cost to millions
        self.players_df['price'] = self.players_df['now_cost'] / 10.0
    
    def _build_fixture_map(self, fixtures_data: List[Dict]):
        """Build a map of team_id -> list of next 5 fixtures with FDR."""
        for team_id in range(1, 21):
            team_fixtures = []
            for fix in fixtures_data:
                event = fix.get('event')
                if event is None or event <= self.current_gw:
                    continue
                
                if fix.get('team_h') == team_id:
                    opponent_id = fix.get('team_a')
                    fdr = fix.get('team_h_difficulty', 3)
                    is_home = True
                elif fix.get('team_a') == team_id:
                    opponent_id = fix.get('team_h')
                    fdr = fix.get('team_a_difficulty', 3)
                    is_home = False
                else:
                    continue
                
                team_fixtures.append({
                    'gw': event,
                    'opponent': self.team_names.get(opponent_id, '?'),
                    'fdr': fdr,
                    'is_home': is_home
                })
            
            team_fixtures.sort(key=lambda x: x['gw'])
            self.team_fixtures[team_id] = team_fixtures[:5]
        
    def _is_available(self, row) -> bool:
        """Check if player is available to play."""
        status = row.get('status', 'a')
        if status != 'a':
            return False
        chance = row.get('chance_of_playing_next_round')
        if chance is not None and chance < 75:
            return False
        return True
    
    def _calculate_score(self, row) -> float:
        """Calculate score for a player, prioritizing xP predictions and premium value.
        
        Strategy:
        - If predictions available: Use xp_5gw as primary metric (80%) + historical tiebreaker (20%)
        - If no predictions: Fall back to historical "season-balanced" scoring
        - Add PREMIUM BONUS: Expensive players get extra points since unspent budget is wasted
        
        This ensures the Wildcard optimizer selects players with highest PREDICTED points,
        while also favoring premium assets to utilize full budget.
        """
        player_id = int(row.get('id', 0))
        price = float(row.get('now_cost', 50) / 10)  # Price in millions
        
        # Premium player bonus: reward expensive players to ensure budget utilization
        # Players over 10m get significant bonus, scaled by price
        if price >= 12:
            premium_bonus = 8.0  # Elite tier (Haaland, Salah class)
        elif price >= 10:
            premium_bonus = 5.0  # Premium tier
        elif price >= 8:
            premium_bonus = 2.0  # Mid-premium tier
        elif price >= 6:
            premium_bonus = 1.0  # Budget-friendly
        else:
            premium_bonus = 0.0  # Fodder
        
        # Check if we have predictions for this player
        if player_id in self.predictions:
            pred_data = self.predictions[player_id]
            xp_5gw = pred_data.get('cumulative', 0)
            
            # Historical tiebreaker (small weight)
            ppg = float(row.get('points_per_game', 0) or 0)
            form = float(row.get('form', 0) or 0)
            historical_boost = (ppg * 0.5 + form * 0.5)  # 0-10 scale
            
            # Primary score: xP (scaled to ~0-50 range) + tiebreaker + premium bonus
            # This ensures xP dominates but ties are broken by historical form and price tier
            score = xp_5gw + (historical_boost * 0.2) + premium_bonus
            return round(score, 2)
        
        # Fallback: Historical scoring when no predictions available
        ppg = float(row.get('points_per_game', 0) or 0)
        total_pts = float(row.get('total_points', 0) or 0)
        minutes = float(row.get('minutes', 0) or 0)
        xgi = float(row.get('expected_goal_involvements', 0) or 0)
        form = float(row.get('form', 0) or 0)
        
        # Normalize to 0-10 scales
        minutes_score = min(minutes / 90.0, 10.0)
        pts_score = min(total_pts / 10.0, 10.0)
        xgi_score = min(xgi * 2.0, 10.0)
        ppg_score = min(ppg, 10.0)
        form_score = min(form, 10.0)
        
        score = (
            ppg_score * 0.40 +
            pts_score * 0.20 +
            minutes_score * 0.20 +
            xgi_score * 0.10 +
            form_score * 0.10
        ) + premium_bonus
        
        return round(score, 2)
    
    def _get_available_players(self, position: str) -> list:
        """Get available players for a position, sorted by score descending."""
        pos_df = self.players_df[self.players_df['position'] == position].copy()
        
        # Filter available players
        available_mask = pos_df.apply(self._is_available, axis=1)
        pos_df = pos_df[available_mask]
        
        # Filter players with some minutes (at least 45)
        pos_df = pos_df[pos_df['minutes'] >= 45]
        
        # Calculate scores
        pos_df['score'] = pos_df.apply(self._calculate_score, axis=1)
        
        # Sort by score descending
        pos_df = pos_df.sort_values('score', ascending=False)
        
        return pos_df.to_dict('records')
    
    def _get_cheapest_players(self, position: str, exclude_ids: set) -> list:
        """Get cheapest available players for bench filling."""
        pos_df = self.players_df[self.players_df['position'] == position].copy()
        
        # Filter available players
        available_mask = pos_df.apply(self._is_available, axis=1)
        pos_df = pos_df[available_mask]
        
        # Exclude already selected
        pos_df = pos_df[~pos_df['id'].isin(exclude_ids)]
        
        # Sort by price ascending
        pos_df = pos_df.sort_values('price', ascending=True)
        
        return pos_df.to_dict('records')
    
    def build_squad(self) -> Dict:
        """Build optimal 15-player squad using greedy selection with cheap bench.
        
        Strategy:
        1. Select best XI first (greedy by score, respecting constraints)
        2. Fill remaining squad slots with cheapest available players
        3. Determine best starting XI formation
        4. Select captain and vice captain
        
        Returns:
            Dict with squad, starting_xi, bench, formation, captain, vice_captain, budget info.
        """
        squad = []
        team_counts = {}
        position_counts = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        selected_ids = set()
        spent = 0.0
        
        # Phase 1: Build a strong starting XI (11 players)
        # Target: 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD (valid formations)
        xi_targets = {'GKP': 1, 'DEF': 4, 'MID': 4, 'FWD': 2}  # Default 4-4-2
        
        # Get sorted players by position
        candidates_by_pos = {}
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            candidates_by_pos[pos] = self._get_available_players(pos)
        
        # Greedy selection for XI
        for pos in ['GKP', 'FWD', 'MID', 'DEF']:  # Priority order
            target = xi_targets[pos]
            candidates = candidates_by_pos[pos]
            
            for player in candidates:
                if position_counts[pos] >= target:
                    break
                    
                pid = player['id']
                price = player['price']
                team_id = player['team']
                
                # Check constraints
                if pid in selected_ids:
                    continue
                if team_counts.get(team_id, 0) >= self.MAX_PER_TEAM:
                    continue
                if spent + price > self.total_budget - self._estimate_remaining_cost(position_counts, pos):
                    continue
                
                # Select player
                squad.append(self._format_player(player, is_starter=True))
                selected_ids.add(pid)
                position_counts[pos] += 1
                team_counts[team_id] = team_counts.get(team_id, 0) + 1
                spent += price
        
        # Phase 2: Fill remaining quota with cheap bench players
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            remaining = self.POSITION_QUOTAS[pos] - position_counts[pos]
            if remaining <= 0:
                continue
                
            cheap_players = self._get_cheapest_players(pos, selected_ids)
            
            for player in cheap_players:
                if remaining <= 0:
                    break
                    
                pid = player['id']
                price = player['price']
                team_id = player['team']
                
                # Check constraints
                if team_counts.get(team_id, 0) >= self.MAX_PER_TEAM:
                    continue
                if spent + price > self.total_budget:
                    continue
                
                # Add score for bench player
                player['score'] = self._calculate_score(player)
                
                # Select player
                squad.append(self._format_player(player, is_starter=False))
                selected_ids.add(pid)
                position_counts[pos] += 1
                team_counts[team_id] = team_counts.get(team_id, 0) + 1
                spent += price
                remaining -= 1
        
        # Determine best starting XI and formation
        starting_xi, bench, formation = self._select_starting_xi(squad)
        
        # Select captain and vice captain (highest scorers in XI)
        captain, vice_captain = self._select_captains(starting_xi)
        
        return {
            'squad': squad,
            'starting_xi': starting_xi,
            'bench': bench,
            'formation': formation,
            'captain': captain,
            'vice_captain': vice_captain,
            'budget': {
                'total': self.total_budget,
                'spent': round(spent, 1),
                'remaining': round(self.total_budget - spent, 1)
            },
            'team_counts': team_counts,
            'position_counts': position_counts
        }
    
    def _estimate_remaining_cost(self, current_counts: Dict, current_pos: str) -> float:
        """Estimate minimum cost to fill remaining squad slots."""
        # Get minimum prices per position
        min_prices = {}
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            pos_players = self.players_df[self.players_df['position'] == pos]
            available = pos_players[pos_players.apply(self._is_available, axis=1)]
            if not available.empty:
                min_prices[pos] = available['price'].min()
            else:
                min_prices[pos] = 4.0  # Fallback
        
        remaining_cost = 0.0
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            needed = self.POSITION_QUOTAS[pos] - current_counts.get(pos, 0)
            if pos == current_pos:
                needed -= 1  # Account for current selection
            remaining_cost += max(0, needed) * min_prices[pos]
        
        return remaining_cost
    
    def _format_player(self, player: Dict, is_starter: bool) -> Dict:
        """Format player dict for output."""
        player_id = int(player['id'])
        team_id = int(player.get('team', 0))
        team_name = self.team_names.get(team_id, 'UNK')
        fixtures = self.team_fixtures.get(team_id, [])
        
        # Get 5-GW xP from predictions if available
        xp_5gw = 0.0
        if player_id in self.predictions:
            pred_data = self.predictions[player_id]
            xp_5gw = pred_data.get('cumulative', 0.0)
        
        return {
            'id': player_id,
            'name': player.get('web_name', 'Unknown'),
            'position': player.get('position', 'UNK'),
            'team': team_name,  # Now returns short name like "CHE", "MCI"
            'team_id': team_id,
            'price': round(float(player.get('price', 0)), 1),
            'score': round(float(player.get('score', 0)), 2),
            'total_points': int(player.get('total_points', 0) or 0),
            'ppg': round(float(player.get('points_per_game', 0) or 0), 2),
            'form': round(float(player.get('form', 0) or 0), 1),
            'minutes': int(player.get('minutes', 0) or 0),
            'is_starter': is_starter,
            'fixtures': fixtures,  # List of {gw, opponent, fdr, is_home}
            'xp_5gw': round(xp_5gw, 1)  # 5-gameweek expected points
        }
    
    def _select_starting_xi(self, squad: List[Dict]) -> Tuple[List[Dict], List[Dict], str]:
        """Select best starting XI from squad with valid formation.
        
        Valid formations: 3-4-3, 3-5-2, 4-3-3, 4-4-2, 4-5-1, 5-3-2, 5-4-1
        """
        # Group by position
        by_pos = {'GKP': [], 'DEF': [], 'MID': [], 'FWD': []}
        for p in squad:
            by_pos[p['position']].append(p)
        
        # Sort each position by score descending
        for pos in by_pos:
            by_pos[pos].sort(key=lambda x: x['score'], reverse=True)
        
        # Try formations in order of preference
        formations = [
            ('4-4-2', {'GKP': 1, 'DEF': 4, 'MID': 4, 'FWD': 2}),
            ('4-3-3', {'GKP': 1, 'DEF': 4, 'MID': 3, 'FWD': 3}),
            ('3-5-2', {'GKP': 1, 'DEF': 3, 'MID': 5, 'FWD': 2}),
            ('3-4-3', {'GKP': 1, 'DEF': 3, 'MID': 4, 'FWD': 3}),
            ('5-4-1', {'GKP': 1, 'DEF': 5, 'MID': 4, 'FWD': 1}),
            ('5-3-2', {'GKP': 1, 'DEF': 5, 'MID': 3, 'FWD': 2}),
            ('4-5-1', {'GKP': 1, 'DEF': 4, 'MID': 5, 'FWD': 1}),
        ]
        
        best_xi = None
        best_formation = None
        best_score = -1
        
        for form_name, form_counts in formations:
            # Check if we have enough players
            valid = True
            for pos, count in form_counts.items():
                if len(by_pos[pos]) < count:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Build XI for this formation
            xi = []
            for pos, count in form_counts.items():
                xi.extend(by_pos[pos][:count])
            
            # Calculate total score
            total_score = sum(p['score'] for p in xi)
            
            if total_score > best_score:
                best_score = total_score
                best_xi = xi
                best_formation = form_name
        
        # Fallback if no formation works
        if best_xi is None:
            best_xi = squad[:11]
            best_formation = '?-?-?'
        
        # Determine bench (players not in XI)
        xi_ids = {p['id'] for p in best_xi}
        bench = [p for p in squad if p['id'] not in xi_ids]
        
        # Sort bench: GKP first, then by score
        bench.sort(key=lambda x: (0 if x['position'] == 'GKP' else 1, -x['score']))
        
        return best_xi, bench, best_formation
    
    def _select_captains(self, starting_xi: List[Dict]) -> Tuple[Dict, Dict]:
        """Select captain and vice captain from starting XI.
        
        Prioritizes xp_5gw (predicted points) over historical score.
        """
        # Sort by xp_5gw descending, then by score as tiebreaker
        sorted_xi = sorted(starting_xi, key=lambda x: (x.get('xp_5gw', 0), x.get('score', 0)), reverse=True)
        
        # Captain is highest xP, VC is second highest
        captain = sorted_xi[0] if sorted_xi else {}
        vice_captain = sorted_xi[1] if len(sorted_xi) > 1 else captain
        
        return captain, vice_captain


class FreeHitOptimizer:
    """Builds an optimal Free Hit squad optimized for a single gameweek.
    
    Constraints (same as FPL rules):
    - 2 GKP, 5 DEF, 5 MID, 3 FWD
    - Max 3 players per real club
    - Total cost <= total_budget
    - Only available players (status='a', chance_of_playing >= 75)
    
    Scoring approach: Uses ep_next (expected points next GW) as primary metric,
    with league ownership adjustment to find differentials that beat your league.
    
    Strategy modes:
    - 'balanced': Strong template picks + 2-4 differentials (default)
    - 'safe': Maximize overlap with league template, minimal differentials
    - 'aggressive': Chase upside with more differentials, including captain
    """
    
    POSITION_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    POSITION_QUOTAS = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    MAX_PER_TEAM = 3
    
    # Minimum players needed in XI by position
    XI_MIN = {'GKP': 1, 'DEF': 3, 'MID': 2, 'FWD': 1}
    XI_MAX = {'GKP': 1, 'DEF': 5, 'MID': 5, 'FWD': 3}
    
    def __init__(
        self, 
        players_df, 
        total_budget: float, 
        league_ownership: Optional[Dict] = None,
        strategy: str = 'balanced',
        teams_data: Optional[List[Dict]] = None,
        fixtures_data: Optional[List[Dict]] = None,
        current_gw: int = 1,
        target_gw: int = None,
        predictions: Optional[Dict] = None
    ):
        """Initialize the Free Hit optimizer.
        
        Args:
            players_df: DataFrame with all FPL players from bootstrap data.
            total_budget: Total budget in millions (squad_value + bank).
            league_ownership: Optional dict from compute_league_ownership() containing:
                - ownership: {player_id: fraction}
                - captain_counts: {player_id: count}
                - sample_size: int
            strategy: One of 'safe', 'balanced', 'aggressive'.
            teams_data: List of team dicts from bootstrap data (for team names).
            fixtures_data: List of fixture dicts for FDR lookup.
            current_gw: Current gameweek number.
            target_gw: Target GW for Free Hit (fixtures are optimized for this GW).
            predictions: Optional dict of {player_id: prediction_data} with multi-GW xP.
        """
        self.players_df = players_df.copy()
        self.total_budget = total_budget
        self.league_ownership = league_ownership or {}
        self.strategy = strategy
        self.current_gw = current_gw
        self.target_gw = target_gw if target_gw else current_gw + 1
        self.predictions = predictions or {}
        
        # Build team ID -> short name mapping
        self.team_names = {}
        if teams_data:
            for team in teams_data:
                self.team_names[team['id']] = team.get('short_name', 'UNK')
        
        # Build team ID -> upcoming fixtures mapping
        self.team_fixtures = {}
        if fixtures_data:
            self._build_fixture_map(fixtures_data)
        
        # Extract ownership map
        self.ownership_map = self.league_ownership.get('ownership', {})
        self.sample_size = self.league_ownership.get('sample_size', 0)
        
        # Add position labels
        self.players_df['position'] = self.players_df['element_type'].map(self.POSITION_MAP)
        # Convert cost to millions
        self.players_df['price'] = self.players_df['now_cost'] / 10.0
        
        # Strategy-specific parameters for differential bonus
        # Bonus formula: diff_bonus = base_bonus * (1 - league_ownership)
        # Higher base_bonus = more aggressive differential seeking
        self.strategy_params = {
            'safe': {'diff_bonus': 0.5, 'max_differentials': 2},
            'balanced': {'diff_bonus': 1.5, 'max_differentials': 4},
            'aggressive': {'diff_bonus': 3.0, 'max_differentials': 6}
        }
    
    def _build_fixture_map(self, fixtures_data: List[Dict]):
        """Build a map of team_id -> fixture for target GW (and next 5 for context)."""
        # Group fixtures by team, prioritizing target GW
        for team_id in range(1, 21):  # FPL teams are 1-20
            team_fixtures = []
            for fix in fixtures_data:
                event = fix.get('event')
                if event is None:
                    continue
                
                if fix.get('team_h') == team_id:
                    opponent_id = fix.get('team_a')
                    fdr = fix.get('team_h_difficulty', 3)
                    is_home = True
                elif fix.get('team_a') == team_id:
                    opponent_id = fix.get('team_h')
                    fdr = fix.get('team_a_difficulty', 3)
                    is_home = False
                else:
                    continue
                
                team_fixtures.append({
                    'gw': event,
                    'opponent': self.team_names.get(opponent_id, '?'),
                    'fdr': fdr,
                    'is_home': is_home
                })
            
            # Sort by gameweek
            team_fixtures.sort(key=lambda x: x['gw'])
            
            # Filter to target GW and beyond (for Free Hit, we care about target GW fixture)
            target_fixtures = [f for f in team_fixtures if f['gw'] >= self.target_gw]
            self.team_fixtures[team_id] = target_fixtures[:5] if target_fixtures else team_fixtures[-5:]
        
    def _is_available(self, row) -> bool:
        """Check if player is available to play."""
        status = row.get('status', 'a')
        if status != 'a':
            return False
        chance = row.get('chance_of_playing_next_round')
        if chance is not None and chance < 75:
            return False
        return True
    
    def _calculate_score(self, row) -> float:
        """Calculate Free Hit score optimized for single GW performance.
        
        Components:
        1. ML predictions (fixture-aware) - primary if available
        2. ep_next from FPL API - fallback
        3. Fixture difficulty bonus - boost players with easy fixtures
        4. Home advantage bonus
        5. Differential bonus - based on league ownership
        """
        player_id = int(row.get('id', 0))
        team_id = int(row.get('team', 0) or 0)
        
        # -----------------------------------------------------------------
        # 1. Base score: Use ML predictions if available (fixture-aware)
        # -----------------------------------------------------------------
        base_score = 0.0
        
        if self.predictions and player_id in self.predictions:
            pred_data = self.predictions[player_id]
            # Predictions are indexed from current_gw+1
            # So for target_gw, we need index = target_gw - current_gw - 1
            preds = pred_data.get('predictions', [])
            pred_index = self.target_gw - self.current_gw - 1
            if preds and 0 <= pred_index < len(preds):
                base_score = preds[pred_index]
        
        # Fallback to ep_next if no ML prediction
        if base_score <= 0:
            ep_next = float(row.get('ep_next', 0) or 0)
            if ep_next > 0:
                base_score = ep_next
            else:
                # Last resort: form-based estimate
                form = float(row.get('form', 0) or 0)
                ppg = float(row.get('points_per_game', 0) or 0)
                base_score = (form + ppg) / 2.0
        
        # -----------------------------------------------------------------
        # 2. Fixture difficulty bonus (FDR) - CRITICAL for Free Hit
        # -----------------------------------------------------------------
        fixture_bonus = 0.0
        home_bonus = 0.0
        
        if team_id in self.team_fixtures and self.team_fixtures[team_id]:
            next_fix = self.team_fixtures[team_id][0]  # Target GW fixture
            fdr = next_fix.get('fdr', 3)
            is_home = next_fix.get('is_home', False)
            
            # FDR bonus: Lower FDR = easier fixture = SIGNIFICANT bonus
            # For Free Hit, fixture is crucial - amplify the impact
            # FDR 1 = +3.0, FDR 2 = +1.5, FDR 3 = 0, FDR 4 = -1.5, FDR 5 = -3.0
            fdr_bonus_map = {1: 3.0, 2: 1.5, 3: 0.0, 4: -1.5, 5: -3.0}
            fixture_bonus = fdr_bonus_map.get(fdr, 0.0)
            
            # Home advantage bonus (meaningful for Free Hit)
            if is_home:
                home_bonus = 1.0
        
        # -----------------------------------------------------------------
        # 3. Differential bonus (league ownership)
        # -----------------------------------------------------------------
        diff_bonus = 0.0
        if self.ownership_map and self.sample_size > 0:
            league_own = self.ownership_map.get(player_id, 0.0)
            
            # Differential bonus: higher for lower ownership
            params = self.strategy_params.get(self.strategy, self.strategy_params['balanced'])
            diff_multiplier = params['diff_bonus']
            diff_bonus = diff_multiplier * ((1 - league_own) ** 2)
        
        # -----------------------------------------------------------------
        # 4. Value premium bonus (encourages spending budget on Free Hit)
        # -----------------------------------------------------------------
        price = float(row.get('price', 0) or row.get('now_cost', 0) / 10.0)
        # Premium players (8m+) get a small bonus to encourage spending
        # This helps ensure we use the full budget on quality players
        value_bonus = 0.0
        if price >= 10.0:
            value_bonus = 1.0  # Premium (10m+)
        elif price >= 8.0:
            value_bonus = 0.5  # Mid-premium (8-10m)
        
        # -----------------------------------------------------------------
        # 5. Total score
        # -----------------------------------------------------------------
        score = base_score + fixture_bonus + home_bonus + diff_bonus + value_bonus
        
        return round(score, 3)
    
    def _get_available_players(self, position: str) -> list:
        """Get available players for a position, sorted by score descending."""
        pos_df = self.players_df[self.players_df['position'] == position].copy()
        
        # Filter available players
        available_mask = pos_df.apply(self._is_available, axis=1)
        pos_df = pos_df[available_mask]
        
        # Filter players with some minutes (at least 45)
        pos_df = pos_df[pos_df['minutes'] >= 45]
        
        # Calculate scores
        pos_df['score'] = pos_df.apply(self._calculate_score, axis=1)
        
        # Sort by score descending
        pos_df = pos_df.sort_values('score', ascending=False)
        
        return pos_df.to_dict('records')
    
    def _get_cheapest_players(self, position: str, exclude_ids: set) -> list:
        """Get cheapest available players for bench filling."""
        pos_df = self.players_df[self.players_df['position'] == position].copy()
        
        # Filter available players
        available_mask = pos_df.apply(self._is_available, axis=1)
        pos_df = pos_df[available_mask]
        
        # Exclude already selected
        pos_df = pos_df[~pos_df['id'].isin(exclude_ids)]
        
        # Sort by price ascending
        pos_df = pos_df.sort_values('price', ascending=True)
        
        return pos_df.to_dict('records')
    
    def build_squad(self) -> Dict:
        """Build optimal 15-player Free Hit squad.
        
        Strategy:
        1. Select best XI first (greedy by score, respecting constraints)
        2. Fill remaining squad slots with cheapest available players
        3. Determine best starting XI formation
        4. Select captain and vice captain
        
        Returns:
            Dict with squad, starting_xi, bench, formation, captain, vice_captain,
            budget info, and league analysis data.
        """
        squad = []
        team_counts = {}
        position_counts = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        selected_ids = set()
        spent = 0.0
        
        # Phase 1: Build a strong starting XI (11 players)
        # Target: 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD (valid formations)
        xi_targets = {'GKP': 1, 'DEF': 4, 'MID': 4, 'FWD': 2}  # Default 4-4-2
        
        # Get sorted players by position
        candidates_by_pos = {}
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            candidates_by_pos[pos] = self._get_available_players(pos)
        
        # Greedy selection for XI
        for pos in ['GKP', 'FWD', 'MID', 'DEF']:  # Priority order
            target = xi_targets[pos]
            candidates = candidates_by_pos[pos]
            
            for player in candidates:
                if position_counts[pos] >= target:
                    break
                    
                pid = player['id']
                price = player['price']
                team_id = player['team']
                
                # Check constraints
                if pid in selected_ids:
                    continue
                if team_counts.get(team_id, 0) >= self.MAX_PER_TEAM:
                    continue
                if spent + price > self.total_budget - self._estimate_remaining_cost(position_counts, pos):
                    continue
                
                # Select player
                squad.append(self._format_player(player, is_starter=True))
                selected_ids.add(pid)
                position_counts[pos] += 1
                team_counts[team_id] = team_counts.get(team_id, 0) + 1
                spent += price
        
        # Phase 2: Fill remaining quota with cheap bench players
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            remaining = self.POSITION_QUOTAS[pos] - position_counts[pos]
            if remaining <= 0:
                continue
                
            cheap_players = self._get_cheapest_players(pos, selected_ids)
            
            for player in cheap_players:
                if remaining <= 0:
                    break
                    
                pid = player['id']
                price = player['price']
                team_id = player['team']
                
                # Check constraints
                if team_counts.get(team_id, 0) >= self.MAX_PER_TEAM:
                    continue
                if spent + price > self.total_budget:
                    continue
                
                # Add score for bench player
                player['score'] = self._calculate_score(player)
                
                # Select player
                squad.append(self._format_player(player, is_starter=False))
                selected_ids.add(pid)
                position_counts[pos] += 1
                team_counts[team_id] = team_counts.get(team_id, 0) + 1
                spent += price
                remaining -= 1
        
        # Determine best starting XI and formation
        starting_xi, bench, formation = self._select_starting_xi(squad)
        
        # Select captain and vice captain (highest scorers in XI)
        captain, vice_captain = self._select_captains(starting_xi)
        
        # Analyze differentials vs league
        differentials = self._identify_differentials(starting_xi)
        template_picks = self._identify_template_picks(starting_xi)
        
        return {
            'squad': squad,
            'starting_xi': starting_xi,
            'bench': bench,
            'formation': formation,
            'captain': captain,
            'vice_captain': vice_captain,
            'budget': {
                'total': self.total_budget,
                'spent': round(spent, 1),
                'remaining': round(self.total_budget - spent, 1)
            },
            'team_counts': team_counts,
            'position_counts': position_counts,
            'strategy': self.strategy,
            'league_analysis': {
                'sample_size': self.sample_size,
                'differentials': differentials,
                'template_picks': template_picks
            }
        }
    
    def _identify_differentials(self, starting_xi: List[Dict]) -> List[Dict]:
        """Identify players in XI with low league ownership (differentials)."""
        if not self.ownership_map:
            return []
        
        differentials = []
        for player in starting_xi:
            pid = player['id']
            ownership = self.ownership_map.get(pid, 0.0)
            # Differential: < 30% league ownership
            if ownership < 0.30:
                differentials.append({
                    'name': player['name'],
                    'position': player['position'],
                    'ownership': round(ownership * 100, 1),
                    'ep_next': player.get('ep_next', 0)
                })
        
        # Sort by ownership ascending (best differentials first)
        differentials.sort(key=lambda x: x['ownership'])
        return differentials
    
    def _identify_template_picks(self, starting_xi: List[Dict]) -> List[Dict]:
        """Identify players in XI with high league ownership (template picks)."""
        if not self.ownership_map:
            return []
        
        template = []
        for player in starting_xi:
            pid = player['id']
            ownership = self.ownership_map.get(pid, 0.0)
            # Template: >= 50% league ownership
            if ownership >= 0.50:
                template.append({
                    'name': player['name'],
                    'position': player['position'],
                    'ownership': round(ownership * 100, 1),
                    'ep_next': player.get('ep_next', 0)
                })
        
        # Sort by ownership descending (most common first)
        template.sort(key=lambda x: x['ownership'], reverse=True)
        return template
    
    def _estimate_remaining_cost(self, current_counts: Dict, current_pos: str) -> float:
        """Estimate minimum cost to fill remaining squad slots."""
        # Get minimum prices per position
        min_prices = {}
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            pos_players = self.players_df[self.players_df['position'] == pos]
            available = pos_players[pos_players.apply(self._is_available, axis=1)]
            if not available.empty:
                min_prices[pos] = available['price'].min()
            else:
                min_prices[pos] = 4.0  # Fallback
        
        remaining_cost = 0.0
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            needed = self.POSITION_QUOTAS[pos] - current_counts.get(pos, 0)
            if pos == current_pos:
                needed -= 1  # Account for current selection
            remaining_cost += max(0, needed) * min_prices[pos]
        
        return remaining_cost
    
    def _format_player(self, player: Dict, is_starter: bool) -> Dict:
        """Format player dict for output."""
        player_id = int(player['id'])
        team_id = int(player.get('team', 0))
        league_own = self.ownership_map.get(player_id, 0.0) if self.ownership_map else 0.0
        
        # Get team name
        team_name = self.team_names.get(team_id, 'UNK')
        
        # Get fixtures for this team
        fixtures = self.team_fixtures.get(team_id, [])
        
        # Get 5-GW xP from predictions if available
        xp_5gw = 0.0
        if player_id in self.predictions:
            pred_data = self.predictions[player_id]
            xp_5gw = pred_data.get('cumulative', 0.0)
        
        return {
            'id': player_id,
            'name': player.get('web_name', 'Unknown'),
            'position': player.get('position', 'UNK'),
            'team': team_name,  # Now returns short name like "CHE", "MCI"
            'team_id': team_id,
            'price': round(float(player.get('price', 0)), 1),
            'score': round(float(player.get('score', 0)), 3),
            'ep_next': round(float(player.get('ep_next', 0) or 0), 2),
            'form': round(float(player.get('form', 0) or 0), 1),
            'total_points': int(player.get('total_points', 0) or 0),
            'ppg': round(float(player.get('points_per_game', 0) or 0), 2),
            'minutes': int(player.get('minutes', 0) or 0),
            'league_ownership': round(league_own * 100, 1),  # As percentage
            'is_starter': is_starter,
            'fixtures': fixtures,  # List of {gw, opponent, fdr, is_home}
            'xp_5gw': round(xp_5gw, 1)  # 5-gameweek expected points
        }
    
    def _select_starting_xi(self, squad: List[Dict]) -> Tuple[List[Dict], List[Dict], str]:
        """Select best starting XI from squad with valid formation.
        
        Valid formations: 3-4-3, 3-5-2, 4-3-3, 4-4-2, 4-5-1, 5-3-2, 5-4-1
        """
        # Group by position
        by_pos = {'GKP': [], 'DEF': [], 'MID': [], 'FWD': []}
        for p in squad:
            by_pos[p['position']].append(p)
        
        # Sort each position by score descending
        for pos in by_pos:
            by_pos[pos].sort(key=lambda x: x['score'], reverse=True)
        
        # Try formations in order of preference
        formations = [
            ('4-4-2', {'GKP': 1, 'DEF': 4, 'MID': 4, 'FWD': 2}),
            ('4-3-3', {'GKP': 1, 'DEF': 4, 'MID': 3, 'FWD': 3}),
            ('3-5-2', {'GKP': 1, 'DEF': 3, 'MID': 5, 'FWD': 2}),
            ('3-4-3', {'GKP': 1, 'DEF': 3, 'MID': 4, 'FWD': 3}),
            ('5-4-1', {'GKP': 1, 'DEF': 5, 'MID': 4, 'FWD': 1}),
            ('5-3-2', {'GKP': 1, 'DEF': 5, 'MID': 3, 'FWD': 2}),
            ('4-5-1', {'GKP': 1, 'DEF': 4, 'MID': 5, 'FWD': 1}),
        ]
        
        best_xi = None
        best_formation = None
        best_score = -1
        
        for form_name, form_counts in formations:
            # Check if we have enough players
            valid = True
            for pos, count in form_counts.items():
                if len(by_pos[pos]) < count:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Build XI for this formation
            xi = []
            for pos, count in form_counts.items():
                xi.extend(by_pos[pos][:count])
            
            # Calculate total score
            total_score = sum(p['score'] for p in xi)
            
            if total_score > best_score:
                best_score = total_score
                best_xi = xi
                best_formation = form_name
        
        # Fallback if no formation works
        if best_xi is None:
            best_xi = squad[:11]
            best_formation = '?-?-?'
        
        # Determine bench (players not in XI)
        xi_ids = {p['id'] for p in best_xi}
        bench = [p for p in squad if p['id'] not in xi_ids]
        
        # Sort bench: GKP first, then by score
        bench.sort(key=lambda x: (0 if x['position'] == 'GKP' else 1, -x['score']))
        
        return best_xi, bench, best_formation
    
    def _select_captains(self, starting_xi: List[Dict]) -> Tuple[Dict, Dict]:
        """Select captain and vice captain from starting XI.
        
        For Free Hit, captain selection considers both xPts and differential value.
        In aggressive mode, may pick differential captain if xPts is close.
        """
        # Sort by score descending (score already includes differential bonus)
        sorted_xi = sorted(starting_xi, key=lambda x: x['score'], reverse=True)
        
        # Captain is highest scorer, VC is second highest
        captain = sorted_xi[0] if sorted_xi else {}
        vice_captain = sorted_xi[1] if len(sorted_xi) > 1 else captain
        
        return captain, vice_captain





@dataclass
class MultiPeriodPlan:
    """Result from multi-period transfer planning.
    
    Represents a sequence of transfers across multiple gameweeks with
    squad continuity and free transfer banking.
    """
    status: str  # 'optimal', 'infeasible', 'unavailable', 'error'
    horizon: int = 5
    weekly_plans: List[Dict] = field(default_factory=list)  # One per GW
    total_expected_points: float = 0.0
    total_hit_cost: int = 0
    free_transfers_banked: List[int] = field(default_factory=list)  # FTs available each week
    message: str = ''
    solver_time: float = 0.0


def build_transfer_timeline(
    mip_result: MIPSolverResult,
    current_gw: int,
    horizon: int = 5
) -> Dict:
    """Build a transfer timeline from solver result for visualization.
    
    Converts the MIP solver output into a timeline format suitable for
    LaTeX rendering, showing transfers and expected points per gameweek.
    
    Args:
        mip_result: Result from TransferMIPSolver.
        current_gw: Current gameweek number.
        horizon: Number of gameweeks in the plan.
        
    Returns:
        Dict with timeline data for rendering.
    """
    if mip_result.status != 'optimal':
        return {
            'status': mip_result.status,
            'message': mip_result.message,
            'weeks': []
        }
    
    weeks = []
    per_gw_xp = mip_result.per_gw_xp or [0.0] * horizon
    
    # Week 1: Show transfers and first week lineup
    transfers_in = mip_result.transfers_in or []
    transfers_out = mip_result.transfers_out or []
    
    for w in range(horizon):
        gw = current_gw + w + 1
        
        week_data = {
            'gameweek': gw,
            'expected_points': per_gw_xp[w] if w < len(per_gw_xp) else 0.0,
            'transfers_in': transfers_in if w == 0 else [],
            'transfers_out': transfers_out if w == 0 else [],
            'hit_cost': mip_result.hit_cost if w == 0 else 0,
            'formation': mip_result.formation,
            'captain': mip_result.captain.get('name', '') if mip_result.captain else ''
        }
        weeks.append(week_data)
    
    return {
        'status': 'optimal',
        'current_gw': current_gw,
        'horizon': horizon,
        'weeks': weeks,
        'total_expected_points': mip_result.expected_points,
        'total_hit_cost': mip_result.hit_cost,
        'budget_remaining': mip_result.budget_remaining,
        'num_transfers': mip_result.num_transfers
    }


def format_timeline_for_latex(timeline: Dict) -> str:
    """Format transfer timeline as LaTeX TikZ diagram.
    
    Creates a horizontal timeline showing transfers and expected points
    across the planning horizon.
    
    Args:
        timeline: Dict from build_transfer_timeline().
        
    Returns:
        LaTeX content string.
    """
    if timeline.get('status') != 'optimal':
        return rf"\textit{{{timeline.get('message', 'No timeline available')}}}"
    
    weeks = timeline.get('weeks', [])
    if not weeks:
        return r"\textit{No timeline data available.}"
    
    lines = []
    lines.append(r"\begin{center}")
    lines.append(r"\begin{tikzpicture}[")
    lines.append(r"    week/.style={draw=fplpurple,rounded corners=5pt,minimum width=2.5cm,minimum height=2cm,fill=white},")
    lines.append(r"    transfer/.style={draw=fplgreen,thick,->},")
    lines.append(r"    hit/.style={draw=fplpink,thick,dashed}")
    lines.append(r"]")
    
    for i, week in enumerate(weeks[:5]):  # Max 5 weeks
        x_pos = i * 3.5
        gw = week.get('gameweek', '?')
        xp = week.get('expected_points', 0)
        captain = week.get('captain', '')
        transfers_in = week.get('transfers_in', [])
        hit = week.get('hit_cost', 0)
        
        # Week node
        lines.append(rf"\node[week] at ({x_pos},0) {{}};")
        lines.append(rf"\node[font=\bfseries\small,color=fplpurple] at ({x_pos},0.6) {{GW{gw}}};")
        lines.append(rf"\node[font=\Large\bfseries,color=fplgreen] at ({x_pos},0) {{{xp:.1f}}};")
        lines.append(rf"\node[font=\tiny,color=fplgray] at ({x_pos},-0.5) {{xP}};")
        
        # Show transfers for week 1
        if i == 0 and transfers_in:
            in_names = ', '.join([t.get('name', '?')[:8] for t in transfers_in[:2]])
            in_names = in_names.replace('_', r'\_')
            lines.append(rf"\node[font=\tiny,color=fplgreen,text width=2.5cm,align=center] at ({x_pos},-1.3) {{IN: {in_names}}};")
            if hit > 0:
                lines.append(rf"\node[font=\tiny,color=fplpink] at ({x_pos},-1.7) {{-{hit} hit}};")
        
        # Captain indicator
        if captain:
            cap_short = captain[:6].replace('_', r'\_')
            lines.append(rf"\node[font=\tiny,color=gold] at ({x_pos},0.85) {{C: {cap_short}}};")
        
        # Arrow to next week
        if i < len(weeks) - 1 and i < 4:
            lines.append(rf"\draw[transfer] ({x_pos + 1.3},0) -- ({x_pos + 2.2},0);")
    
    lines.append(r"\end{tikzpicture}")
    lines.append(r"\end{center}")
    
    return '\n'.join(lines)
