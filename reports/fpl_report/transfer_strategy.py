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

from solver.optimizer import TransferMIPSolver, MultiPeriodMIPSolver, MIP_AVAILABLE
from solver.definitions import MIPSolverResult, MultiPeriodResult, WeeklyPlan


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
        fixture_analysis = self._analyze_squad_fixtures(squad_analysis, num_weeks, current_gw)
        
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
            if mip_result.get('status') == 'optimal':
                # Store multi-period specific data
                result['scenarios'] = mip_result.get('scenarios', {})
                result['weekly_plans'] = mip_result.get('weekly_plans', [])
                result['transfer_sequence'] = mip_result.get('transfer_sequence', [])
                result['watchlist'] = mip_result.get('watchlist', [])
                result['confidence_per_gw'] = mip_result.get('confidence_per_gw', [])
                result['recommended_scenario'] = mip_result.get('recommended', 'balanced')

                if mip_result.get('transfers_in'):
                    try:
                        # Convert MIP plan to standard format for compatibility
                        result['planned_transfers'] = self._convert_mip_to_plan(
                            mip_result, current_gw
                        )

                        # UPDATE EV based on MIP
                        if mip_result.get('expected_points'):
                            baseline = mip_result.get('baseline_xp', current_squad_ev)
                            result['expected_value']['current_squad'] = baseline
                            result['expected_value']['optimized_squad'] = mip_result['expected_points']
                            result['expected_value']['potential_gain'] = round(
                                mip_result['expected_points'] - baseline, 1
                            )

                        # Remove heuristic recommendations - MIP takes over
                        result['immediate_recommendations'] = []
                        result['alternative_strategies'] = {}

                    except Exception as e:
                        print(f"Failed to convert MIP plan: {e}")
                        import traceback
                        traceback.print_exc()

                # If MIP is optimal but suggests NO transfers
                else:
                    result['planned_transfers'] = []
                    result['immediate_recommendations'] = []
                    result['alternative_strategies'] = {}
                    baseline = mip_result.get('baseline_xp', current_squad_ev)
                    result['expected_value']['current_squad'] = baseline
                    result['expected_value']['optimized_squad'] = baseline
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
        """Run the multi-period MIP solver and return results with three scenarios.

        Args:
            squad_analysis: Current squad analysis.
            num_weeks: Planning horizon.
            time_limit: Solver time limit per scenario.
            candidate_pool: Candidates per position.
            free_transfers: Available free transfers.

        Returns:
            Dict with solver results including three scenarios and weekly plans.
        """
        try:
            # Get current squad from fetcher
            if current_gw is None:
                current_gw = self.fetcher.get_current_gameweek()

            current_squad = self.fetcher.get_current_squad(current_gw)
            bank = self.fetcher.get_bank()

            # Build xP matrix for all candidates
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

            # Get rival squad IDs for differential analysis
            rival_ids = self._get_rival_squad_ids()

            # Create multi-period solver
            solver = MultiPeriodMIPSolver(
                current_squad=current_squad,
                bank=bank,
                players_df=self.fetcher.players_df,
                xp_matrix=xp_matrix,
                free_transfers=free_transfers,
                horizon=num_weeks,
                candidate_pool_size=candidate_pool,
                time_limit=time_limit,
                teams_data=teams_data,
                rival_squad_ids=rival_ids,
                current_gw=current_gw,
            )

            # Solve for all three scenarios
            multi_result = solver.solve(['conservative', 'balanced', 'aggressive'])

            # Convert MultiPeriodResult to dict for compatibility
            return self._format_multi_period_result(multi_result, current_gw)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'message': f'MIP solver error: {str(e)}',
                'scenarios': {},
                'recommended': 'balanced',
                'transfers_out': [],
                'transfers_in': [],
                'hit_cost': 0,
                'num_transfers': 0,
                'expected_points': 0,
                'weekly_plans': [],
            }

    def _get_rival_squad_ids(self) -> set:
        """Get player IDs owned by mini-league rivals for differential analysis."""
        rival_ids = set()
        try:
            competitors = getattr(self.fetcher, 'competitors', [])
            if competitors:
                for comp_id in competitors[:5]:  # Limit to top 5 rivals
                    try:
                        comp_squad = self.fetcher.get_team_picks(comp_id)
                        if comp_squad:
                            for pick in comp_squad.get('picks', []):
                                rival_ids.add(pick.get('element'))
                    except Exception:
                        pass
        except Exception:
            pass
        return rival_ids

    def _format_multi_period_result(self, multi_result: MultiPeriodResult, current_gw: int) -> Dict:
        """Format MultiPeriodResult for LaTeX/JSON compatibility."""
        # Get the recommended scenario's result
        recommended = multi_result.recommended or 'balanced'
        main_result = getattr(multi_result, recommended) or multi_result.balanced

        if not main_result:
            return {
                'status': 'error',
                'message': 'No valid scenario results',
                'scenarios': {},
                'recommended': recommended,
            }

        # Format scenarios summary (including weekly plans for each)
        scenarios_summary = {}
        for scenario_name in ['conservative', 'balanced', 'aggressive']:
            result = getattr(multi_result, scenario_name)
            if result and result.status == 'optimal':
                # Format weekly plans for this scenario
                scenario_weekly_plans = []
                for wp in result.weekly_plans:
                    plan_dict = {
                        'gameweek': wp.gameweek,
                        'is_hold': wp.is_hold,
                        'transfers_in': [{'name': p.get('name'), 'position': p.get('position'),
                                          'team': p.get('team'), 'xp': p.get('xp', [0]),
                                          'sell_price': p.get('sell_price', p.get('price', 0)),
                                          'buy_price': p.get('buy_price', p.get('price', 0))}
                                         for p in wp.transfers_in],
                        'transfers_out': [{'name': p.get('name'), 'position': p.get('position'),
                                           'team': p.get('team'), 'xp': p.get('xp', [0]),
                                           'sell_price': p.get('sell_price', p.get('price', 0)),
                                           'buy_price': p.get('buy_price', p.get('price', 0))}
                                          for p in wp.transfers_out],
                        'ft_available': wp.ft_available,
                        'ft_used': wp.ft_used,
                        'ft_remaining': wp.ft_remaining,
                        'hit_cost': wp.hit_cost,
                        'expected_xp': wp.expected_xp,
                        'confidence': wp.confidence,
                        'reasoning': wp.reasoning,
                        'captain': {'name': wp.captain.get('name'), 'xp': wp.captain.get('xp', 0)} if wp.captain else None,
                        'differential_captain': {'name': wp.differential_captain.get('name'),
                                                 'xp': wp.differential_captain.get('xp', 0),
                                                 'eo': wp.differential_captain.get('eo', 0)} if wp.differential_captain else None,
                        'formation': wp.formation,
                    }
                    scenario_weekly_plans.append(plan_dict)

                scenarios_summary[scenario_name] = {
                    'num_transfers': result.num_transfers,
                    'hit_cost': result.hit_cost,
                    'expected_points': result.expected_points,
                    'baseline_xp': result.baseline_xp,
                    'net_gain': round(result.expected_points - result.baseline_xp, 1),
                    'weekly_plans': scenario_weekly_plans,
                    'backup_transfers': result.backup_transfers,
                    'sell_rebuy_warnings': result.sell_rebuy_warnings,
                    'price_alerts': result.price_alerts,
                }

        # Format weekly plans for main result (for backward compatibility)
        weekly_plans_formatted = []
        for wp in main_result.weekly_plans:
            plan_dict = {
                'gameweek': wp.gameweek,
                'is_hold': wp.is_hold,
                'transfers_in': [{'name': p.get('name'), 'position': p.get('position'),
                                  'team': p.get('team'), 'xp': p.get('xp', [0]),
                                  'sell_price': p.get('sell_price', p.get('price', 0)),
                                  'buy_price': p.get('buy_price', p.get('price', 0))}
                                 for p in wp.transfers_in],
                'transfers_out': [{'name': p.get('name'), 'position': p.get('position'),
                                   'team': p.get('team'), 'xp': p.get('xp', [0]),
                                   'sell_price': p.get('sell_price', p.get('price', 0)),
                                   'buy_price': p.get('buy_price', p.get('price', 0))}
                                  for p in wp.transfers_out],
                'ft_available': wp.ft_available,
                'ft_used': wp.ft_used,
                'ft_remaining': wp.ft_remaining,
                'hit_cost': wp.hit_cost,
                'expected_xp': wp.expected_xp,
                'confidence': wp.confidence,
                'reasoning': wp.reasoning,
                'captain': {'name': wp.captain.get('name'), 'xp': wp.captain.get('xp', 0)} if wp.captain else None,
                'differential_captain': {'name': wp.differential_captain.get('name'),
                                         'xp': wp.differential_captain.get('xp', 0),
                                         'eo': wp.differential_captain.get('eo', 0)} if wp.differential_captain else None,
                'formation': wp.formation,
            }
            weekly_plans_formatted.append(plan_dict)

        # Format transfer sequence
        transfer_sequence = []
        for action in main_result.transfer_sequence:
            transfer_sequence.append({
                'gameweek': action.gameweek,
                'out': action.player_out.get('name'),
                'out_position': action.player_out.get('position'),
                'in': action.player_in.get('name'),
                'in_position': action.player_in.get('position'),
                'expected_gain': action.expected_gain,
                'cost': action.cost,
                'reasoning': action.reasoning,
                'is_sell_rebuy': action.is_sell_rebuy,
                'price_alert': action.price_alert,
            })

        return {
            'status': main_result.status,
            'message': main_result.message,
            'scenarios': scenarios_summary,
            'recommended': recommended,
            'transfers_out': main_result.transfers_out,
            'transfers_in': main_result.transfers_in,
            'starting_xi': main_result.starting_xi,
            'bench': main_result.bench,
            'formation': main_result.formation,
            'captain': main_result.captain,
            'hit_cost': main_result.hit_cost,
            'num_transfers': main_result.num_transfers,
            'expected_points': main_result.expected_points,
            'baseline_xp': main_result.baseline_xp,
            'per_gw_xp': main_result.per_gw_xp,
            'weekly_plans': weekly_plans_formatted,
            'transfer_sequence': transfer_sequence,
            'confidence_per_gw': main_result.confidence_per_gw,
            'watchlist': multi_result.watchlist,
            'solver_time': main_result.solver_time,
        }

    def _analyze_squad_fixtures(self, squad_analysis: List[Dict],
                                 num_weeks: int,
                                 current_gw: int = None) -> Dict[int, Dict]:
        """Analyze fixture difficulty for each squad player.

        Args:
            squad_analysis: List of player dicts from squad analysis.
            num_weeks: Number of future gameweeks to analyze.
            current_gw: Current gameweek number. Used to build GW-keyed
                fixture data for heatmap alignment.

        Returns:
            Dict mapping player_id to fixture analysis including
            ``fixtures_by_gw`` (keyed by GW number) alongside the flat
            ``fixtures`` list for backward compat.
        """
        fixture_analysis = {}

        # Pre-fetch all fixture difficulties once (avoids 15x Elo recalc)
        # Pass current_gw so the calculator excludes only GWs the report
        # considers past, not the API's live current GW.
        all_difficulties = {}
        if hasattr(self.fetcher, '_difficulty_calculator') and self.fetcher._difficulty_calculator:
            all_difficulties = self.fetcher._difficulty_calculator.get_fixture_difficulties(
                current_gw_override=current_gw
            )

        start_gw = (current_gw or 0) + 1
        end_gw = start_gw + num_weeks - 1

        for player in squad_analysis:
            pid = player.get('player_id')
            if not pid:
                continue

            player_name = player.get('name', 'Unknown')
            position = player.get('position', 'UNK')

            # Get team_id - try multiple locations
            team_id = player.get('stats', {}).get('team', 0)
            if not team_id:
                team_id = player.get('raw_stats', {}).get('team', 0)

            # Build GW-keyed fixture data from pre-fetched difficulties
            team_fixes = all_difficulties.get(team_id, [])
            fixtures_by_gw: Dict[int, List[Dict]] = {
                gw: [] for gw in range(start_gw, end_gw + 1)
            } if current_gw else {}

            flat_fixtures = []
            for fix in sorted(team_fixes, key=lambda x: x['gameweek']):
                gw = fix['gameweek']
                if gw < start_gw or gw > end_gw:
                    continue
                mapped = {
                    'gameweek': gw,
                    'opponent': fix['opponent'],
                    'is_home': fix['is_home'],
                    'difficulty': fix['fdr_elo'],
                    'difficulty_ordinal': fix['fdr_original'],
                    'win_prob': fix['win_prob'],
                    'draw_prob': fix['draw_prob'],
                    'loss_prob': fix['loss_prob'],
                    'opponent_elo': fix.get('opponent_elo', 0),
                    'own_elo': fix.get('own_elo', 0),
                }
                flat_fixtures.append(mapped)
                if current_gw and gw in fixtures_by_gw:
                    fixtures_by_gw[gw].append(mapped)

            if not flat_fixtures:
                fixture_analysis[pid] = {
                    'player_name': player_name,
                    'position': position,
                    'fixtures': [],
                    'fixtures_by_gw': fixtures_by_gw,
                    'avg_difficulty': 3.0,
                    'swing': 'neutral',
                    'swing_gw': None,
                    'difficulty_trend': [3.0] * num_weeks
                }
                continue

            # Per-GW difficulties for swing calc (use average for DGW, skip BGW)
            difficulties = []
            for gw in range(start_gw, end_gw + 1):
                gw_fixes = fixtures_by_gw.get(gw, []) if current_gw else []
                if not gw_fixes:
                    # For swing calc, use flat list order as fallback
                    continue
                elif len(gw_fixes) == 1:
                    difficulties.append(gw_fixes[0].get('difficulty', 3))
                else:
                    difficulties.append(
                        sum(f.get('difficulty', 3) for f in gw_fixes) / len(gw_fixes)
                    )

            # Fallback to flat list if GW-keyed calc produced nothing
            if not difficulties:
                difficulties = [f.get('difficulty', 3) for f in flat_fixtures]

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
                'fixtures': flat_fixtures,
                'fixtures_by_gw': fixtures_by_gw,
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
        
        # Match transfers by position - FPL requires transfers within same position
        # Group by position first, then match within each group
        from collections import defaultdict

        out_by_pos = defaultdict(list)
        in_by_pos = defaultdict(list)

        for p in transfers_out_list:
            pos = p.get('position', 'UNK')
            out_by_pos[pos].append(p)

        for p in transfers_in_list:
            pos = p.get('position', 'UNK')
            in_by_pos[pos].append(p)

        # Match within each position
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            outs = out_by_pos.get(pos, [])
            ins = in_by_pos.get(pos, [])

            for p_out, p_in in zip(outs, ins):
                name_out = p_out.get('web_name', p_out.get('name', 'Unknown'))
                id_out = p_out.get('id')

                name_in = p_in.get('web_name', p_in.get('name', 'Unknown'))
                id_in = p_in.get('id')

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
                    'out_position': pos,
                    'in': name_in,
                    'in_position': pos,
                    'expected_gain': max(0.1, gain),
                    'reasoning': reason,
                    'priority': 'high',
                    'take_hit': False  # Solver handled cost
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

    def analyze_chip_opportunities(
        self,
        squad_analysis: List[Dict],
        chips_used: List[Dict],
        gw_history: List[Dict],
        ml_position: Dict = None
    ) -> Dict:
        """Analyze chip usage opportunities for the current squad state.

        Generates personalized chip recommendations based on:
        - Squad issues (injuries, suspensions, price drops)
        - Fixture difficulty and BGW/DGW calendar
        - Mini-league position (chasing vs protecting)
        - Opportunity cost vs optimal free transfers

        Args:
            squad_analysis: Player analysis data for current squad
            chips_used: List of chips already used this season
            gw_history: Gameweek history for the team
            ml_position: Mini-league position data (optional)

        Returns:
            Dict with chip analysis for each available chip
        """
        from simulation.state import CHIP_RESET_GW, CHIPS_PER_HALF
        from .data_fetcher import get_bgw_dgw_gameweeks
        from .dgw_bgw_fetcher import fetch_dgw_bgw_intelligence, merge_bgw_dgw_data

        current_gw = len(gw_history) if gw_history else 1
        first_half = current_gw < CHIP_RESET_GW
        half_label = 'first' if first_half else 'second'

        # Filter chips used in current half
        if first_half:
            chips_used_this_half = [c for c in chips_used if c.get('event', 0) < CHIP_RESET_GW]
        else:
            chips_used_this_half = [c for c in chips_used if c.get('event', 0) >= CHIP_RESET_GW]

        chips_remaining = CHIPS_PER_HALF - len(chips_used_this_half)

        # Build chip usage map from current-half chips only (avoids cross-half confusion)
        used_chips = {c.get('name', ''): c.get('event', 0) for c in chips_used_this_half}

        # Get squad issues
        squad_issues = self.fetcher.get_squad_issues()

        # Get BGW/DGW data
        try:
            bgw_dgw_data = get_bgw_dgw_gameweeks(use_cache=True, session_cache=self.fetcher.cache)
            predicted_data = fetch_dgw_bgw_intelligence(current_gw=current_gw, session_cache=self.fetcher.cache)
            merged_data = merge_bgw_dgw_data(bgw_dgw_data, predicted_data)
            
            bgws = [b for b in merged_data.get('bgw', []) if b.get('gw', 0) > current_gw]
            dgws = [d for d in merged_data.get('dgw', []) if d.get('gw', 0) > current_gw]
        except Exception:
            bgws, dgws = [], []

        # Analyze ML position context
        ml_context = self._analyze_ml_context(ml_position) if ml_position else {
            'strategy': 'balanced',
            'points_gap': 0,
            'recommendation': 'Play for optimal expected value'
        }

        # Build analysis for each chip
        analysis = {
            'current_gw': current_gw,
            'half': half_label,
            'chips_remaining': chips_remaining,
            'chips_remaining_display': f"{chips_remaining}/{CHIPS_PER_HALF}",
            'squad_issues': squad_issues,
            'ml_context': ml_context,
            'bgws': bgws[:3],
            'dgws': dgws[:3],
            'deadline_warning': None,
            'chips': {}
        }

        # Deadline warning for first half
        if first_half and current_gw >= 15 and chips_remaining > 0:
            gws_left = CHIP_RESET_GW - current_gw
            analysis['deadline_warning'] = {
                'chips_expiring': chips_remaining,
                'gws_remaining': gws_left,
                'message': f"{chips_remaining} unused chip(s) will expire at GW{CHIP_RESET_GW}. {gws_left} GWs remaining to use them."
            }

        # Analyze Wildcard
        wc_analysis = self._analyze_wildcard(
            squad_analysis, squad_issues, used_chips, current_gw, first_half, ml_context
        )
        analysis['chips']['wildcard'] = wc_analysis

        # Analyze Free Hit
        fh_analysis = self._analyze_free_hit(
            squad_analysis, squad_issues, used_chips, current_gw, first_half, bgws
        )
        analysis['chips']['freehit'] = fh_analysis

        # Analyze Bench Boost
        bb_analysis = self._analyze_bench_boost(
            squad_analysis, used_chips, current_gw, dgws
        )
        analysis['chips']['bboost'] = bb_analysis

        # Analyze Triple Captain
        tc_analysis = self._analyze_triple_captain(
            squad_analysis, used_chips, current_gw, dgws
        )
        analysis['chips']['3xc'] = tc_analysis

        # Identify chip synergies
        analysis['synergies'] = self._identify_chip_synergies(
            analysis['chips'], current_gw, bgws, dgws, first_half
        )

        # Build trigger list if no urgent chip needed
        analysis['triggers'] = self._build_chip_triggers(analysis['chips'], squad_issues)

        return analysis

    def _analyze_ml_context(self, ml_position: Dict) -> Dict:
        """Analyze mini-league position to inform chip strategy."""
        if not ml_position:
            return {'strategy': 'balanced', 'points_gap': 0, 'recommendation': 'Play for optimal expected value'}

        user_points = ml_position.get('user_points', 0)
        leader_points = ml_position.get('leader_points', 0)
        user_rank = ml_position.get('user_rank', 1)
        total_managers = ml_position.get('total_managers', 1)

        points_gap = leader_points - user_points

        if user_rank == 1:
            # Leading
            if points_gap == 0:
                return {
                    'strategy': 'protect',
                    'points_gap': 0,
                    'recommendation': 'Protect lead - prefer template/safe picks, avoid unnecessary risks'
                }
        elif points_gap > 100:
            # Far behind
            return {
                'strategy': 'aggressive',
                'points_gap': points_gap,
                'recommendation': f'{points_gap} pts behind leader - consider differential picks and aggressive chip timing'
            }
        elif points_gap > 50:
            # Moderately behind
            return {
                'strategy': 'chase',
                'points_gap': points_gap,
                'recommendation': f'{points_gap} pts behind - balance risk with reward, consider chip timing edges'
            }
        else:
            # Close race
            return {
                'strategy': 'balanced',
                'points_gap': points_gap,
                'recommendation': f'Close to leader ({points_gap} pts) - play optimal strategy, avoid costly mistakes'
            }

        return {'strategy': 'balanced', 'points_gap': points_gap, 'recommendation': 'Play for optimal expected value'}

    def _analyze_wildcard(
        self, squad_analysis, squad_issues, used_chips, current_gw, first_half, ml_context
    ) -> Dict:
        """Analyze Wildcard chip opportunity."""
        chip_name = 'wildcard'

        # Check if already used this half
        if chip_name in used_chips:
            used_gw = used_chips[chip_name]
            from simulation.state import CHIP_RESET_GW
            used_this_half = (first_half and used_gw < CHIP_RESET_GW) or (not first_half and used_gw >= CHIP_RESET_GW)
            if used_this_half:
                return {
                    'status': 'used',
                    'used_gw': used_gw,
                    'available': False,
                    'recommendation': None,
                    'issues_addressed': [],
                    'opportunity_cost': None
                }

        # Count issues WC would address
        issues_addressed = []
        if squad_issues['injuries']:
            issues_addressed.extend([f"{p['name']} injured" for p in squad_issues['injuries']])
        if squad_issues['suspension_risk']:
            issues_addressed.extend([f"{p['name']} {p['yellows']} yellows" for p in squad_issues['suspension_risk'] if p['risk_level'] == 'high'])
        if squad_issues['price_drops']:
            issues_addressed.extend([f"{p['name']} price drop" for p in squad_issues['price_drops'][:2]])

        # Calculate opportunity cost (simplified - full version in Phase 2)
        total_issues = len(issues_addressed)
        estimated_points_loss = total_issues * 3  # Rough estimate: 3 pts per issue per week

        # Build recommendation
        urgency = 'low'
        recommendation = None

        if total_issues >= 4:
            urgency = 'high'
            recommendation = f"Strong WC candidate - {total_issues} issues to address"
        elif total_issues >= 2:
            urgency = 'medium'
            recommendation = f"Consider WC - {total_issues} issues affecting squad"
        else:
            # Generic timing advice
            if first_half:
                if current_gw < 7:
                    recommendation = "Save for GW7-9 fixture swing period"
                elif current_gw >= 15:
                    recommendation = f"Use before GW20 - doesn't carry over! ({20 - current_gw} GWs left)"
                    urgency = 'medium'
            else:
                if current_gw < 30:
                    recommendation = "Target GW30-32 for run-in preparation"
                else:
                    recommendation = "Use soon for season run-in optimization"

        # Adjust for ML context
        if ml_context['strategy'] == 'aggressive' and total_issues >= 2:
            recommendation += " - consider differential targets"
        elif ml_context['strategy'] == 'protect':
            recommendation += " - prioritize template players"

        return {
            'status': 'available',
            'available': True,
            'urgency': urgency,
            'recommendation': recommendation,
            'issues_addressed': issues_addressed[:5],  # Limit to 5
            'opportunity_cost': {
                'estimated_weekly_loss': estimated_points_loss,
                'description': f"~{estimated_points_loss} pts/week from unaddressed issues" if estimated_points_loss > 0 else None
            }
        }

    def _analyze_free_hit(
        self, squad_analysis, squad_issues, used_chips, current_gw, first_half, bgws
    ) -> Dict:
        """Analyze Free Hit chip opportunity."""
        chip_name = 'freehit'

        if chip_name in used_chips:
            used_gw = used_chips[chip_name]
            from simulation.state import CHIP_RESET_GW
            used_this_half = (first_half and used_gw < CHIP_RESET_GW) or (not first_half and used_gw >= CHIP_RESET_GW)
            if used_this_half:
                return {
                    'status': 'used',
                    'used_gw': used_gw,
                    'available': False,
                    'recommendation': None,
                    'target_gw': None
                }

        # Find best FH target
        target_gw = None
        recommendation = None
        urgency = 'low'

        if bgws:
            next_bgw = bgws[0]
            bgw_gw = next_bgw.get('gw', 0)
            teams_missing = next_bgw.get('teams_missing', 0)

            # Count how many of our players are affected
            # (simplified - would need fixture data in full implementation)
            if teams_missing >= 6:
                target_gw = bgw_gw
                urgency = 'high' if (bgw_gw - current_gw) <= 3 else 'medium'
                recommendation = f"Target BGW{bgw_gw} - {teams_missing} teams blanking"

        if not recommendation:
            if first_half:
                recommendation = "Save for next confirmed BGW"
            else:
                recommendation = "Use on BGW or DGW with significant team gaps"

        return {
            'status': 'available',
            'available': True,
            'urgency': urgency,
            'recommendation': recommendation,
            'target_gw': target_gw,
            'bgw_targets': [{'gw': b.get('gw'), 'teams_missing': b.get('teams_missing', 0)} for b in bgws[:2]]
        }

    def _analyze_bench_boost(self, squad_analysis, used_chips, current_gw, dgws) -> Dict:
        """Analyze Bench Boost chip opportunity."""
        chip_name = 'bboost'
        first_half = current_gw < 20

        if chip_name in used_chips:
            used_gw = used_chips[chip_name]
            from simulation.state import CHIP_RESET_GW
            used_this_half = (first_half and used_gw < CHIP_RESET_GW) or (not first_half and used_gw >= CHIP_RESET_GW)
            if used_this_half:
                return {
                    'status': 'used',
                    'used_gw': used_gw,
                    'available': False,
                    'recommendation': None
                }

        # Analyze bench quality
        bench_players = [p for p in squad_analysis if p.get('position_in_squad', 0) > 11]

        recommendation = None
        urgency = 'low'
        target_gw = None

        if dgws:
            next_dgw = dgws[0]
            dgw_gw = next_dgw.get('gw', 0)
            teams_doubled = next_dgw.get('teams_doubled', 0)

            if teams_doubled >= 8:
                target_gw = dgw_gw
                urgency = 'medium' if (dgw_gw - current_gw) <= 5 else 'low'
                recommendation = f"Target DGW{dgw_gw} - {teams_doubled} teams doubled. Build bench value beforehand."

        if not recommendation:
            recommendation = "Target DGW34-37 historically. Build bench 2-3 weeks before."

        return {
            'status': 'available',
            'available': True,
            'urgency': urgency,
            'recommendation': recommendation,
            'target_gw': target_gw,
            'bench_players': [{'name': p.get('name'), 'position': p.get('position')} for p in bench_players],
            'dgw_targets': [{'gw': d.get('gw'), 'teams_doubled': d.get('teams_doubled', 0)} for d in dgws[:2]]
        }

    def _analyze_triple_captain(self, squad_analysis, used_chips, current_gw, dgws) -> Dict:
        """Analyze Triple Captain chip opportunity."""
        chip_name = '3xc'
        first_half = current_gw < 20

        if chip_name in used_chips:
            used_gw = used_chips[chip_name]
            from simulation.state import CHIP_RESET_GW
            used_this_half = (first_half and used_gw < CHIP_RESET_GW) or (not first_half and used_gw >= CHIP_RESET_GW)
            if used_this_half:
                return {
                    'status': 'used',
                    'used_gw': used_gw,
                    'available': False,
                    'recommendation': None
                }

        # Find premium captain options (price > 10m)
        premiums = []
        for p in squad_analysis:
            stats = p.get('stats', {})
            price = stats.get('now_cost', 0) / 10
            if price >= 10.0:
                premiums.append({
                    'name': p.get('name'),
                    'price': price,
                    'form': float(stats.get('form', 0) or 0),
                    'position': p.get('position')
                })

        recommendation = None
        urgency = 'low'
        target_gw = None

        if dgws:
            next_dgw = dgws[0]
            dgw_gw = next_dgw.get('gw', 0)

            if premiums:
                top_premium = max(premiums, key=lambda x: x['form'])
                target_gw = dgw_gw
                urgency = 'medium' if (dgw_gw - current_gw) <= 3 else 'low'
                recommendation = f"Target DGW{dgw_gw} with {top_premium['name']} if form holds"
            else:
                recommendation = f"Target DGW{dgw_gw} - acquire a premium (Haaland/Salah) first"

        if not recommendation:
            recommendation = "Wait for DGW with premium attacker in strong form"

        return {
            'status': 'available',
            'available': True,
            'urgency': urgency,
            'recommendation': recommendation,
            'target_gw': target_gw,
            'premium_options': premiums[:3],
            'dgw_targets': [{'gw': d.get('gw'), 'teams_doubled': d.get('teams_doubled', 0)} for d in dgws[:2]]
        }

    def _identify_chip_synergies(self, chips, current_gw, bgws, dgws, first_half) -> List[Dict]:
        """Identify beneficial chip combinations."""
        synergies = []

        # WC before BB synergy (build BB bench with WC)
        if chips.get('wildcard', {}).get('available') and chips.get('bboost', {}).get('available'):
            if dgws:
                dgw_gw = dgws[0].get('gw', 0)
                if dgw_gw > current_gw + 2:  # At least 2 weeks before DGW
                    synergies.append({
                        'chips': ['wildcard', 'bboost'],
                        'strategy': f"WC in GW{dgw_gw - 2} to build BB-ready bench, then BB in DGW{dgw_gw}",
                        'value': 'high'
                    })

        # FH saves squad for following week
        if chips.get('freehit', {}).get('available') and chips.get('wildcard', {}).get('available'):
            if bgws:
                bgw_gw = bgws[0].get('gw', 0)
                synergies.append({
                    'chips': ['freehit', 'wildcard'],
                    'strategy': f"FH on BGW{bgw_gw} preserves squad for WC planning",
                    'value': 'medium'
                })

        # First half deadline synergy
        if first_half and current_gw >= 15:
            from simulation.state import CHIP_RESET_GW
            remaining = CHIP_RESET_GW - current_gw
            if chips.get('wildcard', {}).get('available') and chips.get('bboost', {}).get('available'):
                synergies.append({
                    'chips': ['wildcard', 'bboost'],
                    'strategy': f"Use WC then BB before GW{CHIP_RESET_GW} - {remaining} GWs to use both!",
                    'value': 'critical'
                })

        return synergies

    def _build_chip_triggers(self, chips, squad_issues) -> List[str]:
        """Build list of triggers that would change chip recommendations."""
        triggers = []

        # If no high urgency chips, show what would trigger them
        high_urgency = any(c.get('urgency') == 'high' for c in chips.values() if isinstance(c, dict))

        if not high_urgency:
            if squad_issues['total_issues'] < 4:
                triggers.append(f"2+ more injuries would trigger Wildcard consideration (currently {len(squad_issues['injuries'])})")
            triggers.append("DGW announcement would trigger BB/TC planning")
            triggers.append("BGW announcement would trigger FH consideration")
            if squad_issues['price_drops']:
                triggers.append(f"Watch price drops on: {', '.join([p['name'] for p in squad_issues['price_drops'][:3]])}")

        return triggers

    # =========================================================================
    # PHASE 2: Optimizer-based chip projections
    # =========================================================================

    def calculate_bb_projections(
        self, squad_analysis: List[Dict], dgws: List[Dict], num_gws: int = 5
    ) -> Dict:
        """Calculate projected bench points for upcoming DGWs.

        Args:
            squad_analysis: Current squad analysis data
            dgws: List of upcoming DGW dicts with 'gw' and 'teams_doubled'
            num_gws: Number of gameweeks to project

        Returns:
            Dict with BB projections per DGW
        """
        if not dgws:
            return {'projections': [], 'best_dgw': None, 'recommendation': 'No DGWs detected yet'}

        # Get bench players (positions 12-15)
        bench_players = [p for p in squad_analysis if p.get('position_in_squad', 0) > 11]

        if not bench_players:
            return {'projections': [], 'best_dgw': None, 'recommendation': 'No bench data available'}

        # Get predictions for bench players
        bench_ids = [p.get('player_id') for p in bench_players if p.get('player_id')]

        try:
            # Ensure predictor is trained
            if not self.predictor.is_trained:
                self._ensure_predictor_trained()

            predictions = self._predict_multiple_gws(bench_ids, num_gws)
        except Exception as e:
            return {'projections': [], 'best_dgw': None, 'recommendation': f'Prediction failed: {str(e)[:50]}'}

        current_gw = self.fetcher.get_current_gameweek()
        projections = []

        for dgw in dgws[:3]:  # Limit to next 3 DGWs
            dgw_gw = dgw.get('gw', 0)
            if dgw_gw <= current_gw:
                continue

            gw_offset = dgw_gw - current_gw - 1  # 0-indexed offset

            # Calculate projected bench points for this DGW
            bench_points = []
            for player in bench_players:
                pid = player.get('player_id')
                name = player.get('name', 'Unknown')
                position = player.get('position', 'UNK')

                player_pred = predictions.get(pid, {})
                gw_predictions = player_pred.get('gw_predictions', [])

                # Get projection for the DGW (doubled because DGW)
                if gw_offset < len(gw_predictions):
                    base_pts = gw_predictions[gw_offset]
                    # DGW doubles the points expectation (roughly)
                    dgw_pts = base_pts * 1.8  # Slightly less than 2x due to rotation risk
                else:
                    dgw_pts = 4.0  # Default estimate

                bench_points.append({
                    'name': name,
                    'position': position,
                    'projected_pts': round(dgw_pts, 1)
                })

            total_bench_pts = sum(p['projected_pts'] for p in bench_points)

            projections.append({
                'gw': dgw_gw,
                'teams_doubled': dgw.get('teams_doubled', 0),
                'bench_players': bench_points,
                'total_projected': round(total_bench_pts, 1),
                'recommendation': self._bb_recommendation(total_bench_pts, dgw.get('teams_doubled', 0))
            })

        # Find best DGW
        best = max(projections, key=lambda x: x['total_projected']) if projections else None

        return {
            'projections': projections,
            'best_dgw': best,
            'recommendation': f"Best BB opportunity: DGW{best['gw']} (~{best['total_projected']} pts)" if best else 'No suitable DGW found'
        }

    def _bb_recommendation(self, total_pts: float, teams_doubled: int) -> str:
        """Generate BB recommendation based on projected points."""
        if total_pts >= 20 and teams_doubled >= 10:
            return "Excellent BB opportunity"
        elif total_pts >= 15 and teams_doubled >= 8:
            return "Good BB opportunity"
        elif total_pts >= 10:
            return "Decent BB opportunity - consider building bench first"
        else:
            return "Weak BB opportunity - build bench value first"

    def rank_tc_options(
        self, squad_analysis: List[Dict], dgws: List[Dict], num_gws: int = 5
    ) -> Dict:
        """Rank premium players by projected DGW points for TC.

        Args:
            squad_analysis: Current squad analysis data
            dgws: List of upcoming DGW dicts
            num_gws: Number of gameweeks to project

        Returns:
            Dict with TC rankings per DGW
        """
        if not dgws:
            return {'rankings': [], 'best_option': None, 'recommendation': 'No DGWs detected yet'}

        # Get premium players (price >= 10m) and regular captain candidates
        premiums = []
        for p in squad_analysis:
            stats = p.get('stats', {})
            price = stats.get('now_cost', 0) / 10
            form = float(stats.get('form', 0) or 0)
            position = p.get('position', 'UNK')

            # Include premiums and high-form attackers/midfielders
            if price >= 10.0 or (position in ['MID', 'FWD'] and form >= 5.0):
                premiums.append({
                    'player_id': p.get('player_id'),
                    'name': p.get('name'),
                    'position': position,
                    'price': price,
                    'form': form
                })

        if not premiums:
            return {'rankings': [], 'best_option': None, 'recommendation': 'No premium players in squad'}

        premium_ids = [p['player_id'] for p in premiums if p.get('player_id')]

        try:
            if not self.predictor.is_trained:
                self._ensure_predictor_trained()

            predictions = self._predict_multiple_gws(premium_ids, num_gws)
        except Exception as e:
            return {'rankings': [], 'best_option': None, 'recommendation': f'Prediction failed: {str(e)[:50]}'}

        current_gw = self.fetcher.get_current_gameweek()
        rankings = []

        for dgw in dgws[:3]:
            dgw_gw = dgw.get('gw', 0)
            if dgw_gw <= current_gw:
                continue

            gw_offset = dgw_gw - current_gw - 1

            # Calculate projected TC points for each premium
            tc_options = []
            for player in premiums:
                pid = player.get('player_id')
                player_pred = predictions.get(pid, {})
                gw_predictions = player_pred.get('gw_predictions', [])

                if gw_offset < len(gw_predictions):
                    base_pts = gw_predictions[gw_offset]
                    # DGW + TC = 3x base (roughly, with DGW bonus)
                    dgw_pts = base_pts * 1.8  # DGW expectation
                    tc_pts = dgw_pts * 3  # TC multiplier
                else:
                    tc_pts = 15.0  # Default

                tc_options.append({
                    'name': player['name'],
                    'position': player['position'],
                    'price': player['price'],
                    'form': player['form'],
                    'projected_pts': round(base_pts, 1),
                    'tc_projected': round(tc_pts, 1)
                })

            # Sort by TC projected points
            tc_options.sort(key=lambda x: x['tc_projected'], reverse=True)

            rankings.append({
                'gw': dgw_gw,
                'teams_doubled': dgw.get('teams_doubled', 0),
                'options': tc_options[:5],  # Top 5
                'best': tc_options[0] if tc_options else None
            })

        # Find best overall TC option
        best_option = None
        best_pts = 0
        for r in rankings:
            if r.get('best') and r['best']['tc_projected'] > best_pts:
                best_pts = r['best']['tc_projected']
                best_option = {**r['best'], 'gw': r['gw']}

        return {
            'rankings': rankings,
            'best_option': best_option,
            'recommendation': f"Best TC: {best_option['name']} in DGW{best_option['gw']} (~{best_option['tc_projected']} pts)" if best_option else 'No suitable TC option'
        }

    def generate_fh_squad_suggestion(
        self, bgws: List[Dict], budget: float = None
    ) -> Dict:
        """Generate suggested Free Hit squad for upcoming BGW.

        Args:
            bgws: List of upcoming BGW dicts
            budget: Team value budget (defaults to current team value)

        Returns:
            Dict with FH squad suggestion
        """
        if not bgws:
            return {'squad': None, 'recommendation': 'No BGWs detected yet'}

        target_bgw = bgws[0]
        bgw_gw = target_bgw.get('gw', 0)

        if budget is None:
            budget = self.fetcher.get_team_value() + self.fetcher.get_bank()

        try:
            # Use existing FreeHitOptimizer
            fh_optimizer = FreeHitOptimizer(
                players_df=self.fetcher.players_df,
                total_budget=budget,
                predictions=None,  # Will use form-based scoring
                teams_data=self.fetcher.bootstrap_data.get('teams', []),
                fixtures_data=self.fetcher.bootstrap_data.get('fixtures', []),
                current_gw=self.fetcher.get_current_gameweek(),
                target_gw=bgw_gw
            )

            result = fh_optimizer.build_squad()

            if result and result.get('squad'):
                # Extract budget info (FreeHitOptimizer uses different key names)
                budget_info = result.get('budget', {})
                total_cost = budget_info.get('spent', 0)

                # Calculate projected points from starting XI
                starting_xi = result.get('starting_xi', [])
                projected_pts = sum(p.get('xp_5gw', p.get('score', 0)) for p in starting_xi)

                return {
                    'target_gw': bgw_gw,
                    'teams_missing': target_bgw.get('teams_missing', 0),
                    'squad': result['squad'],
                    'starting_xi': starting_xi,
                    'bench': result.get('bench', []),
                    'formation': result.get('formation', ''),
                    'total_cost': total_cost,
                    'projected_pts': round(projected_pts, 1),
                    'recommendation': f"FH squad for BGW{bgw_gw}: {result.get('formation', '')} formation, ~{projected_pts:.1f} pts expected"
                }
            else:
                return {'squad': None, 'recommendation': 'FH squad optimization failed'}

        except Exception as e:
            return {'squad': None, 'recommendation': f'FH optimization error: {str(e)[:50]}'}

    def generate_wc_squad_suggestion(self, budget: float = None) -> Dict:
        """Generate suggested Wildcard squad.

        Args:
            budget: Team value budget (defaults to current team value + bank)

        Returns:
            Dict with WC squad suggestion
        """
        if budget is None:
            budget = self.fetcher.get_team_value() + self.fetcher.get_bank()

        try:
            # Get predictions for all top players
            top_players = self.fetcher.players_df.nlargest(200, 'total_points')['id'].tolist()

            if self.predictor.is_trained:
                predictions = self._predict_multiple_gws(top_players, num_gws=5)
            else:
                predictions = {}

            # Use existing WildcardOptimizer
            wc_optimizer = WildcardOptimizer(
                players_df=self.fetcher.players_df,
                total_budget=budget,
                predictions=predictions,
                teams_data=self.fetcher.bootstrap_data.get('teams', []),
                fixtures_data=self.fetcher.bootstrap_data.get('fixtures', []),
                current_gw=self.fetcher.get_current_gameweek()
            )

            result = wc_optimizer.build_squad()

            if result and result.get('squad'):
                # Extract budget info (WildcardOptimizer uses different key names)
                budget_info = result.get('budget', {})
                total_cost = budget_info.get('spent', 0)
                budget_remaining = budget_info.get('remaining', 0)

                # Calculate projected points from starting XI
                starting_xi = result.get('starting_xi', [])
                projected_pts = sum(p.get('xp_5gw', 0) for p in starting_xi)

                return {
                    'squad': result['squad'],
                    'starting_xi': starting_xi,
                    'bench': result.get('bench', []),
                    'formation': result.get('formation', ''),
                    'total_cost': total_cost,
                    'projected_pts': round(projected_pts, 1),
                    'budget_remaining': round(budget_remaining, 1),
                    'recommendation': f"WC target: {result.get('formation', '')} formation, {total_cost:.1f}m used, ~{projected_pts:.1f} 5-GW pts"
                }
            else:
                return {'squad': None, 'recommendation': 'WC squad optimization failed'}

        except Exception as e:
            return {'squad': None, 'recommendation': f'WC optimization error: {str(e)[:50]}'}

    def get_phase2_chip_analysis(
        self, squad_analysis: List[Dict], chips_used: List[Dict], gw_history: List[Dict]
    ) -> Dict:
        """Get comprehensive Phase 2 chip analysis with optimizer projections.

        This is the main entry point for Phase 2 chip analysis.

        Args:
            squad_analysis: Current squad analysis
            chips_used: List of chips already used
            gw_history: Gameweek history

        Returns:
            Dict with all Phase 2 projections
        """
        from .data_fetcher import get_bgw_dgw_gameweeks

        current_gw = len(gw_history) if gw_history else self.fetcher.get_current_gameweek()

        # Get BGW/DGW data
        try:
            bgw_dgw_data = get_bgw_dgw_gameweeks(use_cache=True, session_cache=self.fetcher.cache)
            bgws = [b for b in bgw_dgw_data.get('bgw', []) if b.get('gw', 0) > current_gw]
            dgws = [d for d in bgw_dgw_data.get('dgw', []) if d.get('gw', 0) > current_gw]
        except Exception:
            bgws, dgws = [], []

        # Build used chips map for current half
        from simulation.state import CHIP_RESET_GW
        first_half = current_gw < CHIP_RESET_GW
        if first_half:
            chips_used_this_half = {c.get('name', ''): c.get('event', 0) for c in chips_used if c.get('event', 0) < CHIP_RESET_GW}
        else:
            chips_used_this_half = {c.get('name', ''): c.get('event', 0) for c in chips_used if c.get('event', 0) >= CHIP_RESET_GW}

        result = {
            'bb_projections': None,
            'tc_rankings': None,
            'fh_squad': None,
            'wc_squad': None
        }

        # BB projections (only if BB available and DGWs exist)
        if 'bboost' not in chips_used_this_half and dgws:
            result['bb_projections'] = self.calculate_bb_projections(squad_analysis, dgws)

        # TC rankings (only if TC available and DGWs exist)
        if '3xc' not in chips_used_this_half and dgws:
            result['tc_rankings'] = self.rank_tc_options(squad_analysis, dgws)

        # FH squad (only if FH available and BGWs exist)
        if 'freehit' not in chips_used_this_half and bgws:
            # Only generate if BGW is within 5 GWs
            if bgws[0].get('gw', 0) - current_gw <= 5:
                result['fh_squad'] = self.generate_fh_squad_suggestion(bgws)

        # WC squad (only if WC available)
        if 'wildcard' not in chips_used_this_half:
            result['wc_squad'] = self.generate_wc_squad_suggestion()

        return result


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
            'safe': {
                'diff_bonus': 0.3, 'max_differentials': 1,
                'ownership_floor': 10.0,  # Only consider players owned by 10%+
                'captain_from_template': True,  # Captain must be high-ownership
            },
            'balanced': {
                'diff_bonus': 1.5, 'max_differentials': 4,
                'ownership_floor': 2.0,
                'captain_from_template': False,
            },
            'aggressive': {
                'diff_bonus': 4.0, 'max_differentials': 8,
                'ownership_floor': 0.5,  # Include very low-ownership punts
                'captain_from_template': False,
            },
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
        6. Momentum bonus - rewards players with high form/ppg/total_points
        
        Note: Budget maximization is handled as a tie-break in squad selection,
        not in the score itself (to avoid sacrificing expected points for spend).
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
        # 4. Momentum bonus - rewards proven in-form performers
        # -----------------------------------------------------------------
        # Momentum is real in FPL - players on hot streaks tend to continue
        # Use form (recent avg), ppg (season avg), and total_points (volume)
        form = float(row.get('form', 0) or 0)
        ppg = float(row.get('points_per_game', 0) or 0)
        total_points = float(row.get('total_points', 0) or 0)
        
        # Normalize each component to a 0-1 scale (bounded)
        # Form typically ranges 0-10, ppg 0-10, total_points 0-200+
        form_norm = min(form / 10.0, 1.0)
        ppg_norm = min(ppg / 10.0, 1.0)
        total_pts_norm = min(total_points / 150.0, 1.0)
        
        # Weighted combination: form (recent) > ppg (consistency) > total_points (volume)
        # Max momentum_bonus = 2.0 (significant but not overwhelming base score)
        momentum_bonus = 2.0 * (form_norm * 0.45 + ppg_norm * 0.35 + total_pts_norm * 0.20)
        
        # -----------------------------------------------------------------
        # 5. Total score
        # -----------------------------------------------------------------
        score = base_score + fixture_bonus + home_bonus + diff_bonus + momentum_bonus
        
        return round(score, 3)
    
    def _get_available_players(self, position: str) -> list:
        """Get available players for a position, sorted by score descending.

        Applies strategy-specific ownership_floor to differentiate safe/balanced/aggressive.
        """
        pos_df = self.players_df[self.players_df['position'] == position].copy()

        # Filter available players
        available_mask = pos_df.apply(self._is_available, axis=1)
        pos_df = pos_df[available_mask]

        # Filter players with some minutes (at least 45)
        pos_df = pos_df[pos_df['minutes'] >= 45]

        # Apply strategy-specific ownership floor (safe = template-only, aggressive = punts OK)
        params = self.strategy_params.get(self.strategy, self.strategy_params['balanced'])
        ownership_floor = params.get('ownership_floor', 0.0)
        if ownership_floor > 0 and 'selected_by_percent' in pos_df.columns:
            # For safe strategy: only well-owned players. For aggressive: include punts.
            pos_df = pos_df[
                pos_df['selected_by_percent'].astype(float).fillna(0) >= ownership_floor
            ]

        # Calculate scores
        pos_df['score'] = pos_df.apply(self._calculate_score, axis=1)

        # Sort by score descending
        pos_df = pos_df.sort_values('score', ascending=False)

        return pos_df.to_dict('records')

    def _get_bench_candidates(self, position: str, exclude_ids: set) -> list:
        """Get bench candidates sorted by score (desc) then price (desc) for tie-break.

        This ensures we pick high-quality bench options that also maximize spend.
        """
        pos_df = self.players_df[self.players_df['position'] == position].copy()
        
        # Filter available players
        available_mask = pos_df.apply(self._is_available, axis=1)
        pos_df = pos_df[available_mask]
        
        # Exclude already selected
        pos_df = pos_df[~pos_df['id'].isin(exclude_ids)]
        
        # Calculate scores
        pos_df['score'] = pos_df.apply(self._calculate_score, axis=1)
        
        # Sort by score descending, then by price descending (tie-break: spend more)
        pos_df = pos_df.sort_values(['score', 'price'], ascending=[False, False])
        
        return pos_df.to_dict('records')
    
    def build_squad(self) -> Dict:
        """Build optimal 15-player Free Hit squad.
        
        Uses MIP optimization when available to:
        1. Maximize starting XI score (primary objective)
        2. Maximize budget usage as tie-break (secondary objective)
        
        Falls back to greedy selection when MIP is unavailable.
        
        Returns:
            Dict with squad, starting_xi, bench, formation, captain, vice_captain,
            budget info, and league analysis data.
        """
        # Try MIP-based optimization first (if available)
        if MIP_AVAILABLE:
            try:
                result = self._build_squad_mip()
                if result is not None:
                    return result
            except Exception:
                pass  # Fall back to greedy
        
        # Fall back to greedy selection
        return self._build_squad_greedy()
    
    def _build_squad_mip(self) -> Optional[Dict]:
        """Build squad using MIP optimization.
        
        Primary objective: Maximize total XI score
        Secondary objective: Maximize spend (as tie-break, small weight)
        
        Returns:
            Squad dict if successful, None if optimization fails.
        """
        import sasoptpy as so
        import highspy
        
        # Build candidate pool (top players by score per position)
        candidates = []
        candidate_pool_size = 25  # Per position
        
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            pos_candidates = self._get_available_players(pos)[:candidate_pool_size]
            for p in pos_candidates:
                candidates.append({
                    'id': p['id'],
                    'name': p.get('web_name', 'Unknown'),
                    'position': pos,
                    'team_id': p['team'],
                    'price': p['price'],
                    'score': p['score']
                })
        
        if len(candidates) < 15:
            return None  # Not enough candidates
        
        # Create MIP model
        model = so.Model(name='FreeHit_Squad_Optimizer')
        
        # Index sets
        players = list(range(len(candidates)))
        positions = ['GKP', 'DEF', 'MID', 'FWD']
        
        cand = candidates
        
        # Decision variables
        # x[p] = 1 if player p is in the squad
        x = model.add_variables(players, vartype=so.BIN, name='squad')
        
        # lineup[p] = 1 if player p is in starting XI
        lineup = model.add_variables(players, vartype=so.BIN, name='lineup')
        
        # CONSTRAINTS
        
        # 1. Squad size = 15
        model.add_constraint(
            so.quick_sum(x[p] for p in players) == 15,
            name='squad_size'
        )
        
        # 2. Position quotas for full squad
        for pos in positions:
            quota = self.POSITION_QUOTAS[pos]
            model.add_constraint(
                so.quick_sum(x[p] for p in players if cand[p]['position'] == pos) == quota,
                name=f'position_{pos}'
            )
        
        # 3. Max 3 players per team
        team_ids = set(c['team_id'] for c in cand)
        for tid in team_ids:
            model.add_constraint(
                so.quick_sum(x[p] for p in players if cand[p]['team_id'] == tid) <= self.MAX_PER_TEAM,
                name=f'team_{tid}'
            )
        
        # 4. Budget constraint
        model.add_constraint(
            so.quick_sum(x[p] * cand[p]['price'] for p in players) <= self.total_budget,
            name='budget'
        )
        
        # 5. Lineup size = 11
        model.add_constraint(
            so.quick_sum(lineup[p] for p in players) == 11,
            name='lineup_size'
        )
        
        # 6. Can only be in lineup if in squad
        for p in players:
            model.add_constraint(lineup[p] <= x[p], name=f'lineup_squad_{p}')
        
        # 7. Formation constraints (min/max per position in XI)
        for pos in positions:
            pos_lineup = so.quick_sum(
                lineup[p] for p in players if cand[p]['position'] == pos
            )
            model.add_constraint(pos_lineup >= self.XI_MIN[pos], name=f'xi_min_{pos}')
            model.add_constraint(pos_lineup <= self.XI_MAX[pos], name=f'xi_max_{pos}')
        
        # OBJECTIVE: Maximize XI score, with small tie-break for budget usage
        # Primary: sum of scores for starting XI
        # Secondary: total spend (tiny coefficient to act as tie-break only)
        xi_score = so.quick_sum(lineup[p] * cand[p]['score'] for p in players)
        total_spend = so.quick_sum(x[p] * cand[p]['price'] for p in players)
        
        # Tie-break weight: small enough not to sacrifice score for spend
        # Max spend ~100m, max score difference ~50, so 0.001 ensures tie-break only
        objective = xi_score + 0.001 * total_spend
        
        model.set_objective(objective, sense=so.MAX, name='maximize_xi_score')
        
        # Solve using HiGHS
        solution = self._solve_mip_with_highs(model)
        
        if solution is None or solution.get('status') != 'optimal':
            return None
        
        sol_values = solution['solution']
        
        # Extract squad and lineup
        squad = []
        starting_xi = []
        bench = []
        team_counts = {}
        position_counts = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        spent = 0.0
        
        for p in players:
            x_val = sol_values.get(f'squad[{p}]', 0)
            if x_val > 0.5:  # Selected in squad
                player_data = cand[p]
                
                # Get full player info from DataFrame
                pid = player_data['id']
                player_row = self.players_df[self.players_df['id'] == pid].iloc[0].to_dict()
                player_row['score'] = player_data['score']
                
                lineup_val = sol_values.get(f'lineup[{p}]', 0)
                is_starter = lineup_val > 0.5
                
                formatted = self._format_player(player_row, is_starter=is_starter)
                squad.append(formatted)
                
                if is_starter:
                    starting_xi.append(formatted)
                else:
                    bench.append(formatted)
                
                team_counts[player_data['team_id']] = team_counts.get(player_data['team_id'], 0) + 1
                position_counts[player_data['position']] += 1
                spent += player_data['price']
        
        # Sort starting XI and bench
        pos_order = {'GKP': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
        starting_xi.sort(key=lambda x: (pos_order.get(x['position'], 4), -x['score']))
        bench.sort(key=lambda x: (0 if x['position'] == 'GKP' else 1, -x['score']))
        
        # Determine formation
        formation = self._get_formation(starting_xi)
        
        # Select captain and vice captain
        captain, vice_captain = self._select_captains(starting_xi)
        
        # Analyze differentials
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
    
    def _solve_mip_with_highs(self, model) -> Optional[Dict]:
        """Solve sasoptpy model using HiGHS solver."""
        import highspy
        import re
        
        fd, mps_file = tempfile.mkstemp(suffix='.mps')
        os.close(fd)
        
        try:
            model.export_mps(filename=mps_file)
            
            # Fix MPS format for HiGHS
            with open(mps_file, 'r') as f:
                content = f.read()
            is_max = ' MAX ' in content or '\tMAX\t' in content
            content = re.sub(r'(\s)(MAX|MIN)(\s+)', r'\1N  \3', content)
            with open(mps_file, 'w') as f:
                f.write(content)
            
            # Solve with HiGHS
            h = highspy.Highs()
            h.setOptionValue('time_limit', 30.0)
            h.setOptionValue('output_flag', False)
            h.readModel(mps_file)
            
            if is_max:
                h.changeObjectiveSense(highspy.ObjSense.kMaximize)
            
            h.run()
            status = h.getModelStatus()
            
            if status == highspy.HighsModelStatus.kOptimal:
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
                    sasoptpy_vars = list(model.get_variables())
                    for i, val in enumerate(col_values):
                        if i < len(sasoptpy_vars):
                            solution[sasoptpy_vars[i].get_name()] = val
                
                return {'status': 'optimal', 'solution': solution}
            
            return None
            
        except Exception:
            return None
        finally:
            if os.path.exists(mps_file):
                try:
                    os.unlink(mps_file)
                except:
                    pass
    
    def _get_formation(self, starting_xi: List[Dict]) -> str:
        """Determine formation from starting XI."""
        counts = {'DEF': 0, 'MID': 0, 'FWD': 0}
        for p in starting_xi:
            pos = p['position']
            if pos in counts:
                counts[pos] += 1
        return f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"
    
    def _build_squad_greedy(self) -> Dict:
        """Build squad using greedy selection (fallback when MIP unavailable).
        
        Strategy:
        1. Select best XI first (greedy by score, respecting constraints)
        2. Fill remaining slots with high-score players (using price as tie-break)
        3. Determine best starting XI formation
        4. Select captain and vice captain
        """
        squad = []
        team_counts = {}
        position_counts = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        selected_ids = set()
        spent = 0.0
        
        # Phase 1: Build a strong starting XI (11 players)
        xi_targets = {'GKP': 1, 'DEF': 4, 'MID': 4, 'FWD': 2}  # Default 4-4-2
        
        # Get sorted players by position (by score desc, then price desc)
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
        
        # Phase 2: Fill remaining quota with bench candidates (score+price sorted)
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            remaining = self.POSITION_QUOTAS[pos] - position_counts[pos]
            if remaining <= 0:
                continue
                
            bench_candidates = self._get_bench_candidates(pos, selected_ids)
            
            for player in bench_candidates:
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
