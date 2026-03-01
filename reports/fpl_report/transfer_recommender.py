"""Transfer Recommender Module

Identifies underperforming players and suggests optimal replacements
based on form, fixtures, value, and expected metrics. Includes fixture
swing detection and model performance metrics for transfer planning.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .predictor import FPLPointsPredictor
from .fpl_core_predictor import FPLCorePredictor

logger = logging.getLogger(__name__)

# Try to import new inference pipeline (position-specific models)
try:
    import sys
    from pathlib import Path
    # Add project root to path for models import
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from models.inference import FPLInferencePipeline
    NEW_MODELS_AVAILABLE = True
except ImportError:
    NEW_MODELS_AVAILABLE = False


class TransferRecommender:
    """Recommends transfers based on player analysis and upcoming fixtures."""

    # Weights for replacement scoring
    WEIGHTS = {
        'form': 0.18,
        'fixtures': 0.18,
        'expected_points': 0.08,
        'predicted_points': 0.28,  # ML model prediction
        'ownership_diff': 0.08,
        'value': 0.10,
        'transfer_momentum': 0.10,  # NEW: bandwagon/panic signal
    }
    
    # Transfer momentum thresholds
    MOMENTUM_THRESHOLDS = {
        'bandwagon': 0.05,    # >5% net inflow = bandwagon
        'falling_knife': -0.05,  # <-5% net outflow = falling knife
    }

    def __init__(self, data_fetcher, player_analyzer, use_new_models: bool = True,
                 use_fpl_core_predictor: bool = True,
                 all_gw_data: Dict = None,
                 fpl_core_season_data: Dict = None,
                 current_gw: int = None,
                 # Cross-season training parameters (optional)
                 prev_all_gw_data: Dict = None,
                 prev_fpl_core_season_data: Dict = None,
                 prev_season_start_year: int = None,
                 current_season_start_year: int = None):
        """Initialize with data fetcher and analyzer.

        Args:
            data_fetcher: FPLDataFetcher instance.
            player_analyzer: PlayerAnalyzer instance.
            use_new_models: If True, use position-specific ML models when available.
            use_fpl_core_predictor: If True, use the enhanced FPL Core predictor with Opta-like features.
            all_gw_data: Dict of gameweek data from FPL Core Insights (required for FPL Core predictor).
            fpl_core_season_data: Season-level data from FPL Core Insights (required for FPL Core predictor).
            current_gw: Current gameweek number.
            prev_all_gw_data: Previous season gameweek data for cross-season training.
            prev_fpl_core_season_data: Previous season-level data for cross-season training.
            prev_season_start_year: Start year of previous season (e.g., 2024 for 2024-25).
            current_season_start_year: Start year of current season (e.g., 2025 for 2025-26).
        """
        self.fetcher = data_fetcher
        self.analyzer = player_analyzer
        self.all_gw_data = all_gw_data
        self.fpl_core_season_data = fpl_core_season_data
        self.current_gw = current_gw
        self.use_fpl_core_predictor = use_fpl_core_predictor
        
        # Cross-season training data
        self.prev_all_gw_data = prev_all_gw_data
        self.prev_fpl_core_season_data = prev_fpl_core_season_data
        self.prev_season_start_year = prev_season_start_year
        self.current_season_start_year = current_season_start_year
        
        # Check if cross-season training is available
        self._cross_season_available = (
            prev_all_gw_data is not None and
            prev_fpl_core_season_data is not None and
            prev_season_start_year is not None and
            current_season_start_year is not None
        )
        
        # Initialize predictor based on preference
        if use_fpl_core_predictor and all_gw_data and fpl_core_season_data:
            if self._cross_season_available:
                print("Using FPL Core Insights predictor with cross-season training")
            else:
                print("Using FPL Core Insights predictor with 40+ Opta-like features")
            self.predictor = FPLCorePredictor()
            self._fpl_core_predictor_ready = True
        else:
            self.predictor = FPLPointsPredictor(data_fetcher)
            self._fpl_core_predictor_ready = False
        
        # New inference pipeline (position-specific models)
        self.inference_pipeline = None
        self._xp_cache = {}  # Cache for ML predictions
        if use_new_models and NEW_MODELS_AVAILABLE:
            try:
                self.inference_pipeline = FPLInferencePipeline(use_fallback=True)
                print("Using position-specific ML models for predictions")
            except Exception as e:
                print(f"Warning: Could not initialize new inference pipeline: {e}")
    
    def calculate_transfer_momentum(self, player_data: Dict) -> Dict:
        """Calculate transfer momentum signal for a player.
        
        Transfer momentum = (transfers_in - transfers_out) / (ownership% * total_players)
        
        Args:
            player_data: Player stats dictionary from API.
            
        Returns:
            Dict with momentum value, classification, and raw transfer counts.
        """
        transfers_in = int(player_data.get('transfers_in_event', 0) or 0)
        transfers_out = int(player_data.get('transfers_out_event', 0) or 0)
        ownership_pct = float(player_data.get('selected_by_percent', 0) or 0)
        
        # Estimate total players (roughly 10M players)
        total_players = 10_000_000
        
        # Calculate net transfers as percentage of ownership
        net_transfers = transfers_in - transfers_out
        ownership_base = max(ownership_pct * total_players / 100, 1000)  # Avoid div by 0
        
        momentum = net_transfers / ownership_base
        
        # Classify
        if momentum > self.MOMENTUM_THRESHOLDS['bandwagon']:
            classification = 'bandwagon'
            trend_icon = '📈'
        elif momentum < self.MOMENTUM_THRESHOLDS['falling_knife']:
            classification = 'falling_knife'
            trend_icon = '📉'
        else:
            classification = 'neutral'
            trend_icon = '➡️'
        
        return {
            'momentum': round(momentum, 4),
            'classification': classification,
            'trend_icon': trend_icon,
            'transfers_in': transfers_in,
            'transfers_out': transfers_out,
            'net_transfers': net_transfers,
        }



    def _is_player_available(self, player_data: Dict) -> bool:
        """Check if a player is available (not injured/suspended).

        Args:
            player_data: Player stats dictionary.

        Returns:
            True if player is fit and available to play.
        """
        # Status codes: 'a' = available, 'i' = injured, 's' = suspended, 'd' = doubtful, 'u' = unavailable
        status = player_data.get('status', 'a')
        if status != 'a':
            return False

        # Check chance of playing - None means 100%, otherwise check threshold
        chance = player_data.get('chance_of_playing_next_round')
        if chance is not None and chance < 75:
            return False

        return True

    def identify_underperformers(self, squad_analysis: List[Dict],
                                quadrant_data: Dict = None) -> List[Dict]:
        """Identify underperforming players in the squad.

        Uses continuous metrics from PlayerAnalyzer (slope magnitude, per-metric
        percentiles, assist underperformance) plus suspension risk and quadrant
        classification for a nuanced severity score.

        Args:
            squad_analysis: List of player analysis dictionaries.
            quadrant_data: Optional dict mapping player_id -> quadrant label
                (CLINICAL, VOLUME, ELITE, AVOID) from usage/output analysis.

        Returns:
            List of underperformer dictionaries with reasons, sorted by severity.
        """
        underperformers = []
        quadrant_data = quadrant_data or {}

        for player in squad_analysis:
            reasons = []
            severity = 0  # Higher = more urgent

            # -- Form analysis (enhanced: use slope magnitude) --
            form_data = player.get('form_analysis', {})
            form_avg = form_data.get('average', 0)
            if form_avg < 3.0:
                reasons.append(f"Low form ({form_avg:.1f})")
                severity += 3

            trend_slope = form_data.get('trend_slope', 0)
            if form_data.get('trend') == 'falling':
                # Scale severity by slope magnitude (steeper decline = worse)
                slope_severity = min(abs(trend_slope) * 2, 5)
                reasons.append(f"Declining form (slope {trend_slope:+.1f})")
                severity += max(2, round(slope_severity))

            # -- Expected vs actual (enhanced: check assists too) --
            exp_data = player.get('expected_vs_actual', {})
            if exp_data.get('goals_performance') == 'under':
                xg = exp_data.get('expected_goals', 0)
                goals = exp_data.get('actual_goals', 0)
                if xg > 1:
                    reasons.append(f"Underperforming xG ({goals} vs {xg:.1f} expected)")
                    severity += 1
            if exp_data.get('assists_performance') == 'under':
                xa = exp_data.get('expected_assists', 0)
                assists = exp_data.get('actual_assists', 0)
                if xa > 1:
                    reasons.append(f"Underperforming xA ({assists} vs {xa:.1f} expected)")
                    severity += 1

            # -- Minutes check --
            minutes = player.get('raw_stats', {}).get('minutes', 0)
            gws_played = len(form_data.get('recent_points', []))
            if gws_played > 0:
                mins_per_gw = minutes / max(gws_played, 1)
                if mins_per_gw < 60:
                    reasons.append(f"Limited minutes ({mins_per_gw:.0f}/GW)")
                    severity += 2

            # -- Peer comparison (enhanced: use per-metric percentiles) --
            peer_data = player.get('peer_comparison', {})
            overall_rating = peer_data.get('overall_rating', 100)
            if overall_rating < 25:
                reasons.append("Bottom quartile vs peers")
                severity += 2
            # Check individual metric weakness
            percentiles = peer_data.get('percentiles', {})
            for metric in ('total_points', 'form', 'ict_index'):
                pctl = percentiles.get(metric, 50)
                if pctl < 15:
                    reasons.append(f"Bottom 15% in {metric.replace('_', ' ')}")
                    severity += 1
                    break  # One flag is enough

            # -- Suspension risk (yellow card proximity) --
            stats = player.get('raw_stats', {})
            yellow_cards = stats.get('yellow_cards', 0)
            if yellow_cards >= 4:
                reasons.append(f"Suspension risk ({yellow_cards} yellows, banned at 5)")
                severity += 3
            elif yellow_cards >= 3:
                reasons.append(f"Yellow card warning ({yellow_cards}/5)")
                severity += 1

            # -- Injury/unavailability --
            if minutes == 0:
                reasons.append("No minutes this season")
                severity += 4
            chance = stats.get('chance_of_playing_next_round')
            if chance is not None and chance <= 25:
                reasons.append(f"Injury doubt ({chance}% chance)")
                severity += 3

            # -- Quadrant classification (regression risk) --
            pid = player.get('player_id')
            quadrant = quadrant_data.get(pid, '')
            if quadrant == 'CLINICAL':
                reasons.append("CLINICAL quadrant (regression risk)")
                severity += 2

            if reasons:
                underperformers.append({
                    'player_id': pid,
                    'name': player.get('name'),
                    'position': player.get('position'),
                    'team': player.get('team'),
                    'price': player.get('price'),
                    'reasons': reasons,
                    'severity': severity,
                    'current_form': form_avg,
                    'total_points': stats.get('total_points', 0),
                    'quadrant': quadrant,
                })

        # Sort by severity (most urgent first)
        return sorted(underperformers, key=lambda x: x['severity'], reverse=True)

    def score_replacement(self, candidate: Dict, budget: float,
                          fixtures: List[Dict], current_player: Dict,
                          predicted_points: float = 0.0,
                          quadrant_data: Dict = None,
                          differential_targets: set = None) -> float:
        """Score a potential replacement player.

        Args:
            candidate: Candidate player stats dictionary.
            budget: Available budget in millions.
            fixtures: Upcoming fixtures for the candidate's team.
            current_player: Current player being replaced.
            predicted_points: RF Model prediction for next GW.
            quadrant_data: Optional dict mapping player_id -> quadrant label.
            differential_targets: Optional set of player IDs owned by top
                managers but not by user (template convergence signal).

        Returns:
            Weighted score (0-100).
        """
        quadrant_data = quadrant_data or {}
        differential_targets = differential_targets or set()
        scores = {}

        # Form score (0-100)
        form = float(candidate.get('form', 0) or 0)
        scores['form'] = min(form * 10, 100)

        # Fixture score (0-100) using Elo win probabilities when available
        if fixtures:
            position_type = candidate.get('element_type', 3)
            is_defensive = position_type in (1, 2)  # GKP or DEF
            fix_scores = []
            for f in fixtures[:3]:
                win_prob = f.get('win_prob')
                draw_prob = f.get('draw_prob')
                if win_prob is not None and draw_prob is not None:
                    if is_defensive:
                        # Defenders/GKPs benefit from wins AND draws (clean sheets)
                        fix_scores.append(win_prob * 0.4 + draw_prob * 0.6)
                    else:
                        # Attackers/mids benefit primarily from wins (goals/assists)
                        fix_scores.append(win_prob)
                else:
                    # Fallback to FDR inversion when Elo data unavailable
                    diff = f.get('difficulty', 3)
                    fix_scores.append((5 - diff) / 4)
            scores['fixtures'] = np.mean(fix_scores) * 100 if fix_scores else 50
        else:
            scores['fixtures'] = 50

        # Expected points score (0-100) - based on season stats
        xgi = float(candidate.get('expected_goal_involvements', 0) or 0)
        ppg = float(candidate.get('points_per_game', 0) or 0)
        scores['expected_points'] = min((xgi * 20 + ppg * 10), 100)
        
        # Predicted points score (0-100) - based on RF model
        # Cap at 10 points for normalization (10 pts = 100 score)
        scores['predicted_points'] = min(predicted_points * 10, 100)

        # Ownership differential score (0-100)
        ownership = float(candidate.get('selected_by_percent', 0) or 0)
        # Reward players under 20% ownership (differential picks)
        if ownership < 5:
            scores['ownership_diff'] = 90
        elif ownership < 10:
            scores['ownership_diff'] = 70
        elif ownership < 20:
            scores['ownership_diff'] = 50
        else:
            scores['ownership_diff'] = 30

        # Value score (0-100)
        price = float(candidate.get('now_cost', 0) or 0) / 10
        if price <= budget:
            # Value = points per million
            if price > 0:
                value_ratio = float(candidate.get('total_points', 0) or 0) / price
                scores['value'] = min(value_ratio * 5, 100)
            else:
                scores['value'] = 50
        else:
            scores['value'] = 0  # Can't afford
        
        # Transfer momentum score (0-100) - NEW
        momentum_data = self.calculate_transfer_momentum(candidate)
        if momentum_data['classification'] == 'bandwagon':
            scores['transfer_momentum'] = 80  # High demand = good signal
        elif momentum_data['classification'] == 'falling_knife':
            scores['transfer_momentum'] = 30  # Being mass-sold = risk signal
        else:
            scores['transfer_momentum'] = 50  # Neutral

        # Calculate weighted total
        total = sum(scores.get(k, 50) * self.WEIGHTS[k] for k in self.WEIGHTS)

        # Quadrant bonus: VOLUME candidates are buy signals (high usage, due returns)
        cid = candidate.get('id', 0)
        quadrant = quadrant_data.get(cid, '')
        if quadrant == 'VOLUME':
            total += 5  # Modest boost for statistically due players

        # Template convergence: boost players owned by top managers but not by user
        if cid in differential_targets:
            total += 8  # Significant boost for matching top-manager template

        return round(total, 1)

    def get_recommendations(self, underperformers: List[Dict],
                            num_recommendations: int = 3,
                            quadrant_data: Dict = None,
                            differential_targets: set = None) -> List[Dict]:
        """Get transfer recommendations for underperforming players.

        Args:
            underperformers: List of underperforming player dicts.
            num_recommendations: Max replacements to suggest per player.
            quadrant_data: Optional quadrant labels for buy-signal scoring.
            differential_targets: Optional set of player IDs for template convergence.

        Returns:
            List of recommendation dictionaries.
        """
        budget = self.fetcher.get_bank()
        recommendations = []
        
        # Ensure model is trained (loads pre-trained if available)
        print("Ensuring prediction model is ready...")
        self._ensure_model_trained()
        
        # Collect candidate player IDs for prediction
        all_candidate_ids = []
        for player in underperformers[:5]:
            position = player['position']
            candidates = self.fetcher.get_all_players_by_position(position)
            candidates = candidates[candidates['minutes'] >= 450]
            all_candidate_ids.extend(candidates['id'].tolist())
            
        unique_ids = list(set(all_candidate_ids))
        if len(unique_ids) > 200:
            unique_ids = unique_ids[:200]
        
        # Get predictions
        if self._fpl_core_predictor_ready and isinstance(self.predictor, FPLCorePredictor):
            predictions = self.predictor.predict(
                self.all_gw_data, self.fpl_core_season_data, unique_ids, self.current_gw
            )
        else:
            predictions = self.predictor.predict(unique_ids)

        for player in underperformers[:5]:  # Top 5 underperformers
            position = player['position']
            current_price = player.get('price', 4.0) or 4.0  # Default to 4.0 if None
            total_budget = budget + current_price

            # Get all players in same position
            candidates = self.fetcher.get_all_players_by_position(position)

            # Filter affordable candidates - use lower minutes threshold for budget players
            min_minutes = 45 if total_budget < 6.0 else 90
            candidates = candidates[
                (candidates['now_cost'] / 10 <= total_budget) &
                (candidates['minutes'] >= min_minutes)
            ].copy()

            # Exclude current player
            player_id = player.get('player_id')
            if player_id:
                candidates = candidates[candidates['id'] != player_id]

            # Filter out injured/unavailable players
            available_mask = candidates.apply(
                lambda row: self._is_player_available(row.to_dict()), axis=1
            )
            candidates = candidates[available_mask]

            # Get peer stats for percentile calculation
            all_peers = self.fetcher.get_position_peers(position, min_minutes=90)

            # Score each candidate
            scored_candidates = []
            for _, cand in candidates.iterrows():
                cand_dict = cand.to_dict()
                pid = int(cand_dict.get('id', 0))
                team_id = int(cand_dict.get('team', 0))
                fixtures = self.fetcher.get_upcoming_fixtures(team_id, num_fixtures=5)
                
                # Get predicted points
                pred_points = predictions.get(pid, 0.0)

                score = self.score_replacement(
                    cand_dict, total_budget, fixtures, player, pred_points,
                    quadrant_data=quadrant_data,
                    differential_targets=differential_targets
                )

                # Calculate peer percentile
                peer_pct = 50  # Default
                if not all_peers.empty:
                    try:
                        cand_pts = float(cand_dict.get('total_points', 0) or 0)
                        peer_pts = pd.to_numeric(all_peers['total_points'], errors='coerce').dropna().values
                        if len(peer_pts) > 0:
                            from scipy import stats as scipy_stats
                            peer_pct = round(scipy_stats.percentileofscore(peer_pts, cand_pts), 0)
                    except:
                        pass

                # Calculate average fixture difficulty
                avg_fdr = 3.0  # Default medium
                if fixtures:
                    avg_fdr = round(np.mean([f.get('difficulty', 3) for f in fixtures[:3]]), 1)

                if score > 20:  # Lower threshold to ensure we get suggestions
                    # Get momentum data for display
                    momentum_data = self.calculate_transfer_momentum(cand_dict)
                    
                    scored_candidates.append({
                        'player_id': pid,
                        'name': cand_dict.get('web_name', 'Unknown'),
                        'team': self.fetcher._get_team_name(team_id),
                        'price': round(float(cand_dict.get('now_cost', 0)) / 10, 1),
                        'form': round(float(cand_dict.get('form', 0) or 0), 1),
                        'total_points': int(cand_dict.get('total_points', 0) or 0),
                        'ownership': round(float(cand_dict.get('selected_by_percent', 0) or 0), 1),
                        'score': score,
                        'fixtures': fixtures[:3],
                        'ppg': round(float(cand_dict.get('points_per_game', 0) or 0), 2),
                        'peer_rank': int(peer_pct),
                        'avg_fdr': avg_fdr,
                        'predicted_points': pred_points,
                        # NEW: Transfer momentum signals
                        'transfer_trend': momentum_data['trend_icon'],
                        'transfer_momentum': momentum_data['classification'],
                        'net_transfers': momentum_data['net_transfers'],
                    })

            # Sort by score and take top N
            scored_candidates.sort(key=lambda x: x['score'], reverse=True)
            top_replacements = scored_candidates[:num_recommendations]

            recommendations.append({
                'out': player,
                'in_options': top_replacements,
                'budget_after': round(total_budget - (top_replacements[0]['price'] if top_replacements else 0), 1)
            })

        return recommendations

    def format_fixture_difficulty(self, fixtures: List[Dict]) -> str:
        """Format fixtures as a readable string with difficulty colors.

        Args:
            fixtures: List of fixture dictionaries.

        Returns:
            Formatted string like "ARS(H)2 MUN(A)4 CHE(H)3"
        """
        parts = []
        for f in fixtures[:5]:
            venue = 'H' if f.get('is_home') else 'A'
            diff = f.get('difficulty', 3)
            opp = f.get('opponent', 'UNK')
            parts.append(f"{opp}({venue}){diff}")
        return ' '.join(parts)

    def get_best_captain_picks(self, squad_analysis: List[Dict],
                                num_picks: int = 3) -> List[Dict]:
        """Suggest best captain options from squad.

        Args:
            squad_analysis: List of player analyses.
            num_picks: Number of picks to return.

        Returns:
            Top captain picks with reasoning.
        """
        captain_scores = []

        for player in squad_analysis:
            # Skip bench players (positions 12-15 typically)
            if player.get('position_in_squad', 0) > 11:
                continue

            score = 0
            reasons = []

            # Form contribution
            form = player.get('form_analysis', {}).get('average', 0)
            score += form * 15
            if form >= 6:
                reasons.append(f"Excellent form ({form:.1f})")

            # Fixture difficulty (get from team)
            team_id = player.get('stats', {}).get('team', 0)
            fixtures = self.fetcher.get_upcoming_fixtures(team_id, num_fixtures=1)
            if fixtures:
                diff = fixtures[0].get('difficulty', 3)
                fixture_score = (5 - diff) * 10
                score += fixture_score
                if diff <= 2:
                    venue = 'Home' if fixtures[0].get('is_home') else 'Away'
                    reasons.append(f"Easy fixture ({fixtures[0].get('opponent', 'UNK')} {venue})")

            # Historical points
            ppg = player.get('expected_vs_actual', {}).get('points_per_game', 0)
            score += ppg * 8

            # ICT
            ict = player.get('ict_analysis', {}).get('ict_index', 0)
            score += ict / 10
            
            # Dream Team consistency score (NEW)
            # Players who regularly make the Dream Team are reliable captaincy options
            raw_stats = player.get('raw_stats', {})
            dreamteam_count = raw_stats.get('dreamteam_count', 0)
            if dreamteam_count is None or dreamteam_count == 0:
                # Try to get from player stats directly
                stats = self.fetcher.get_player_stats(player.get('player_id', 0))
                dreamteam_count = int(stats.get('dreamteam_count', 0) or 0)
            
            # Bonus: +5 points per Dream Team appearance, capped at 25
            if dreamteam_count > 0:
                dreamteam_bonus = min(dreamteam_count * 5, 25)
                score += dreamteam_bonus
                
                if dreamteam_count >= 3:
                    reasons.append(f"Dream Team regular ({dreamteam_count}x)")
            
            # Fallback reasons if none qualified
            if not reasons:
                if ppg >= 5:
                    reasons.append(f"Strong PPG ({ppg:.1f})")
                elif ict >= 50:
                    reasons.append(f"High ICT ({ict:.0f})")
                else:
                    reasons.append("Consistent performer")

            captain_scores.append({
                'name': player.get('name'),
                'position': player.get('position'),
                'team': player.get('team'),
                'score': round(score, 1),
                'form': form,
                'reasons': reasons[:2],
                'fixture': fixtures[0] if fixtures else None
            })

        captain_scores.sort(key=lambda x: x['score'], reverse=True)
        return captain_scores[:num_picks]

    def get_differential_picks(self, position: str,
                                max_ownership: float = 10.0,
                                budget: float = 15.0) -> List[Dict]:
        """Find differential picks (low ownership, high potential).

        Args:
            position: Position code (GKP, DEF, MID, FWD).
            max_ownership: Maximum ownership percentage.
            budget: Maximum price in millions.

        Returns:
            List of differential picks.
        """
        candidates = self.fetcher.get_position_peers(position, min_minutes=180)

        differentials = candidates[
            (candidates['selected_by_percent'] <= max_ownership) &
            (candidates['now_cost'] / 10 <= budget) &
            (candidates['form'] >= 3)  # Minimum form
        ].copy()

        # Filter out injured/unavailable players
        available_mask = differentials.apply(
            lambda row: self._is_player_available(row.to_dict()), axis=1
        )
        differentials = differentials[available_mask]

        differentials = differentials.sort_values('form', ascending=False)

        picks = []
        for _, player in differentials.head(5).iterrows():
            team_id = int(player.get('team', 0))
            fixtures = self.fetcher.get_upcoming_fixtures(team_id, num_fixtures=3)

            picks.append({
                'name': player.get('web_name', 'Unknown'),
                'team': self.fetcher._get_team_name(team_id),
                'price': round(float(player.get('now_cost', 0)) / 10, 1),
                'form': round(float(player.get('form', 0) or 0), 1),
                'ownership': round(float(player.get('selected_by_percent', 0) or 0), 1),
                'total_points': int(player.get('total_points', 0) or 0),
                'fixtures': self.format_fixture_difficulty(fixtures)
            })

        return picks

    def identify_fixture_swing(self, team_id: int, 
                                num_weeks: int = 5) -> Dict:
        """Identify fixture difficulty swings for a team.
        
        Compares early fixtures vs later fixtures to detect improving
        or worsening runs.
        
        Args:
            team_id: FPL team ID (1-20).
            num_weeks: Number of weeks to analyze.
            
        Returns:
            Dict with swing analysis:
            {
                'fixtures': [...],
                'early_avg': float,
                'late_avg': float,
                'swing': 'improving'|'worsening'|'neutral',
                'swing_gw': int or None,
                'difficulty_trend': [...]
            }
        """
        fixtures = self.fetcher.get_upcoming_fixtures(team_id, num_weeks)
        
        if not fixtures:
            return {
                'fixtures': [],
                'early_avg': 3.0,
                'late_avg': 3.0,
                'swing': 'neutral',
                'swing_gw': None,
                'difficulty_trend': []
            }
        
        difficulties = [f.get('difficulty', 3) for f in fixtures]
        
        # Split into early (first 2) and late (remaining)
        early_avg = np.mean(difficulties[:2]) if len(difficulties) >= 2 else 3.0
        late_avg = np.mean(difficulties[2:]) if len(difficulties) > 2 else 3.0
        
        # Determine swing direction
        swing_diff = late_avg - early_avg
        if swing_diff < -0.5:
            swing = 'improving'
        elif swing_diff > 0.5:
            swing = 'worsening'
        else:
            swing = 'neutral'
        
        # Find specific swing point (GW where difficulty changes significantly)
        swing_gw = None
        if len(difficulties) >= 2:
            for i in range(1, len(difficulties)):
                diff_change = difficulties[i] - difficulties[i-1]
                if abs(diff_change) >= 1.5:
                    swing_gw = i + 1  # 1-indexed relative to planning horizon
                    break
        
        return {
            'fixtures': fixtures,
            'early_avg': round(early_avg, 2),
            'late_avg': round(late_avg, 2),
            'swing': swing,
            'swing_gw': swing_gw,
            'difficulty_trend': difficulties
        }

    def get_squad_fixture_swings(self, squad_analysis: List[Dict], 
                                  num_weeks: int = 5) -> Dict[int, Dict]:
        """Get fixture swing analysis for entire squad.
        
        Args:
            squad_analysis: List of player analysis dicts.
            num_weeks: Planning horizon.
            
        Returns:
            Dict mapping player_id to swing analysis.
        """
        swings = {}
        
        for player in squad_analysis:
            pid = player.get('player_id')
            team_id = player.get('stats', {}).get('team', 0)
            
            if pid and team_id:
                swings[pid] = self.identify_fixture_swing(team_id, num_weeks)
                swings[pid]['player_name'] = player.get('name', 'Unknown')
                swings[pid]['position'] = player.get('position', 'UNK')
        
        return swings

    def get_model_performance_metrics(self) -> Dict:
        """Get performance metrics from the prediction model.
        
        Returns:
            Dict with MAE, RMSE, R², sample counts, and confidence level.
        """
        metrics = self.predictor.get_model_metrics()
        # FPL Core predictor may not have get_confidence_level
        if hasattr(self.predictor, 'get_confidence_level'):
            metrics['confidence'] = self.predictor.get_confidence_level()
        else:
            # Calculate confidence based on R²
            r2 = metrics.get('r2', 0) or 0
            mae = metrics.get('mae', 999) or 999
            if r2 > 0.3 and mae < 2.0:
                metrics['confidence'] = 'high'
            elif r2 > 0.15 or mae < 2.5:
                metrics['confidence'] = 'medium'
            else:
                metrics['confidence'] = 'low'
        return metrics

    def _ensure_model_trained(self):
        """Ensure the prediction model is trained.
        
        First tries to load pre-trained models from disk (faster).
        Falls back to training from scratch if no saved models found.
        """
        if self.predictor.is_trained:
            return
        
        if self._fpl_core_predictor_ready and isinstance(self.predictor, FPLCorePredictor):
            # Try loading pre-trained models from disk first
            if self.predictor.load_from_disk():
                logger.info("Using pre-trained FPL Core Predictor models")
                return
            
            # Fall back to training if no saved models
            logger.info("No pre-trained models found, training from scratch...")
            
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
        if self._fpl_core_predictor_ready and isinstance(self.predictor, FPLCorePredictor):
            return self.predictor.predict_multiple_gws(
                self.all_gw_data, self.fpl_core_season_data,
                player_ids, self.current_gw, num_gws
            )
        else:
            return self.predictor.predict_multiple_gws(player_ids, num_gws)

    def calculate_expected_value(self, player_id: int, 
                                  num_gws: int = 5) -> Dict:
        """Calculate expected value for a player over multiple gameweeks.
        
        Args:
            player_id: Player ID.
            num_gws: Number of gameweeks to project.
            
        Returns:
            Dict with predictions and cumulative expected value.
        """
        self._ensure_model_trained()
        
        result = self._predict_multiple_gws([player_id], num_gws)
        return result.get(player_id, {
            'predictions': [0] * num_gws,
            'cumulative': 0,
            'confidence': 'low',
            'std_dev': 0,
            'avg_per_gw': 0
        })

    def compare_transfer_ev(self, player_out_id: int, player_in_id: int,
                            num_gws: int = 5) -> Dict:
        """Compare expected value between two players over planning horizon.
        
        Args:
            player_out_id: Current player ID.
            player_in_id: Replacement player ID.
            num_gws: Planning horizon.
            
        Returns:
            Dict with comparison data:
            {
                'player_out_ev': cumulative EV,
                'player_in_ev': cumulative EV,
                'expected_gain': difference,
                'per_gw_breakdown': comparison per GW
            }
        """
        self._ensure_model_trained()
        
        predictions = self._predict_multiple_gws(
            [player_out_id, player_in_id], num_gws
        )
        
        out_pred = predictions.get(player_out_id, {'predictions': [0]*num_gws, 'cumulative': 0})
        in_pred = predictions.get(player_in_id, {'predictions': [0]*num_gws, 'cumulative': 0})
        
        # Build per-GW comparison
        per_gw = []
        current_gw = self.fetcher.get_current_gameweek()
        for i in range(num_gws):
            out_pts = out_pred['predictions'][i] if i < len(out_pred['predictions']) else 0
            in_pts = in_pred['predictions'][i] if i < len(in_pred['predictions']) else 0
            per_gw.append({
                'gameweek': current_gw + i + 1,
                'player_out_pts': round(out_pts, 1),
                'player_in_pts': round(in_pts, 1),
                'gain': round(in_pts - out_pts, 1)
            })
        
        return {
            'player_out_ev': out_pred['cumulative'],
            'player_in_ev': in_pred['cumulative'],
            'expected_gain': round(in_pred['cumulative'] - out_pred['cumulative'], 1),
            'per_gw_breakdown': per_gw,
            'confidence_out': out_pred.get('confidence', 'unknown'),
            'confidence_in': in_pred.get('confidence', 'unknown')
        }

    def get_fixture_heatmap_data(self, squad_analysis: List[Dict],
                                  num_weeks: int = 5) -> List[Dict]:
        """Generate fixture difficulty data for heatmap visualization.
        
        Args:
            squad_analysis: List of player analysis dicts.
            num_weeks: Number of gameweeks.
            
        Returns:
            List of dicts with player info and fixture difficulties.
        """
        heatmap_data = []
        current_gw = self.fetcher.get_current_gameweek()
        
        for player in squad_analysis:
            pid = player.get('player_id')
            team_id = player.get('stats', {}).get('team', 0)
            
            fixtures = self.fetcher.get_upcoming_fixtures(team_id, num_weeks)
            
            row = {
                'player_id': pid,
                'name': player.get('name', 'Unknown'),
                'position': player.get('position', 'UNK'),
                'team': player.get('team', 'UNK'),
                'gameweeks': []
            }
            
            for i in range(num_weeks):
                gw = current_gw + i + 1
                if i < len(fixtures):
                    fix = fixtures[i]
                    row['gameweeks'].append({
                        'gw': gw,
                        'opponent': fix.get('opponent', 'BGW'),
                        'is_home': fix.get('is_home', False),
                        'difficulty': fix.get('difficulty', 0)
                    })
                else:
                    row['gameweeks'].append({
                        'gw': gw,
                        'opponent': 'BGW',
                        'is_home': False,
                        'difficulty': 0
                    })
            
            heatmap_data.append(row)
        
        # Sort by position
        position_order = {'GKP': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
        heatmap_data.sort(key=lambda x: position_order.get(x['position'], 4))
        
        return heatmap_data
