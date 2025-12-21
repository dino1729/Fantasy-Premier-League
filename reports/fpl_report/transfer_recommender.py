"""Transfer Recommender Module

Identifies underperforming players and suggests optimal replacements
based on form, fixtures, value, and expected metrics. Includes fixture
swing detection and model performance metrics for transfer planning.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .predictor import FPLPointsPredictor

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

    def __init__(self, data_fetcher, player_analyzer, use_new_models: bool = True):
        """Initialize with data fetcher and analyzer.

        Args:
            data_fetcher: FPLDataFetcher instance.
            player_analyzer: PlayerAnalyzer instance.
            use_new_models: If True, use position-specific ML models when available.
        """
        self.fetcher = data_fetcher
        self.analyzer = player_analyzer
        self.predictor = FPLPointsPredictor(data_fetcher)
        
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

    def identify_underperformers(self, squad_analysis: List[Dict]) -> List[Dict]:
        """Identify underperforming players in the squad.

        Criteria:
        - Form < 3.0
        - Underperforming xG/xA by > 30%
        - Minutes < 60 per game average
        - Bottom 25% peer ranking
        - Falling form trend

        Args:
            squad_analysis: List of player analysis dictionaries.

        Returns:
            List of underperformer dictionaries with reasons.
        """
        underperformers = []

        for player in squad_analysis:
            reasons = []
            severity = 0  # Higher = more urgent

            # Check form
            form_data = player.get('form_analysis', {})
            form_avg = form_data.get('average', 0)
            if form_avg < 3.0:
                reasons.append(f"Low form ({form_avg:.1f})")
                severity += 3

            if form_data.get('trend') == 'falling':
                reasons.append("Declining form trend")
                severity += 2

            # Check expected vs actual
            exp_data = player.get('expected_vs_actual', {})
            if exp_data.get('goals_performance') == 'under':
                xg = exp_data.get('expected_goals', 0)
                goals = exp_data.get('actual_goals', 0)
                if xg > 1:  # Only flag if significant xG
                    reasons.append(f"Underperforming xG ({goals} vs {xg:.1f} expected)")
                    severity += 1

            # Check minutes
            minutes = player.get('raw_stats', {}).get('minutes', 0)
            gws_played = len(form_data.get('recent_points', []))
            if gws_played > 0:
                mins_per_gw = minutes / max(gws_played, 1)
                if mins_per_gw < 60:
                    reasons.append(f"Limited minutes ({mins_per_gw:.0f}/GW)")
                    severity += 2

            # Check peer comparison
            peer_data = player.get('peer_comparison', {})
            if peer_data.get('overall_rating', 100) < 25:
                reasons.append("Bottom quartile vs peers")
                severity += 2

            # Check for injury/unavailability
            stats = player.get('raw_stats', {})
            if minutes == 0:
                reasons.append("No minutes this season")
                severity += 4

            if reasons:
                underperformers.append({
                    'player_id': player.get('player_id'),
                    'name': player.get('name'),
                    'position': player.get('position'),
                    'team': player.get('team'),
                    'price': player.get('price'),
                    'reasons': reasons,
                    'severity': severity,
                    'current_form': form_avg,
                    'total_points': stats.get('total_points', 0)
                })

        # Sort by severity (most urgent first)
        return sorted(underperformers, key=lambda x: x['severity'], reverse=True)

    def score_replacement(self, candidate: Dict, budget: float,
                          fixtures: List[Dict], current_player: Dict,
                          predicted_points: float = 0.0) -> float:
        """Score a potential replacement player.

        Args:
            candidate: Candidate player stats dictionary.
            budget: Available budget in millions.
            fixtures: Upcoming fixtures for the candidate's team.
            current_player: Current player being replaced.
            predicted_points: RF Model prediction for next GW.

        Returns:
            Weighted score (0-100).
        """
        scores = {}

        # Form score (0-100)
        form = float(candidate.get('form', 0) or 0)
        scores['form'] = min(form * 10, 100)

        # Fixture difficulty score (0-100)
        if fixtures:
            avg_difficulty = np.mean([f.get('difficulty', 3) for f in fixtures[:3]])
            scores['fixtures'] = (5 - avg_difficulty) / 4 * 100  # Invert: easier = higher
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

        return round(total, 1)

    def get_recommendations(self, underperformers: List[Dict],
                            num_recommendations: int = 3) -> List[Dict]:
        """Get transfer recommendations for underperforming players.

        Args:
            underperformers: List of underperforming player dicts.
            num_recommendations: Max replacements to suggest per player.

        Returns:
            List of recommendation dictionaries.
        """
        budget = self.fetcher.get_bank()
        recommendations = []
        
        # Prepare training for top players to ensure model is ready
        # Collect IDs of potential candidates
        all_candidate_ids = []
        for player in underperformers[:5]:
            position = player['position']
            candidates = self.fetcher.get_all_players_by_position(position)
            # Filter reasonably good options to train on
            candidates = candidates[candidates['minutes'] >= 450] # At least 5 games
            all_candidate_ids.extend(candidates['id'].tolist())
            
        # Train model once on these players (unique)
        unique_ids = list(set(all_candidate_ids))
        # Limit to top 200 to save time
        if len(unique_ids) > 200:
            unique_ids = unique_ids[:200]
            
        print(f"Training prediction model on {len(unique_ids)} players...")
        self.predictor.train(unique_ids)
        
        # Predict for all trained players
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
                    cand_dict, total_budget, fixtures, player, pred_points
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
        metrics['confidence'] = self.predictor.get_confidence_level()
        return metrics

    def calculate_expected_value(self, player_id: int, 
                                  num_gws: int = 5) -> Dict:
        """Calculate expected value for a player over multiple gameweeks.
        
        Args:
            player_id: Player ID.
            num_gws: Number of gameweeks to project.
            
        Returns:
            Dict with predictions and cumulative expected value.
        """
        # Ensure model is trained
        if not self.predictor.is_trained:
            all_players = self.fetcher.players_df
            train_ids = all_players[all_players['minutes'] >= 450]['id'].tolist()[:200]
            self.predictor.train(train_ids)
        
        result = self.predictor.predict_multiple_gws([player_id], num_gws)
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
        # Ensure model is trained
        if not self.predictor.is_trained:
            all_players = self.fetcher.players_df
            train_ids = all_players[all_players['minutes'] >= 450]['id'].tolist()[:200]
            self.predictor.train(train_ids)
        
        predictions = self.predictor.predict_multiple_gws(
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
