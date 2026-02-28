"""Player Analyzer Module

Performs deep analysis on individual players including form trends,
ICT breakdown, expected vs actual performance, and peer comparisons.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats as scipy_stats


class PlayerAnalyzer:
    """Analyzes individual player performance and compares to peers."""

    def __init__(self, data_fetcher):
        """Initialize with a data fetcher instance.

        Args:
            data_fetcher: FPLDataFetcher instance for data access.
        """
        self.fetcher = data_fetcher

    def analyze_form_trend(self, player_id: int, window: int = 5) -> Dict:
        """Analyze recent form trend for a player.

        Args:
            player_id: FPL element ID.
            window: Number of recent gameweeks to analyze.

        Returns:
            Dictionary with form metrics and trend analysis.
        """
        history = self.fetcher.get_player_history(player_id)

        if history.empty:
            return {
                'recent_points': [],
                'average': 0,
                'trend': 'unknown',
                'volatility': 0,
                'form_rating': 0
            }

        # Get recent gameweeks
        recent = history.tail(window)
        points = recent['total_points'].tolist()

        if len(points) < 2:
            return {
                'recent_points': points,
                'average': np.mean(points) if points else 0,
                'trend': 'insufficient_data',
                'volatility': 0,
                'form_rating': np.mean(points) if points else 0
            }

        # Calculate trend using linear regression
        x = np.arange(len(points))
        slope, _, r_value, _, _ = scipy_stats.linregress(x, points)

        # Determine trend direction
        if slope > 0.5:
            trend = 'rising'
        elif slope < -0.5:
            trend = 'falling'
        else:
            trend = 'stable'

        # Calculate volatility (coefficient of variation)
        mean_pts = np.mean(points)
        std_pts = np.std(points)
        volatility = (std_pts / mean_pts * 100) if mean_pts > 0 else 0

        return {
            'recent_points': points,
            'average': round(mean_pts, 2),
            'trend': trend,
            'trend_slope': round(slope, 3),
            'trend_strength': round(abs(r_value), 3),
            'volatility': round(volatility, 1),
            'form_rating': round(mean_pts, 2),
            'total_recent': sum(points),
            'best_gw': max(points),
            'worst_gw': min(points)
        }

    def analyze_ict_breakdown(self, player_stats: Dict) -> Dict:
        """Analyze ICT (Influence, Creativity, Threat) breakdown.

        Args:
            player_stats: Player stats dictionary.

        Returns:
            ICT analysis with percentages and rankings.
        """
        influence = float(player_stats.get('influence', 0) or 0)
        creativity = float(player_stats.get('creativity', 0) or 0)
        threat = float(player_stats.get('threat', 0) or 0)
        ict_total = float(player_stats.get('ict_index', 0) or 0)

        # Calculate percentages of ICT index
        total_raw = influence + creativity + threat
        if total_raw > 0:
            influence_pct = influence / total_raw * 100
            creativity_pct = creativity / total_raw * 100
            threat_pct = threat / total_raw * 100
        else:
            influence_pct = creativity_pct = threat_pct = 33.3

        return {
            'influence': round(influence, 1),
            'creativity': round(creativity, 1),
            'threat': round(threat, 1),
            'ict_index': round(ict_total, 1),
            'influence_pct': round(influence_pct, 1),
            'creativity_pct': round(creativity_pct, 1),
            'threat_pct': round(threat_pct, 1),
            'influence_rank': player_stats.get('influence_rank', 0),
            'creativity_rank': player_stats.get('creativity_rank', 0),
            'threat_rank': player_stats.get('threat_rank', 0),
            'ict_rank': player_stats.get('ict_index_rank', 0),
            'dominant_trait': self._get_dominant_trait(influence_pct, creativity_pct, threat_pct)
        }

    def _get_dominant_trait(self, inf: float, cre: float, thr: float) -> str:
        """Determine the dominant ICT trait."""
        max_val = max(inf, cre, thr)
        if max_val == inf:
            return 'Influential'
        elif max_val == cre:
            return 'Creative'
        else:
            return 'Threatening'

    def calculate_expected_vs_actual(self, player_stats: Dict) -> Dict:
        """Compare expected metrics (xG, xA) vs actual.

        Args:
            player_stats: Player stats dictionary.

        Returns:
            Comparison of expected vs actual with over/underperformance.
        """
        xg = float(player_stats.get('expected_goals', 0) or 0)
        xa = float(player_stats.get('expected_assists', 0) or 0)
        xgi = float(player_stats.get('expected_goal_involvements', 0) or 0)

        goals = int(player_stats.get('goals_scored', 0) or 0)
        assists = int(player_stats.get('assists', 0) or 0)
        gi = goals + assists

        # Calculate over/underperformance
        goals_diff = goals - xg
        assists_diff = assists - xa
        gi_diff = gi - xgi

        # Performance ratings
        goals_perf = 'over' if goals_diff > 0.5 else ('under' if goals_diff < -0.5 else 'inline')
        assists_perf = 'over' if assists_diff > 0.5 else ('under' if assists_diff < -0.5 else 'inline')

        return {
            'expected_goals': round(xg, 2),
            'actual_goals': goals,
            'goals_diff': round(goals_diff, 2),
            'goals_performance': goals_perf,
            'expected_assists': round(xa, 2),
            'actual_assists': assists,
            'assists_diff': round(assists_diff, 2),
            'assists_performance': assists_perf,
            'expected_gi': round(xgi, 2),
            'actual_gi': gi,
            'gi_diff': round(gi_diff, 2),
            'total_points': int(player_stats.get('total_points', 0) or 0),
            'points_per_game': round(float(player_stats.get('points_per_game', 0) or 0), 2),
            'minutes': int(player_stats.get('minutes', 0) or 0)
        }

    def compare_to_peers(self, player_stats: Dict, position: str) -> Dict:
        """Compare player to position peers.

        Args:
            player_stats: Player stats dictionary.
            position: Position code (GKP, DEF, MID, FWD).

        Returns:
            Percentile rankings against peers.
        """
        peers = self.fetcher.get_position_peers(position, min_minutes=90)

        if peers.empty:
            return {'percentiles': {}, 'peer_count': 0}

        metrics = ['total_points', 'form', 'ict_index', 'points_per_game',
                   'expected_goals', 'expected_assists', 'bps']

        percentiles = {}
        for metric in metrics:
            if metric in peers.columns:
                try:
                    player_val = float(player_stats.get(metric, 0) or 0)
                    # Convert to numeric, coercing errors to NaN
                    peer_vals = pd.to_numeric(peers[metric], errors='coerce').dropna().values
                    if len(peer_vals) > 0:
                        pct = scipy_stats.percentileofscore(peer_vals, player_val)
                        percentiles[metric] = round(pct, 1)
                except (ValueError, TypeError):
                    continue

        # Calculate overall rating (average of key percentiles)
        key_metrics = ['total_points', 'form', 'ict_index']
        key_pcts = [percentiles.get(m, 50) for m in key_metrics]
        overall_rating = np.mean(key_pcts)

        return {
            'percentiles': percentiles,
            'peer_count': len(peers),
            'overall_rating': round(overall_rating, 1),
            'position': position
        }

    def calculate_contribution_score(self, player_id: int, squad_points: int) -> Dict:
        """Calculate player's contribution to squad.

        Args:
            player_id: FPL element ID.
            squad_points: Total squad points for the period.

        Returns:
            Contribution metrics.
        """
        stats = self.fetcher.get_player_stats(player_id)
        player_points = int(stats.get('total_points', 0) or 0)

        contribution_pct = (player_points / squad_points * 100) if squad_points > 0 else 0

        return {
            'player_points': player_points,
            'squad_points': squad_points,
            'contribution_pct': round(contribution_pct, 1)
        }

    def generate_player_summary(self, player_id: int, position: str) -> Dict:
        """Generate comprehensive analysis for a player.

        Args:
            player_id: FPL element ID.
            position: Position code.

        Returns:
            Complete player analysis dictionary.
        """
        stats = self.fetcher.get_player_stats(player_id)

        return {
            'player_id': player_id,
            'name': stats.get('web_name', 'Unknown'),
            'full_name': f"{stats.get('first_name', '')} {stats.get('second_name', '')}".strip(),
            'position': position,
            'team': self._get_team_short_name(stats.get('team', 0)),
            'price': round(float(stats.get('now_cost', 0) or 0) / 10, 1),
            'ownership': round(float(stats.get('selected_by_percent', 0) or 0), 1),
            'form_analysis': self.analyze_form_trend(player_id),
            'ict_analysis': self.analyze_ict_breakdown(stats),
            'expected_vs_actual': self.calculate_expected_vs_actual(stats),
            'peer_comparison': self.compare_to_peers(stats, position),
            'raw_stats': {
                'team': int(stats.get('team', 0) or 0),
                'minutes': int(stats.get('minutes', 0) or 0),
                'total_points': int(stats.get('total_points', 0) or 0),
                'goals': int(stats.get('goals_scored', 0) or 0),
                'assists': int(stats.get('assists', 0) or 0),
                'clean_sheets': int(stats.get('clean_sheets', 0) or 0),
                'saves': int(stats.get('saves', 0) or 0),
                'goals_conceded': int(stats.get('goals_conceded', 0) or 0),
                'bonus': int(stats.get('bonus', 0) or 0),
                'bps': int(stats.get('bps', 0) or 0)
            }
        }

    def _get_team_short_name(self, team_id: int) -> str:
        """Get team short name from ID."""
        return self.fetcher._get_team_name(team_id)

    def get_form_classification(self, form: float) -> str:
        """Classify form rating into categories."""
        if form >= 7:
            return 'Excellent'
        elif form >= 5:
            return 'Good'
        elif form >= 3:
            return 'Average'
        elif form >= 1:
            return 'Poor'
        else:
            return 'Very Poor'

    def identify_key_strengths(self, analysis: Dict) -> List[str]:
        """Identify key strengths from player analysis."""
        strengths = []

        # Check form
        form = analysis.get('form_analysis', {})
        if form.get('trend') == 'rising':
            strengths.append('Form is trending upward')
        if form.get('average', 0) >= 5:
            strengths.append('Consistent point scorer')

        # Check ICT
        ict = analysis.get('ict_analysis', {})
        if ict.get('ict_rank', 999) <= 20:
            strengths.append(f"Top 20 ICT ranking ({ict.get('dominant_trait')})")

        # Check expected metrics
        exp = analysis.get('expected_vs_actual', {})
        if exp.get('goals_performance') == 'over':
            strengths.append('Outperforming xG')
        if exp.get('assists_performance') == 'over':
            strengths.append('Outperforming xA')

        # Check peer comparison
        peer = analysis.get('peer_comparison', {})
        if peer.get('overall_rating', 0) >= 75:
            strengths.append('Top quartile among position peers')

        return strengths[:3]  # Return top 3 strengths

    def identify_concerns(self, analysis: Dict) -> List[str]:
        """Identify areas of concern from player analysis."""
        concerns = []

        # Check form
        form = analysis.get('form_analysis', {})
        if form.get('trend') == 'falling':
            concerns.append('Form is declining')
        if form.get('volatility', 0) > 100:
            concerns.append('Highly inconsistent returns')
        if form.get('average', 0) < 3:
            concerns.append('Low recent points average')

        # Check expected metrics
        exp = analysis.get('expected_vs_actual', {})
        if exp.get('goals_performance') == 'under' and exp.get('expected_goals', 0) > 1:
            concerns.append('Underperforming xG (regression risk)')
        if exp.get('minutes', 0) < 500:
            concerns.append('Limited minutes played')

        # Check peer comparison
        peer = analysis.get('peer_comparison', {})
        if peer.get('overall_rating', 100) < 25:
            concerns.append('Bottom quartile among position peers')

        return concerns[:3]  # Return top 3 concerns
