"""FPL Points Predictor Module

Implements an enhanced Gradient Boosting model to predict player points for upcoming
gameweeks based on historical performance, xG/xA data, and fixture difficulty.
Supports multi-gameweek predictions for transfer planning horizon analysis.

Enhanced version includes:
- xG/xA data from Understat
- Position-specific features (clean sheets for DEF/GKP, goals for attackers)
- Multiple rolling windows (3, 5, 10 games)
- Team strength features
- Gradient Boosting (typically outperforms Random Forest)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from typing import List, Dict, Optional, Tuple

class FPLPointsPredictor:
    """Predicts FPL points using Gradient Boosting Regression.
    
    Enhanced model with xG/xA data, position-specific features,
    and multi-horizon predictions.
    """

    def __init__(self, data_fetcher):
        """Initialize with data fetcher.
        
        Args:
            data_fetcher: FPLDataFetcher instance.
        """
        self.fetcher = data_fetcher
        
        # Use Gradient Boosting with tuned parameters
        self.gb_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.7,
            max_features='sqrt',
            random_state=42
        )
        
        # Also use Random Forest for ensemble
        self.rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_split=15,
            min_samples_leaf=8,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # Primary model (will use average of both)
        self.model = self.gb_model
        self.use_ensemble = True
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Model performance metrics
        self.metrics = {
            'mae': None,
            'rmse': None,
            'r2': None,
            'training_samples': 0,
            'validation_samples': 0
        }
        
        # Cache for understat data
        self._understat_data = None
        self._team_strength = {}
        
        # Streamlined feature set - focus on most predictive features
        self.feature_cols = [
            # Core rolling stats (4-game window) - most predictive
            'avg_points_last_4',
            'avg_minutes_last_4',
            'avg_ict_last_4',
            'avg_bonus_last_4',
            # Expected stats (key for predicting future points)
            'avg_xg_last_4',
            'avg_xa_last_4',
            'avg_xgi_last_4',
            # Position-specific (important for clean sheet/save points)
            'avg_cs_last_4',
            'avg_saves_last_4',
            # Player quality indicators
            'value',
            # Position encoding (different scoring profiles)
            'is_gkp',
            'is_def',
            'is_mid',
            'is_fwd',
            # Fixture info (affects goal involvement and clean sheets)
            'fixture_difficulty',
            'is_home',
            # Team quality (affects overall performance)
            'team_attack_strength',
            'opp_defense_strength',
            # Form momentum
            'form_trend',
            'minutes_trend',
        ]

    def _load_understat_data(self) -> pd.DataFrame:
        """Load Understat xG/xA data if available."""
        if self._understat_data is not None:
            return self._understat_data
            
        understat_path = Path(__file__).parent.parent.parent / 'data' / '2024-25' / 'understat' / 'understat_player.csv'
        if understat_path.exists():
            try:
                self._understat_data = pd.read_csv(understat_path)
                return self._understat_data
            except Exception:
                pass
        return pd.DataFrame()

    def _calculate_team_strength(self):
        """Calculate team attack and defense strength based on recent results."""
        if self._team_strength:
            return
            
        # Get fixtures that have been played
        fixtures = self.fetcher.fixtures_df
        played = fixtures[fixtures['finished'] == True].copy()
        
        if played.empty:
            return
            
        teams = self.fetcher.bootstrap_data.get('teams', [])
        
        for team in teams:
            tid = team['id']
            
            # Home games
            home = played[played['team_h'] == tid]
            # Away games
            away = played[played['team_a'] == tid]
            
            goals_scored = home['team_h_score'].sum() + away['team_a_score'].sum()
            goals_conceded = home['team_a_score'].sum() + away['team_h_score'].sum()
            games = len(home) + len(away)
            
            if games > 0:
                self._team_strength[tid] = {
                    'attack': goals_scored / games,
                    'defense': goals_conceded / games
                }
            else:
                self._team_strength[tid] = {'attack': 1.5, 'defense': 1.5}

    def _get_rolling_stats(self, history_df: pd.DataFrame, target_gw: int, 
                           window: int = 4, position: str = 'MID') -> Optional[Dict]:
        """Calculate rolling stats for the window prior to target_gw.
        
        Enhanced with position-specific features and expected stats.
        """
        # Filter for the window range
        mask = (history_df['round'] >= target_gw - window) & (history_df['round'] < target_gw)
        window_df = history_df[mask]
        
        if len(window_df) < 1:
            return None
        
        # Also get longer window for stability
        mask_10 = (history_df['round'] >= target_gw - 10) & (history_df['round'] < target_gw)
        window_10_df = history_df[mask_10]
        
        # Safe mean calculation
        def safe_mean(series, default=0.0):
            try:
                val = series.mean()
                return val if pd.notna(val) else default
            except:
                return default
        
        def safe_float(series, default=0.0):
            try:
                return float(series.astype(float).mean())
            except:
                return default
        
        stats = {
            # 4-game rolling
            'avg_points_last_4': safe_mean(window_df['total_points']),
            'avg_minutes_last_4': safe_mean(window_df['minutes']),
            'avg_ict_last_4': safe_float(window_df['ict_index']),
            'avg_goals_last_4': safe_mean(window_df['goals_scored']),
            'avg_assists_last_4': safe_mean(window_df['assists']),
            'avg_bonus_last_4': safe_mean(window_df['bonus']),
            'avg_bps_last_4': safe_mean(window_df['bps']),
            # 10-game rolling
            'avg_points_last_10': safe_mean(window_10_df['total_points']) if len(window_10_df) > 0 else safe_mean(window_df['total_points']),
            'avg_minutes_last_10': safe_mean(window_10_df['minutes']) if len(window_10_df) > 0 else safe_mean(window_df['minutes']),
            # Expected stats (from FPL API history)
            'avg_xg_last_4': safe_float(window_df.get('expected_goals', pd.Series([0]))),
            'avg_xa_last_4': safe_float(window_df.get('expected_assists', pd.Series([0]))),
            'avg_xgi_last_4': safe_float(window_df.get('expected_goal_involvements', pd.Series([0]))),
            # Position-specific
            'avg_cs_last_4': safe_mean(window_df['clean_sheets']) if position in ['GKP', 'DEF'] else 0.0,
            'avg_saves_last_4': safe_mean(window_df['saves']) if position == 'GKP' else 0.0,
            'avg_gc_last_4': safe_mean(window_df['goals_conceded']),
        }
        
        # Form trend: compare recent 2 games to previous 2
        if len(window_df) >= 4:
            recent_2 = window_df.tail(2)['total_points'].mean()
            older_2 = window_df.head(2)['total_points'].mean()
            stats['form_trend'] = recent_2 - older_2
        else:
            stats['form_trend'] = 0.0
            
        # Minutes trend (playing time stability)
        if len(window_df) >= 2:
            recent_mins = window_df.tail(2)['minutes'].mean()
            older_mins = window_df.head(max(1, len(window_df)-2))['minutes'].mean()
            stats['minutes_trend'] = (recent_mins - older_mins) / max(older_mins, 1)
        else:
            stats['minutes_trend'] = 0.0
            
        return stats

    def _get_fixture_difficulty(self, player_id: int, gameweek: int) -> Tuple[float, int, int, int]:
        """Get fixture difficulty and home/away status for a specific gameweek.
        
        Returns:
            Tuple of (fdr, is_home, team_id, opponent_id)
        """
        player_stats = self.fetcher.get_player_stats(player_id)
        team_id = player_stats.get('team')
        if not team_id:
            return 3.0, 0, 0, 0
            
        fixtures = self.fetcher.fixtures_df
        
        # Check home games
        home_fix = fixtures[(fixtures['team_h'] == team_id) & (fixtures['event'] == gameweek)]
        if not home_fix.empty:
            opp_id = int(home_fix.iloc[0]['team_a'])
            return float(home_fix.iloc[0]['team_h_difficulty']), 1, team_id, opp_id
            
        # Check away games
        away_fix = fixtures[(fixtures['team_a'] == team_id) & (fixtures['event'] == gameweek)]
        if not away_fix.empty:
            opp_id = int(away_fix.iloc[0]['team_h'])
            return float(away_fix.iloc[0]['team_a_difficulty']), 0, team_id, opp_id
            
        return 3.0, 0, team_id, 0

    def build_training_dataset(self, player_ids: List[int], current_gw: int) -> pd.DataFrame:
        """Build dataset for training the model.
        
        Enhanced with position-specific features and team strength.
        """
        data = []
        
        # Calculate team strengths first
        self._calculate_team_strength()
        
        # Start from GW 5 to allow for window
        start_gw = 5
        
        print(f"Building training data for {len(player_ids)} players (GW{start_gw}-{current_gw})...")
        
        # Get position map
        pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        
        for pid in player_ids:
            history = self.fetcher.get_player_history(pid)
            if history.empty:
                continue
                
            player_stats = self.fetcher.get_player_stats(pid)
            value = float(player_stats.get('now_cost', 0)) / 10.0
            position = pos_map.get(player_stats.get('element_type', 3), 'MID')
            selected_pct = float(player_stats.get('selected_by_percent', 0) or 0)
            
            for gw in range(start_gw, current_gw + 1):
                # Target: Points in this GW
                target_row = history[history['round'] == gw]
                if target_row.empty:
                    continue
                target_points = target_row.iloc[0]['total_points']
                
                # Features: Rolling stats
                rolling_stats = self._get_rolling_stats(history, gw, position=position)
                if not rolling_stats:
                    continue
                    
                # Features: Fixture
                fdr, is_home, team_id, opp_id = self._get_fixture_difficulty(pid, gw)
                
                # Team strength features
                team_str = self._team_strength.get(team_id, {'attack': 1.5, 'defense': 1.5})
                opp_str = self._team_strength.get(opp_id, {'attack': 1.5, 'defense': 1.5})
                
                row = rolling_stats.copy()
                row['value'] = value
                row['selected_pct'] = selected_pct
                row['fixture_difficulty'] = fdr
                row['is_home'] = is_home
                
                # Position encoding
                row['is_gkp'] = 1 if position == 'GKP' else 0
                row['is_def'] = 1 if position == 'DEF' else 0
                row['is_mid'] = 1 if position == 'MID' else 0
                row['is_fwd'] = 1 if position == 'FWD' else 0
                
                # Team/opponent strength
                row['team_attack_strength'] = team_str['attack']
                row['team_defense_strength'] = team_str['defense']
                row['opp_attack_strength'] = opp_str['attack']
                row['opp_defense_strength'] = opp_str['defense']
                
                row['target_points'] = target_points
                row['gameweek'] = gw  # Track GW for chronological split
                
                data.append(row)
                
        return pd.DataFrame(data)
    
    def _chronological_split(self, df: pd.DataFrame, 
                            validation_split: float = 0.2) -> tuple:
        """Split dataset by gameweek with no overlap.
        
        The last N gameweeks (based on validation_split) go entirely to validation.
        Guarantees no GW appears in both train and val sets.
        """
        if 'gameweek' not in df.columns:
            # Fall back to row-based split
            split_idx = int(len(df) * (1 - validation_split))
            return df.iloc[:split_idx], df.iloc[split_idx:]
        
        unique_gws = sorted(df['gameweek'].unique())
        n_gws = len(unique_gws)
        
        if n_gws < 2:
            return df, pd.DataFrame(columns=df.columns)
        
        # Put at least 1 GW in validation, up to validation_split fraction
        n_val_gws = max(1, int(n_gws * validation_split))
        
        train_gws = set(unique_gws[:-n_val_gws])
        val_gws = set(unique_gws[-n_val_gws:])
        
        train_df = df[df['gameweek'].isin(train_gws)].copy()
        val_df = df[df['gameweek'].isin(val_gws)].copy()
        
        return train_df, val_df

    def train(self, player_ids: List[int], validation_split: float = 0.2):
        """Train the Gradient Boosting model with validation metrics.
        
        Uses chronological gameweek-based split to avoid data leakage.
        """
        current_gw = self.fetcher.get_current_gameweek()
        if current_gw < 5:
            print("Not enough gameweeks to train model (needs >= 5).")
            return

        df = self.build_training_dataset(player_ids, current_gw)
        
        if df.empty:
            print("No training data available.")
            return
        
        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        # Chronological split by gameweek (no overlap)
        train_df, val_df = self._chronological_split(df, validation_split)
        
        if 'gameweek' in train_df.columns and len(train_df) > 0:
            train_gws = sorted(train_df['gameweek'].unique())
            val_gws = sorted(val_df['gameweek'].unique()) if len(val_df) > 0 else []
            print(f"Training Ensemble (GB + RF) on {len(train_df)} samples (GW{train_gws[0]}-{train_gws[-1]}), "
                  f"validating on {len(val_df)} samples (GW{val_gws[0] if val_gws else 'none'}-{val_gws[-1] if val_gws else 'none'})...")
        else:
            print(f"Training Ensemble (GB + RF) on {len(train_df)} samples, validating on {len(val_df)}...")
        
        X_train = train_df[self.feature_cols].fillna(0)
        y_train = train_df['target_points']
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train both models for ensemble
        self.gb_model.fit(X_train_scaled, y_train)
        self.rf_model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Calculate validation metrics using ensemble
        if len(val_df) > 0:
            X_val = val_df[self.feature_cols].fillna(0)
            y_val = val_df['target_points']
            X_val_scaled = self.scaler.transform(X_val)
            
            # Ensemble prediction (average of GB and RF)
            if self.use_ensemble:
                y_pred_gb = self.gb_model.predict(X_val_scaled)
                y_pred_rf = self.rf_model.predict(X_val_scaled)
                y_pred = (y_pred_gb + y_pred_rf) / 2
            else:
                y_pred = self.gb_model.predict(X_val_scaled)
            
            self.metrics['mae'] = round(mean_absolute_error(y_val, y_pred), 2)
            self.metrics['rmse'] = round(np.sqrt(mean_squared_error(y_val, y_pred)), 2)
            self.metrics['r2'] = round(r2_score(y_val, y_pred), 3)
            self.metrics['training_samples'] = len(train_df)
            self.metrics['validation_samples'] = len(val_df)
            
            print(f"Model trained. Validation metrics - MAE: {self.metrics['mae']}, "
                  f"RMSE: {self.metrics['rmse']}, R²: {self.metrics['r2']}")
        else:
            y_pred = self.model.predict(X_train_scaled)
            self.metrics['mae'] = round(mean_absolute_error(y_train, y_pred), 2)
            self.metrics['rmse'] = round(np.sqrt(mean_squared_error(y_train, y_pred)), 2)
            self.metrics['r2'] = round(r2_score(y_train, y_pred), 3)
            self.metrics['training_samples'] = len(train_df)
            print(f"Model trained (training set only). R²: {self.metrics['r2']}")

    def predict(self, player_ids: List[int]) -> Dict[int, float]:
        """Predict points for next gameweek for given players."""
        if not self.is_trained:
            print("Model not trained yet.")
            return {}
            
        current_gw = self.fetcher.get_current_gameweek()
        next_gw = current_gw + 1
        
        self._calculate_team_strength()
        pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        
        predictions = {}
        prediction_data = []
        pids_map = []
        
        for pid in player_ids:
            history = self.fetcher.get_player_history(pid)
            if history.empty:
                predictions[pid] = 0.0
                continue
                
            player_stats = self.fetcher.get_player_stats(pid)
            position = pos_map.get(player_stats.get('element_type', 3), 'MID')
            
            rolling_stats = self._get_rolling_stats(history, next_gw, position=position)
            if not rolling_stats:
                predictions[pid] = 0.0
                continue
                
            value = float(player_stats.get('now_cost', 0)) / 10.0
            selected_pct = float(player_stats.get('selected_by_percent', 0) or 0)
            fdr, is_home, team_id, opp_id = self._get_fixture_difficulty(pid, next_gw)
            
            team_str = self._team_strength.get(team_id, {'attack': 1.5, 'defense': 1.5})
            opp_str = self._team_strength.get(opp_id, {'attack': 1.5, 'defense': 1.5})
            
            row = rolling_stats.copy()
            row['value'] = value
            row['selected_pct'] = selected_pct
            row['fixture_difficulty'] = fdr
            row['is_home'] = is_home
            row['is_gkp'] = 1 if position == 'GKP' else 0
            row['is_def'] = 1 if position == 'DEF' else 0
            row['is_mid'] = 1 if position == 'MID' else 0
            row['is_fwd'] = 1 if position == 'FWD' else 0
            row['team_attack_strength'] = team_str['attack']
            row['team_defense_strength'] = team_str['defense']
            row['opp_attack_strength'] = opp_str['attack']
            row['opp_defense_strength'] = opp_str['defense']
            
            prediction_data.append(row)
            pids_map.append(pid)
            
        if not prediction_data:
            return predictions
            
        X_pred = pd.DataFrame(prediction_data)
        
        # Ensure all columns exist
        for col in self.feature_cols:
            if col not in X_pred.columns:
                X_pred[col] = 0.0
                
        X_pred = X_pred[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X_pred)
        
        # Ensemble prediction
        if self.use_ensemble:
            preds_gb = self.gb_model.predict(X_scaled)
            preds_rf = self.rf_model.predict(X_scaled)
            preds = (preds_gb + preds_rf) / 2
        else:
            preds = self.gb_model.predict(X_scaled)
        
        for i, pid in enumerate(pids_map):
            predictions[pid] = round(max(0, preds[i]), 2)
            
        return predictions

    def predict_multiple_gws(self, player_ids: List[int], num_gws: int = 5) -> Dict[int, Dict]:
        """Predict points for multiple upcoming gameweeks.
        
        For each player, predicts expected points for the next N gameweeks,
        along with cumulative totals and confidence intervals.
        """
        if not self.is_trained:
            print("Model not trained yet.")
            return {}
            
        current_gw = self.fetcher.get_current_gameweek()
        self._calculate_team_strength()
        pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        
        results = {}
        
        for pid in player_ids:
            history = self.fetcher.get_player_history(pid)
            player_stats = self.fetcher.get_player_stats(pid)
            value = float(player_stats.get('now_cost', 0)) / 10.0
            position = pos_map.get(player_stats.get('element_type', 3), 'MID')
            selected_pct = float(player_stats.get('selected_by_percent', 0) or 0)
            
            gw_predictions = []
            
            for gw_offset in range(1, num_gws + 1):
                target_gw = current_gw + gw_offset
                
                if history.empty:
                    gw_predictions.append(0.0)
                    continue
                
                rolling_stats = self._get_rolling_stats(history, current_gw + 1, position=position)
                if not rolling_stats:
                    gw_predictions.append(0.0)
                    continue
                
                fdr, is_home, team_id, opp_id = self._get_fixture_difficulty(pid, target_gw)
                
                team_str = self._team_strength.get(team_id, {'attack': 1.5, 'defense': 1.5})
                opp_str = self._team_strength.get(opp_id, {'attack': 1.5, 'defense': 1.5})
                
                row = rolling_stats.copy()
                row['value'] = value
                row['selected_pct'] = selected_pct
                row['fixture_difficulty'] = fdr
                row['is_home'] = is_home
                row['is_gkp'] = 1 if position == 'GKP' else 0
                row['is_def'] = 1 if position == 'DEF' else 0
                row['is_mid'] = 1 if position == 'MID' else 0
                row['is_fwd'] = 1 if position == 'FWD' else 0
                row['team_attack_strength'] = team_str['attack']
                row['team_defense_strength'] = team_str['defense']
                row['opp_attack_strength'] = opp_str['attack']
                row['opp_defense_strength'] = opp_str['defense']
                
                X_pred = pd.DataFrame([row])
                for col in self.feature_cols:
                    if col not in X_pred.columns:
                        X_pred[col] = 0.0
                X_pred = X_pred[self.feature_cols].fillna(0)
                
                X_scaled = self.scaler.transform(X_pred)
                
                # Ensemble prediction
                if self.use_ensemble:
                    pred_gb = self.gb_model.predict(X_scaled)[0]
                    pred_rf = self.rf_model.predict(X_scaled)[0]
                    pred = (pred_gb + pred_rf) / 2
                else:
                    pred = self.gb_model.predict(X_scaled)[0]
                    
                gw_predictions.append(round(max(0, pred), 2))
            
            # Calculate confidence based on player's consistency
            if len(history) >= 4:
                recent_pts = history.tail(4)['total_points'].std()
                if recent_pts < 2:
                    confidence = 'high'
                elif recent_pts < 4:
                    confidence = 'medium'
                else:
                    confidence = 'low'
            else:
                confidence = 'low'
            
            results[pid] = {
                'predictions': gw_predictions,
                'cumulative': round(sum(gw_predictions), 2),
                'confidence': confidence,
                'std_dev': round(np.std(gw_predictions) if gw_predictions else 0, 2),
                'avg_per_gw': round(sum(gw_predictions) / max(len(gw_predictions), 1), 2)
            }
        
        return results

    def get_model_metrics(self) -> Dict:
        """Get model performance metrics."""
        return self.metrics.copy()

    def get_confidence_level(self) -> str:
        """Determine overall model confidence level based on metrics."""
        if self.metrics['r2'] is None:
            return 'unknown'
        
        r2 = self.metrics['r2']
        mae = self.metrics['mae'] or 999
        
        if r2 > 0.3 and mae < 2.0:
            return 'high'
        elif r2 > 0.15 or mae < 2.5:
            return 'medium'
        else:
            return 'low'

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        if not self.is_trained:
            return {}
        
        importances = self.model.feature_importances_
        return {
            col: round(imp, 4) 
            for col, imp in sorted(zip(self.feature_cols, importances), 
                                   key=lambda x: x[1], reverse=True)
        }
