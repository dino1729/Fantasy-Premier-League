"""Points Predictor Module

Implements a Random Forest model to predict FPL player points for future gameweeks.
Based on: https://github.com/francescobarbara/FPL-point-predictor-via-random-forests
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple

class PointsPredictor:
    """Predicts player points using Random Forest Regression."""

    def __init__(self, data_fetcher, model_path: Optional[Path] = None):
        """Initialize the predictor.

        Args:
            data_fetcher: FPLDataFetcher instance.
            model_path: Path to saved model file (optional).
        """
        self.fetcher = data_fetcher
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'value', 'was_home', 'minutes', 'total_points', 
            'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
            'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards',
            'red_cards', 'saves', 'bonus', 'bps', 'influence', 'creativity',
            'threat', 'ict_index', 'selected', 'transfers_in', 'transfers_out'
        ]
        # We need to add "lagged" features for the last 4 gameweeks
        self.lagged_features = []
        for i in range(1, 5):
            self.lagged_features.extend([f"{col}_lag_{i}" for col in self.feature_columns])
        
        self.all_features = ['position_code'] + self.lagged_features # Simplified feature set for now

        if model_path and model_path.exists():
            self.load_model(model_path)

    def _prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare historical data for training.
        
        This constructs a dataset where each row is a (player, gameweek) pair,
        and features are statistics from the previous 4 gameweeks.
        """
        # Get all players
        players = self.fetcher.players_df
        
        training_rows = []
        training_targets = []
        
        # We'll use the current season's history as training data
        # For a production system, we'd want historical season data too
        
        # Iterate through players
        for _, player in players.iterrows():
            pid = player['id']
            pos_code = player['element_type']
            
            # Get GW history
            history = self.fetcher.get_player_history(pid)
            if history.empty:
                continue
                
            # Sort by GW
            history = history.sort_values('round')
            
            # Create lagged features
            # We need at least 5 gameweeks to have 4 previous + 1 target
            if len(history) < 5:
                continue
                
            # Convert relevant columns to numeric
            cols_to_numeric = [
                'value', 'minutes', 'total_points', 'goals_scored', 'assists', 
                'clean_sheets', 'goals_conceded', 'own_goals', 'penalties_saved', 
                'penalties_missed', 'yellow_cards', 'red_cards', 'saves', 
                'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index',
                'selected', 'transfers_in', 'transfers_out'
            ]
            
            for col in cols_to_numeric:
                if col in history.columns:
                    history[col] = pd.to_numeric(history[col], errors='coerce').fillna(0)
            
            history['was_home'] = history['was_home'].astype(int)
            
            # Iterate through gameweeks starting from 5th available
            for i in range(4, len(history)):
                target_row = history.iloc[i]
                target_points = target_row['total_points']
                
                # Get previous 4 rows
                prev_4 = history.iloc[i-4:i]
                
                features = {'position_code': pos_code}
                
                for lag in range(1, 5):
                    # lag 1 is i-1, lag 4 is i-4
                    row_idx = 4 - lag # 0 to 3 in prev_4
                    # Actually, lag 1 is the most recent (i-1)
                    # so if prev_4 is [GW1, GW2, GW3, GW4] and target is GW5
                    # lag 1 is GW4 (index 3), lag 4 is GW1 (index 0)
                    lag_row = prev_4.iloc[4-lag]
                    
                    for col in self.feature_columns:
                        if col in lag_row:
                            features[f"{col}_lag_{lag}"] = lag_row[col]
                        else:
                            features[f"{col}_lag_{lag}"] = 0
                            
                training_rows.append(features)
                training_targets.append(target_points)
                
        return pd.DataFrame(training_rows), pd.Series(training_targets)

    def train_model(self):
        """Train the Random Forest model."""
        print("Preparing training data...")
        X, y = self._prepare_training_data()
        
        if X.empty:
            print("Insufficient data for training.")
            return
            
        print(f"Training model on {len(X)} samples with {len(X.columns)} features...")
        
        # Align features
        # Ensure X has all expected columns, fill missing with 0
        for col in self.all_features:
            if col not in X.columns:
                X[col] = 0
        X = X[self.all_features]
        
        # Simple imputation/filling
        X = X.fillna(0)
        
        # Train Random Forest
        # Parameters from the referenced repo: n_estimators=1000, max_features=5 (or optimized)
        # We'll use a balanced config
        self.model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            n_jobs=-1, 
            random_state=42
        )
        self.model.fit(X, y)
        print("Model training complete.")

    def predict_next_gw(self, players_to_predict: List[Dict]) -> Dict[int, float]:
        """Predict points for the upcoming gameweek for a list of players.
        
        Args:
            players_to_predict: List of player dicts (must contain 'id').
            
        Returns:
            Dictionary mapping player_id to predicted points.
        """
        if self.model is None:
            self.train_model()
            
        if self.model is None:
            return {}
            
        predictions = {}
        prediction_rows = []
        player_ids = []
        
        for player in players_to_predict:
            pid = player['id']
            # Get stats needed for feature construction
            # We need stats from the LAST 4 gameweeks
            history = self.fetcher.get_player_history(pid)
            
            if history.empty or len(history) < 4:
                # Can't predict without history, assume avg or 0
                predictions[pid] = 0.0
                continue
                
            history = history.sort_values('round')
            last_4 = history.tail(4)
            
            # Construct features
            # Get position code from player dict if available, else fetch
            pos_code = player.get('element_type')
            if not pos_code:
                p_stats = self.fetcher.get_player_stats(pid)
                pos_code = p_stats.get('element_type', 0)
                
            features = {'position_code': pos_code}
            
            # Convert numeric columns in history slice if needed
            cols_to_numeric = [
                'value', 'minutes', 'total_points', 'goals_scored', 'assists', 
                'clean_sheets', 'goals_conceded', 'own_goals', 'penalties_saved', 
                'penalties_missed', 'yellow_cards', 'red_cards', 'saves', 
                'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index',
                'selected', 'transfers_in', 'transfers_out'
            ]
            for col in cols_to_numeric:
                if col in last_4.columns:
                    last_4[col] = pd.to_numeric(last_4[col], errors='coerce').fillna(0)
            
            last_4['was_home'] = last_4['was_home'].astype(int)

            for lag in range(1, 5):
                # lag 1 is the most recent (last row of last_4)
                # last_4 has indices 0,1,2,3 relative to itself
                if len(last_4) >= lag:
                    row_idx = len(last_4) - lag
                    lag_row = last_4.iloc[row_idx]
                    
                    for col in self.feature_columns:
                        if col in lag_row:
                            features[f"{col}_lag_{lag}"] = lag_row[col]
                        else:
                            features[f"{col}_lag_{lag}"] = 0
                else:
                     # Pad with 0 if somehow < 4 rows (though check above handles it)
                     for col in self.feature_columns:
                        features[f"{col}_lag_{lag}"] = 0

            prediction_rows.append(features)
            player_ids.append(pid)
            
        if not prediction_rows:
            return predictions
            
        X_pred = pd.DataFrame(prediction_rows)
        
        # Ensure columns match training
        for col in self.all_features:
            if col not in X_pred.columns:
                X_pred[col] = 0
        X_pred = X_pred[self.all_features]
        X_pred = X_pred.fillna(0)
        
        preds = self.model.predict(X_pred)
        
        for pid, val in zip(player_ids, preds):
            predictions[pid] = round(float(val), 1)
            
        return predictions

    def save_model(self, path: Path):
        """Save trained model to file."""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, path: Path):
        """Load trained model from file."""
        try:
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            print(f"Failed to load model: {e}")


