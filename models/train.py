"""Model Training Module

Trains position-specific ML models for FPL point predictions.

Architecture:
- Separate models for GKP, DEF, MID, FWD (different scoring patterns)
- XGBoost as primary model (captures non-linear patterns)
- Ridge Regression as stabilizer (prevents overfitting)
- Ensemble: Final_xP = 0.7 * XGBoost + 0.3 * Ridge

Key: Uses TimeSeriesSplit to avoid training on future data.
"""

import json
import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try importing XGBoost, fall back to GradientBoosting
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models' / 'artifacts'
DATA_PARQUET = PROJECT_ROOT / 'data' / 'parquet'

# Ensure artifacts directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class PositionModel:
    """ML model for a specific position.
    
    Combines XGBoost (or GradientBoosting) with Ridge Regression
    in a weighted ensemble for robust predictions.
    """
    
    def __init__(self, position: str, 
                 xgb_weight: float = 0.7,
                 ridge_weight: float = 0.3):
        """Initialize position-specific model.
        
        Args:
            position: One of 'GKP', 'DEF', 'MID', 'FWD'.
            xgb_weight: Weight for XGBoost predictions (default 0.7).
            ridge_weight: Weight for Ridge predictions (default 0.3).
        """
        self.position = position
        self.xgb_weight = xgb_weight
        self.ridge_weight = ridge_weight
        
        # XGBoost with tuned hyperparameters
        if XGBOOST_AVAILABLE:
            self.xgb_model = XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        else:
            self.xgb_model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42
            )
        
        # Ridge Regression (stabilizer)
        self.ridge_model = Ridge(alpha=1.0, random_state=42)
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        # Feature columns used
        self.feature_cols = []
        
        # Training metrics
        self.metrics = {
            'mae': None,
            'rmse': None,
            'r2': None,
            'training_samples': 0,
            'validation_samples': 0,
            'feature_importance': {}
        }
        
        self.is_trained = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None):
        """Train the ensemble model.
        
        Args:
            X: Feature DataFrame.
            y: Target Series (actual points).
            validation_data: Optional (X_val, y_val) tuple for metrics.
        """
        self.feature_cols = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X.fillna(0))
        
        # Train XGBoost
        self.xgb_model.fit(X_scaled, y)
        
        # Train Ridge
        self.ridge_model.fit(X_scaled, y)
        
        self.is_trained = True
        self.metrics['training_samples'] = len(X)
        
        # Calculate validation metrics if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            self._calculate_metrics(X_val, y_val)
        
        # Store feature importance
        if XGBOOST_AVAILABLE and hasattr(self.xgb_model, 'feature_importances_'):
            importances = self.xgb_model.feature_importances_
            self.metrics['feature_importance'] = {
                col: float(imp) 
                for col, imp in sorted(
                    zip(self.feature_cols, importances),
                    key=lambda x: x[1], reverse=True
                )[:15]  # Top 15 features
            }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions.
        
        Returns:
            Array of predicted points.
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Ensure columns match training
        X_aligned = X.reindex(columns=self.feature_cols, fill_value=0)
        X_scaled = self.scaler.transform(X_aligned.fillna(0))
        
        # Get predictions from both models
        pred_xgb = self.xgb_model.predict(X_scaled)
        pred_ridge = self.ridge_model.predict(X_scaled)
        
        # Ensemble weighted average
        predictions = (self.xgb_weight * pred_xgb + 
                      self.ridge_weight * pred_ridge)
        
        # Ensure non-negative predictions
        return np.maximum(predictions, 0)
    
    def _calculate_metrics(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Calculate validation metrics."""
        y_pred = self.predict(X_val)
        
        self.metrics['mae'] = round(mean_absolute_error(y_val, y_pred), 3)
        self.metrics['rmse'] = round(np.sqrt(mean_squared_error(y_val, y_pred)), 3)
        self.metrics['r2'] = round(r2_score(y_val, y_pred), 4)
        self.metrics['validation_samples'] = len(y_val)
    
    def save(self, filepath: Optional[Path] = None) -> Path:
        """Save model to disk."""
        if filepath is None:
            filepath = MODELS_DIR / f'model_{self.position}.pkl'
        
        model_data = {
            'position': self.position,
            'xgb_model': self.xgb_model,
            'ridge_model': self.ridge_model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'metrics': self.metrics,
            'xgb_weight': self.xgb_weight,
            'ridge_weight': self.ridge_weight,
            'is_trained': self.is_trained,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved {self.position} model to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'PositionModel':
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            position=model_data['position'],
            xgb_weight=model_data.get('xgb_weight', 0.7),
            ridge_weight=model_data.get('ridge_weight', 0.3)
        )
        
        model.xgb_model = model_data['xgb_model']
        model.ridge_model = model_data['ridge_model']
        model.scaler = model_data['scaler']
        model.feature_cols = model_data['feature_cols']
        model.metrics = model_data['metrics']
        model.is_trained = model_data['is_trained']
        
        return model


class AvailabilityPredictor:
    """Predicts probability of playing 60+ minutes.
    
    Used to adjust raw xP by availability:
    Final_xP = Raw_xP * (Predicted_Probability / 100)
    """
    
    def __init__(self):
        """Initialize availability predictor."""
        from sklearn.linear_model import LogisticRegression
        
        self.model = LogisticRegression(
            C=1.0, 
            max_iter=1000, 
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_cols = [
            'minutes_rolling_3',
            'minutes_rolling_6',
            'minutes_trend',
            'chance_playing',  # FPL API flag (0, 25, 50, 75, 100)
        ]
        self.is_trained = False
    
    def fit(self, X: pd.DataFrame, minutes: pd.Series):
        """Train on historical data.
        
        Args:
            X: Features including minutes history.
            minutes: Actual minutes played (target will be >= 60).
        """
        # Target: played 60+ minutes
        y = (minutes >= 60).astype(int)
        
        # Select and prepare features
        X_subset = X[self.feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X_subset)
        
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of playing 60+ minutes."""
        if not self.is_trained:
            # Return FPL's chance_playing if model not trained
            if 'chance_playing' in X.columns:
                return X['chance_playing'].fillna(100).values / 100.0
            return np.ones(len(X))
        
        X_subset = X.reindex(columns=self.feature_cols, fill_value=0)
        X_subset = X_subset.fillna(0)
        
        # Handle case where chance_playing is missing
        if 'chance_playing' not in X_subset.columns:
            X_subset['chance_playing'] = 100
        
        X_scaled = self.scaler.transform(X_subset)
        
        # Return probability of class 1 (plays 60+ minutes)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def save(self, filepath: Optional[Path] = None) -> Path:
        """Save model to disk."""
        if filepath is None:
            filepath = MODELS_DIR / 'availability_model.pkl'
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_cols': self.feature_cols,
                'is_trained': self.is_trained
            }, f)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'AvailabilityPredictor':
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        predictor = cls()
        predictor.model = data['model']
        predictor.scaler = data['scaler']
        predictor.feature_cols = data['feature_cols']
        predictor.is_trained = data['is_trained']
        
        return predictor


class ModelTrainer:
    """Orchestrates training of all position models.
    
    Uses proper time-series cross-validation to prevent data leakage.
    """
    
    POSITIONS = ['GKP', 'DEF', 'MID', 'FWD']
    
    def __init__(self, feature_engineer=None):
        """Initialize trainer.
        
        Args:
            feature_engineer: Optional FeatureEngineer instance.
        """
        self.feature_engineer = feature_engineer
        self.models = {}
        self.availability_predictor = AvailabilityPredictor()
        self.training_summary = {}
    
    def train_all(self, training_data: pd.DataFrame,
                  n_splits: int = 3,
                  test_size: float = 0.2) -> Dict[str, PositionModel]:
        """Train models for all positions.
        
        Args:
            training_data: Full training DataFrame with features and targets.
            n_splits: Number of TimeSeriesSplit folds.
            test_size: Fraction of data to hold out for final validation.
            
        Returns:
            Dict of position -> trained PositionModel.
        """
        logger.info("=" * 60)
        logger.info("Starting Model Training Pipeline")
        logger.info("=" * 60)
        
        # Split data by position
        for position in self.POSITIONS:
            logger.info(f"\nTraining {position} model...")
            
            # Filter data for this position
            pos_data = training_data[training_data['_position'] == position].copy()
            
            if len(pos_data) < 100:
                logger.warning(f"Insufficient data for {position} ({len(pos_data)} samples)")
                continue
            
            # Get feature columns for this position
            feature_cols = self._get_feature_columns(position, pos_data.columns)
            
            # Prepare X and y
            X = pos_data[feature_cols].fillna(0)
            y = pos_data['actual_points']
            
            # Time-series split for validation
            model, metrics = self._train_with_cv(X, y, position, n_splits, test_size)
            
            self.models[position] = model
            self.training_summary[position] = metrics
            
            logger.info(f"  {position}: MAE={metrics['mae']:.3f}, "
                       f"RMSE={metrics['rmse']:.3f}, R²={metrics['r2']:.4f}")
        
        # Train availability predictor
        logger.info("\nTraining availability predictor...")
        self._train_availability(training_data)
        
        # Save all models
        self._save_all()
        
        return self.models
    
    def _get_feature_columns(self, position: str, 
                            available_cols: pd.Index) -> List[str]:
        """Get feature columns available for training."""
        from models.feature_engineering import FeatureEngineer
        
        # Get ideal features for this position
        ideal_features = (FeatureEngineer.COMMON_FEATURES + 
                         FeatureEngineer.POSITION_FEATURES.get(position, []))
        
        # Filter to columns that exist in data
        available = [col for col in ideal_features if col in available_cols]
        
        # Also include any additional numeric columns (except targets and metadata)
        excluded = ['actual_points', 'actual_minutes', '_player_id', '_target_gw', '_position']
        for col in available_cols:
            if col not in available and col not in excluded and not col.startswith('_'):
                # Check if numeric
                available.append(col)
        
        return list(set(available))
    
    def _train_with_cv(self, X: pd.DataFrame, y: pd.Series,
                       position: str, n_splits: int,
                       test_size: float) -> Tuple[PositionModel, Dict]:
        """Train model with time-series cross-validation.
        
        Uses TimeSeriesSplit to ensure we only train on past data
        to predict future outcomes.
        """
        # Hold out final test set
        split_idx = int(len(X) * (1 - test_size))
        X_train_full, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train_full, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Time-series cross-validation on training set
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = {'mae': [], 'rmse': [], 'r2': []}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_full)):
            X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
            y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
            
            # Train temporary model for this fold
            fold_model = PositionModel(position)
            fold_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = fold_model.predict(X_val)
            cv_scores['mae'].append(mean_absolute_error(y_val, y_pred))
            cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred)))
            cv_scores['r2'].append(r2_score(y_val, y_pred))
        
        logger.info(f"  CV Results ({n_splits} folds): "
                   f"MAE={np.mean(cv_scores['mae']):.3f}±{np.std(cv_scores['mae']):.3f}")
        
        # Train final model on full training set
        final_model = PositionModel(position)
        final_model.fit(X_train_full, y_train_full, 
                       validation_data=(X_test, y_test))
        
        # Final metrics
        metrics = {
            'mae': final_model.metrics['mae'],
            'rmse': final_model.metrics['rmse'],
            'r2': final_model.metrics['r2'],
            'cv_mae_mean': round(np.mean(cv_scores['mae']), 3),
            'cv_mae_std': round(np.std(cv_scores['mae']), 3),
            'training_samples': len(X_train_full),
            'test_samples': len(X_test),
            'feature_importance': final_model.metrics.get('feature_importance', {})
        }
        
        return final_model, metrics
    
    def _train_availability(self, training_data: pd.DataFrame):
        """Train the availability predictor."""
        # Need minutes history features
        required_cols = ['minutes_rolling_3', 'minutes_rolling_6', 
                        'minutes_trend', 'actual_minutes']
        
        available_cols = [c for c in required_cols if c in training_data.columns]
        
        if 'actual_minutes' not in training_data.columns:
            logger.warning("Cannot train availability model - missing actual_minutes")
            return
        
        X = training_data[available_cols[:-1]].copy() if len(available_cols) > 1 else pd.DataFrame()
        
        # Add chance_playing if available
        if 'chance_playing' in training_data.columns:
            X['chance_playing'] = training_data['chance_playing']
        else:
            X['chance_playing'] = 100
        
        # Fill missing columns
        for col in self.availability_predictor.feature_cols:
            if col not in X.columns:
                X[col] = 0
        
        y_minutes = training_data['actual_minutes']
        
        if len(X) > 100:
            self.availability_predictor.fit(X, y_minutes)
            logger.info("  Availability predictor trained")
    
    def _save_all(self):
        """Save all trained models."""
        for position, model in self.models.items():
            model.save()
        
        if self.availability_predictor.is_trained:
            self.availability_predictor.save()
        
        # Save training summary
        summary_path = MODELS_DIR / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(self.training_summary, f, indent=2, default=str)
        
        logger.info(f"\nAll models saved to {MODELS_DIR}")
    
    def load_models(self) -> Dict[str, PositionModel]:
        """Load all trained models from disk."""
        for position in self.POSITIONS:
            filepath = MODELS_DIR / f'model_{position}.pkl'
            if filepath.exists():
                self.models[position] = PositionModel.load(filepath)
                logger.info(f"Loaded {position} model")
        
        avail_path = MODELS_DIR / 'availability_model.pkl'
        if avail_path.exists():
            self.availability_predictor = AvailabilityPredictor.load(avail_path)
        
        return self.models


def train_models_from_warehouse(min_minutes: int = 180) -> ModelTrainer:
    """Convenience function to train all models from warehouse data.
    
    Args:
        min_minutes: Minimum total minutes for player inclusion.
        
    Returns:
        Trained ModelTrainer instance.
    """
    from etl.transformers import load_parquet
    from models.feature_engineering import FeatureEngineer
    
    logging.basicConfig(level=logging.INFO)
    
    # Load warehouse data
    logger.info("Loading data from warehouse...")
    players_df = load_parquet('players.parquet')
    fixtures_df = load_parquet('fixtures.parquet')
    
    # Build training data
    logger.info("Building feature vectors...")
    engineer = FeatureEngineer(players_df, fixtures_df)
    training_data = engineer.build_training_dataset(
        start_gw=5, 
        min_minutes=min_minutes
    )
    
    if training_data.empty:
        logger.error("No training data available!")
        return None
    
    # Save training data
    engineer.save_dataset(training_data, 'training_data.parquet')
    
    # Train models
    trainer = ModelTrainer(engineer)
    trainer.train_all(training_data)
    
    return trainer


if __name__ == '__main__':
    trainer = train_models_from_warehouse()
    
    if trainer:
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        
        for pos, metrics in trainer.training_summary.items():
            print(f"\n{pos}:")
            print(f"  MAE: {metrics['mae']:.3f}")
            print(f"  RMSE: {metrics['rmse']:.3f}")
            print(f"  R²: {metrics['r2']:.4f}")
            print(f"  CV MAE: {metrics['cv_mae_mean']:.3f} ± {metrics['cv_mae_std']:.3f}")
            print(f"  Samples: {metrics['training_samples']} train, {metrics['test_samples']} test")
            
            if metrics.get('feature_importance'):
                print("  Top Features:")
                for feat, imp in list(metrics['feature_importance'].items())[:5]:
                    print(f"    - {feat}: {imp:.4f}")

