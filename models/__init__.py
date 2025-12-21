"""ML Models Module for FPL Point Predictions.

This module provides the machine learning infrastructure for predicting
player expected points (xP):

- feature_engineering.py: Build training vectors with rolling windows
- train.py: Train position-specific XGBoost+Ridge ensembles
- inference.py: Generate projections_horizon.parquet for solver

Usage:
    # Train models
    from models.train import train_models_from_warehouse
    trainer = train_models_from_warehouse()
    
    # Generate predictions
    from models.inference import run_inference
    projections = run_inference(horizon=5)
    
    # Or via CLI:
    # python -m models.train
    # python -m models.inference
"""

from models.feature_engineering import FeatureEngineer
from models.train import PositionModel, AvailabilityPredictor, ModelTrainer
from models.inference import FPLInferencePipeline, run_inference

__all__ = [
    'FeatureEngineer',
    'PositionModel',
    'AvailabilityPredictor', 
    'ModelTrainer',
    'FPLInferencePipeline',
    'run_inference',
]

