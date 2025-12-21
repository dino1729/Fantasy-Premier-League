"""ETL Module for FPL Data Warehouse.

This module provides the data infrastructure for the FPL analytics pipeline:

- fetchers.py: Extract data from FPL API, ClubElo, and future sources
- transformers.py: Clean, normalize, and format data for the warehouse
- pipeline.py: Orchestrate the full ETL flow

Usage:
    from etl.fetchers import FPLFetcher, ClubEloFetcher
    from etl.transformers import PlayerTransformer, load_parquet
    from etl.pipeline import ETLPipeline

    # Run full pipeline
    pipeline = ETLPipeline()
    outputs = pipeline.run()
    
    # Or run via CLI
    # python -m etl.pipeline
"""

from etl.fetchers import FPLFetcher, ClubEloFetcher, FixtureDifficultyCalculator
from etl.transformers import (
    PlayerTransformer, 
    FixtureTransformer, 
    ProjectionTransformer,
    load_parquet
)
from etl.pipeline import ETLPipeline, HeuristicAdapter

__all__ = [
    'FPLFetcher',
    'ClubEloFetcher', 
    'FixtureDifficultyCalculator',
    'PlayerTransformer',
    'FixtureTransformer',
    'ProjectionTransformer',
    'load_parquet',
    'ETLPipeline',
    'HeuristicAdapter',
]

