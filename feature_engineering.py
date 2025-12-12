"""
Feature engineering module for SEP event probability calibration.
Creates features from existing data columns.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from issue_time.
    
    Args:
        df: DataFrame with issue_time column
        
    Returns:
        DataFrame with temporal features added
    """
    df = df.copy()
    
    if 'issue_time' not in df.columns:
        return df
    
    df['hour'] = df['issue_time'].dt.hour
    df['day_of_week'] = df['issue_time'].dt.dayofweek
    df['day_of_month'] = df['issue_time'].dt.day
    df['month'] = df['issue_time'].dt.month
    df['day_of_year'] = df['issue_time'].dt.dayofyear
    
    # Cyclical encoding for hour and day_of_week
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df


def create_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from model information.
    
    Args:
        df: DataFrame with model columns
        
    Returns:
        DataFrame with model features added
    """
    df = df.copy()
    
    # Extract prediction horizon from model_short
    if 'model_short' in df.columns:
        df['pred_horizon'] = df['model_short'].str.extract(r'(\d+-\d+ hrs)')
        
        # Create numeric horizon features
        horizon_map = {
            '0-24 hrs': 12,  # midpoint in hours
            '24-48 hrs': 36,
            '48-72 hrs': 60,
            '72-96 hrs': 84
        }
        df['horizon_midpoint'] = df['pred_horizon'].map(horizon_map).fillna(0)
        
        # One-hot encode model names (or use label encoding for simplicity)
        df['model_name_encoded'] = pd.Categorical(df['model_name']).codes if 'model_name' in df.columns else 0
    
    # Extract threshold information
    if 'threshold' in df.columns:
        df['threshold_10'] = (df['threshold'] == 10.0).astype(int)
        df['threshold_1'] = (df['threshold'] == 1.0).astype(int)
    
    # All clear flag
    if 'all_clear' in df.columns:
        df['all_clear_int'] = df['all_clear'].astype(int)
        df['all_clear_prob_thresh'] = pd.to_numeric(df['all_clear_prob_thresh'], errors='coerce').fillna(0)
    
    return df


def create_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregated features by grouping predictions for the same time window.
    
    Args:
        df: DataFrame with prediction data
        
    Returns:
        DataFrame with aggregated features added
    """
    df = df.copy()
    
    # Create a unique identifier for prediction windows
    df['pred_window_id'] = (
        df['pred_start'].astype(str) + '_' + 
        df['pred_end'].astype(str)
    )
    
    # Group by prediction window and calculate aggregated statistics
    agg_features = df.groupby('pred_window_id').agg({
        'prob_value': ['mean', 'std', 'min', 'max', 'count'],
        'SEP_event_label': 'max',  # If any prediction has label=1, window has event
        'all_clear_prob_thresh': 'mean'
    }).reset_index()
    
    # Flatten column names
    agg_features.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                           for col in agg_features.columns]
    agg_features.rename(columns={'SEP_event_label_max': 'window_sep_label'}, inplace=True)
    
    # Merge back to original dataframe
    df = df.merge(agg_features, on='pred_window_id', suffixes=('', '_agg'))
    
    # Create relative features
    df['prob_value_rel_to_mean'] = df['prob_value'] / (df['prob_value_mean'] + 1e-10)
    df['prob_value_rel_to_max'] = df['prob_value'] / (df['prob_value_max'] + 1e-10)
    df['prob_rank_in_window'] = df.groupby('pred_window_id')['prob_value'].rank(pct=True)
    
    return df


def create_probability_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from probability values.
    
    Args:
        df: DataFrame with prob_value column
        
    Returns:
        DataFrame with probability features added
    """
    df = df.copy()
    
    # Log transform of probabilities
    df['prob_log'] = np.log1p(df['prob_value'])
    
    # Probability bins
    df['prob_bin_low'] = (df['prob_value'] < 0.01).astype(int)
    df['prob_bin_med'] = ((df['prob_value'] >= 0.01) & (df['prob_value'] < 0.05)).astype(int)
    df['prob_bin_high'] = (df['prob_value'] >= 0.05).astype(int)
    
    # Probability squared (for non-linearity)
    df['prob_squared'] = df['prob_value'] ** 2
    
    # Interaction with uncertainty if available
    if 'prob_uncertainty' in df.columns:
        df['prob_with_uncertainty'] = df['prob_value'] * (1 + df['prob_uncertainty'])
    
    return df


def create_all_features(df: pd.DataFrame, include_aggregated: bool = True) -> pd.DataFrame:
    """
    Create all features for the model.
    
    Args:
        df: Preprocessed dataframe
        include_aggregated: Whether to include aggregated window features
        
    Returns:
        DataFrame with all features added
    """
    print("Creating temporal features...")
    df = create_temporal_features(df)
    
    print("Creating model features...")
    df = create_model_features(df)
    
    print("Creating probability features...")
    df = create_probability_features(df)
    
    if include_aggregated:
        print("Creating aggregated features...")
        df = create_aggregated_features(df)
    
    return df


def get_feature_columns(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> List[str]:
    """
    Get list of feature columns to use for modeling.
    
    Args:
        df: DataFrame with features
        exclude_cols: List of columns to exclude
        
    Returns:
        List of feature column names
    """
    exclude_cols = exclude_cols or []
    
    # Base columns to exclude
    base_exclude = [
        'issue_time', 'pred_start', 'pred_end', 'issue_hour',
        'SEP_event_label', 'model_short', 'model_name', 'species',
        'location', 'energy_min', 'energy_max', 'energy_units',
        'mode', 'pred_window_id', 'window_sep_label', 'event_id',
        'pred_horizon'
    ]
    
    exclude_cols = set(exclude_cols + base_exclude)
    
    # Select numeric and boolean columns
    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols:
            if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
                feature_cols.append(col)
    
    return feature_cols


if __name__ == "__main__":
    # Test feature engineering
    from data_loader import load_data, preprocess_data
    
    print("Loading data...")
    df = load_data()
    df = preprocess_data(df)
    
    print(f"Original columns: {len(df.columns)}")
    
    print("\nCreating features...")
    df = create_all_features(df)
    
    print(f"Columns after feature engineering: {len(df.columns)}")
    
    feature_cols = get_feature_columns(df)
    print(f"\nFeature columns ({len(feature_cols)}):")
    for col in feature_cols[:20]:  # Show first 20
        print(f"  - {col}")
    if len(feature_cols) > 20:
        print(f"  ... and {len(feature_cols) - 20} more")

