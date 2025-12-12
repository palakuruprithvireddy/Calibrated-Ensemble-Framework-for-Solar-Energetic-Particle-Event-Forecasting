"""
Data loading and preprocessing module for SEP event probability calibration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_data(data_path: str = "data/combined_Labelled_modeldata.csv") -> pd.DataFrame:
    """
    Load the SEP scoreboard data from CSV file.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path, low_memory=False)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the loaded data.
    
    Args:
        df: Raw dataframe
        
    Returns:
        Preprocessed dataframe
    """
    df = df.copy()
    
    # Parse datetime columns
    datetime_cols = ['issue_time', 'pred_start', 'pred_end', 'issue_hour']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
    
    # Handle missing values in critical columns
    df['prob_value'] = pd.to_numeric(df['prob_value'], errors='coerce').fillna(0)
    df['SEP_event_label'] = pd.to_numeric(df['SEP_event_label'], errors='coerce').fillna(0).astype(int)
    
    # Ensure all_clear is boolean
    if 'all_clear' in df.columns:
        df['all_clear'] = df['all_clear'].astype(bool)
    
    # Sort by issue time for temporal analysis
    if 'issue_time' in df.columns:
        df = df.sort_values('issue_time').reset_index(drop=True)
    
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics about the dataset.
    
    Args:
        df: Preprocessed dataframe
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_rows': len(df),
        'sep_events': int(df['SEP_event_label'].sum()),
        'no_sep_events': int((df['SEP_event_label'] == 0).sum()),
        'event_rate': df['SEP_event_label'].mean(),
        'unique_models': df['model_short'].nunique() if 'model_short' in df.columns else 0,
        'date_range': {
            'start': df['issue_time'].min() if 'issue_time' in df.columns else None,
            'end': df['issue_time'].max() if 'issue_time' in df.columns else None
        },
        'probability_stats': {
            'mean': float(df['prob_value'].mean()),
            'median': float(df['prob_value'].median()),
            'std': float(df['prob_value'].std()),
            'min': float(df['prob_value'].min()),
            'max': float(df['prob_value'].max()),
            'q25': float(df['prob_value'].quantile(0.25)),
            'q75': float(df['prob_value'].quantile(0.75)),
            'q95': float(df['prob_value'].quantile(0.95)),
            'q99': float(df['prob_value'].quantile(0.99))
        }
    }
    
    return summary


def identify_sep_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify distinct SEP events by grouping prediction windows that overlap.
    This creates a unique event identifier for each SEP event.
    
    Args:
        df: Preprocessed dataframe
        
    Returns:
        DataFrame with event_id column added
    """
    df = df.copy()
    
    # Create a unique identifier for each prediction window
    df['pred_window_id'] = (
        df['pred_start'].astype(str) + '_' + 
        df['pred_end'].astype(str)
    )
    
    # Identify distinct SEP events
    # Group by prediction windows that have SEP_event_label == 1
    sep_rows = df[df['SEP_event_label'] == 1].copy()
    
    if len(sep_rows) > 0:
        # Get unique prediction windows that had SEP events
        sep_windows = sep_rows['pred_window_id'].unique()
        
        # Create event_id mapping
        df['event_id'] = 0
        for idx, window in enumerate(sep_windows, 1):
            df.loc[df['pred_window_id'] == window, 'event_id'] = idx
        
        # Count distinct events
        distinct_events = len(sep_windows)
        print(f"Identified {distinct_events} distinct SEP events")
    else:
        df['event_id'] = 0
    
    return df


if __name__ == "__main__":
    # Test loading and preprocessing
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} rows")
    
    print("\nPreprocessing data...")
    df = preprocess_data(df)
    
    print("\nGetting summary...")
    summary = get_data_summary(df)
    print(f"Total rows: {summary['total_rows']}")
    print(f"SEP events: {summary['sep_events']}")
    print(f"Event rate: {summary['event_rate']:.4f}")
    print(f"Unique models: {summary['unique_models']}")
    print(f"\nProbability statistics:")
    for key, value in summary['probability_stats'].items():
        print(f"  {key}: {value:.6f}")
    
    print("\nIdentifying SEP events...")
    df = identify_sep_events(df)
    
    print(f"\nColumns: {list(df.columns)}")

