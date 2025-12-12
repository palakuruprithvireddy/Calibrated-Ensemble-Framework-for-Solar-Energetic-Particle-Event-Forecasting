"""
Ensemble meta-learner module for combining multiple model predictions.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Dict, Tuple


class EnsembleMetaLearner:
    """
    Meta-learner that combines predictions from multiple models.
    """
    
    def __init__(self, method: str = 'logistic', 
                 regularization: float = 1.0,
                 class_weight: Optional[str] = 'balanced'):
        """
        Initialize ensemble meta-learner.
        
        Args:
            method: Method to use ('logistic', 'gradient_boosting', or 'weighted_average')
            regularization: Regularization strength (for logistic regression)
            class_weight: Class weight strategy ('balanced' or None)
        """
        self.method = method
        self.regularization = regularization
        self.class_weight = class_weight
        self.model = None
        self.scaler = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the meta-learner.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        
        if self.method == 'logistic':
            self.model = LogisticRegression(
                C=self.regularization,
                class_weight=self.class_weight,
                max_iter=1000,
                solver='lbfgs'
            )
            # Scale features for logistic regression
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
        elif self.method == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=50,  # Small number to prevent overfitting
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
            # Calculate class weights manually
            if self.class_weight == 'balanced':
                from sklearn.utils.class_weight import compute_sample_weight
                sample_weights = compute_sample_weight('balanced', y)
                self.model.fit(X, y, sample_weight=sample_weights)
            else:
                self.model.fit(X, y)
                
        elif self.method == 'weighted_average':
            # Simple weighted average - no fitting needed
            self.model = None
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.is_fitted = True
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted probabilities (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X)
        
        if self.method == 'weighted_average':
            # Simple average of probabilities
            return X.mean(axis=1)
        
        if self.method == 'logistic' and self.scaler is not None:
            X = self.scaler.transform(X)
        
        probabilities = self.model.predict_proba(X)[:, 1]
        return probabilities


def pivot_to_ensemble_format(df: pd.DataFrame, 
                            prob_col: str = 'prob_value',
                            model_col: str = 'model_short',
                            label_col: str = 'SEP_event_label',
                            window_col: str = 'pred_window_id') -> Tuple[pd.DataFrame, List[str]]:
    """
    Pivot data so each row represents a prediction window with all model probabilities.
    
    Args:
        df: DataFrame with predictions
        prob_col: Column name for probabilities
        model_col: Column name for model identifiers
        label_col: Column name for labels
        window_col: Column name for prediction window identifier
        
    Returns:
        Pivoted dataframe and list of model feature columns
    """
    # Pivot table: one row per window, one column per model
    pivot_df = df.pivot_table(
        index=window_col,
        columns=model_col,
        values=prob_col,
        aggfunc='mean'  # Take mean if multiple predictions from same model for same window
    ).reset_index()
    
    # Get model columns
    model_cols = [col for col in pivot_df.columns if col != window_col]
    
    # Fill missing values with 0 (no prediction from that model)
    pivot_df[model_cols] = pivot_df[model_cols].fillna(0)
    
    # Merge labels (use max to ensure label=1 if any prediction has it)
    if label_col in df.columns:
        labels = df.groupby(window_col)[label_col].max().reset_index()
        pivot_df = pivot_df.merge(labels, on=window_col, how='left')
        pivot_df[label_col] = pivot_df[label_col].fillna(0).astype(int)
    
    return pivot_df, model_cols


def create_ensemble_features(df_pivoted: pd.DataFrame, model_cols: List[str]) -> pd.DataFrame:
    """
    Create additional features from pivoted model predictions.
    
    Args:
        df_pivoted: Pivoted dataframe
        model_cols: List of model column names
        
    Returns:
        DataFrame with additional ensemble features
    """
    df = df_pivoted.copy()
    
    # Statistics across models
    model_probs = df[model_cols].values
    df['ensemble_mean'] = np.mean(model_probs, axis=1)
    df['ensemble_std'] = np.std(model_probs, axis=1)
    df['ensemble_max'] = np.max(model_probs, axis=1)
    df['ensemble_min'] = np.min(model_probs, axis=1)
    df['ensemble_median'] = np.median(model_probs, axis=1)
    df['ensemble_q75'] = np.percentile(model_probs, 75, axis=1)
    df['ensemble_q25'] = np.percentile(model_probs, 25, axis=1)
    
    # Number of models with non-zero predictions
    df['n_models_active'] = (model_probs > 0).sum(axis=1)
    
    # Agreement metrics
    df['ensemble_range'] = df['ensemble_max'] - df['ensemble_min']
    
    return df


class HierarchicalEnsemble:
    """
    Two-stage ensemble: first aggregate by model, then combine models.
    """
    
    def __init__(self, 
                 model_calibrators: Optional[Dict] = None,
                 ensemble_method: str = 'logistic'):
        """
        Initialize hierarchical ensemble.
        
        Args:
            model_calibrators: Dict of calibrators for each model
            ensemble_method: Method for final ensemble ('logistic', 'weighted_average')
        """
        self.model_calibrators = model_calibrators or {}
        self.ensemble_method = ensemble_method
        self.meta_learner = None
        self.is_fitted = False
        
    def fit(self, 
            df_pivoted: pd.DataFrame,
            model_cols: List[str],
            label_col: str = 'SEP_event_label'):
        """
        Fit the hierarchical ensemble.
        
        Args:
            df_pivoted: Pivoted dataframe
            model_cols: List of model column names
            label_col: Column name for labels
        """
        # Step 1: Calibrate individual model predictions if calibrators provided
        df_calibrated = df_pivoted.copy()
        
        for model_col in model_cols:
            if model_col in self.model_calibrators:
                calibrator = self.model_calibrators[model_col]
                df_calibrated[model_col] = calibrator.predict_proba(
                    df_calibrated[model_col].values
                )
        
        # Step 2: Create ensemble features
        df_features = create_ensemble_features(df_calibrated, model_cols)
        
        # Step 3: Fit meta-learner
        feature_cols = model_cols + [col for col in df_features.columns 
                                     if col.startswith('ensemble_')]
        X = df_features[feature_cols].values
        y = df_features[label_col].values
        
        self.meta_learner = EnsembleMetaLearner(method=self.ensemble_method)
        self.meta_learner.fit(X, y)
        
        self.is_fitted = True
        
    def predict_proba(self,
                     df_pivoted: pd.DataFrame,
                     model_cols: List[str]) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            df_pivoted: Pivoted dataframe
            model_cols: List of model column names
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Step 1: Calibrate
        df_calibrated = df_pivoted.copy()
        for model_col in model_cols:
            if model_col in self.model_calibrators:
                calibrator = self.model_calibrators[model_col]
                df_calibrated[model_col] = calibrator.predict_proba(
                    df_calibrated[model_col].values
                )
        
        # Step 2: Create features
        df_features = create_ensemble_features(df_calibrated, model_cols)
        
        # Step 3: Predict
        feature_cols = model_cols + [col for col in df_features.columns 
                                     if col.startswith('ensemble_')]
        X = df_features[feature_cols].values
        
        return self.meta_learner.predict_proba(X)


if __name__ == "__main__":
    # Test ensemble
    print("Testing ensemble meta-learner...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 500
    n_models = 5
    
    # Generate predictions from multiple models
    X = np.random.rand(n_samples, n_models) * 0.1  # Conservative predictions
    
    # True labels (some rare events)
    y = (np.random.random(n_samples) < 0.05).astype(int)
    
    # Fit ensemble
    ensemble = EnsembleMetaLearner(method='logistic')
    ensemble.fit(X, y)
    
    # Predict
    predictions = ensemble.predict_proba(X)
    
    print(f"\nOriginal probabilities mean: {X.mean():.4f}")
    print(f"Ensemble predictions mean: {predictions.mean():.4f}")
    print(f"True event rate: {y.mean():.4f}")

