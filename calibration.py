"""
Probability calibration module for SEP event forecasts.
Implements Platt scaling, Isotonic regression, and temperature scaling.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Optional


class ProbabilityCalibrator:
    """
    Base class for probability calibration.
    Takes raw probabilities and calibrates them to better match observed frequencies.
    """
    
    def __init__(self, method: str = 'platt'):
        """
        Initialize calibrator.
        
        Args:
            method: Calibration method ('platt', 'isotonic', or 'temperature')
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        
    def fit(self, y_prob: np.ndarray, y_true: np.ndarray):
        """
        Fit the calibrator.
        
        Args:
            y_prob: Predicted probabilities (1D array)
            y_true: True binary labels (1D array)
        """
        y_prob = np.asarray(y_prob).ravel()
        y_true = np.asarray(y_true).ravel()
        
        if self.method == 'platt':
            self._fit_platt(y_prob, y_true)
        elif self.method == 'isotonic':
            self._fit_isotonic(y_prob, y_true)
        elif self.method == 'temperature':
            self._fit_temperature(y_prob, y_true)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        self.is_fitted = True
        
    def _fit_platt(self, y_prob: np.ndarray, y_true: np.ndarray):
        """Fit Platt scaling (logistic regression on log-odds)."""
        # Convert probabilities to log-odds, handling edge cases
        y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
        log_odds = np.log(y_prob_clipped / (1 - y_prob_clipped))
        
        # Fit logistic regression
        X = log_odds.reshape(-1, 1)
        self.calibrator = LogisticRegression()
        self.calibrator.fit(X, y_true)
        
    def _fit_isotonic(self, y_prob: np.ndarray, y_true: np.ndarray):
        """Fit isotonic regression."""
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(y_prob, y_true)
        
    def _fit_temperature(self, y_prob: np.ndarray, y_true: np.ndarray):
        """Fit temperature scaling (single parameter)."""
        # Temperature scaling: P_cal = sigmoid(logit(P) / T)
        # Find optimal T using optimization
        from scipy.optimize import minimize_scalar
        
        y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped))
        
        def objective(T):
            T = max(T, 0.01)  # Prevent division by zero
            scaled_logits = logits / T
            scaled_probs = 1 / (1 + np.exp(-scaled_logits))
            
            # Cross-entropy loss
            loss = -np.mean(y_true * np.log(scaled_probs + 1e-15) + 
                           (1 - y_true) * np.log(1 - scaled_probs + 1e-15))
            return loss
        
        result = minimize_scalar(objective, bounds=(0.01, 100), method='bounded')
        self.temperature = result.x
        
    def predict_proba(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Predict calibrated probabilities.
        
        Args:
            y_prob: Raw predicted probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before prediction")
        
        y_prob = np.asarray(y_prob).ravel()
        
        if self.method == 'platt':
            y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
            log_odds = np.log(y_prob_clipped / (1 - y_prob_clipped))
            X = log_odds.reshape(-1, 1)
            calibrated = self.calibrator.predict_proba(X)[:, 1]
        elif self.method == 'isotonic':
            calibrated = self.calibrator.predict(y_prob)
        elif self.method == 'temperature':
            y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
            logits = np.log(y_prob_clipped / (1 - y_prob_clipped))
            scaled_logits = logits / self.temperature
            calibrated = 1 / (1 + np.exp(-scaled_logits))
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        # Ensure probabilities are in [0, 1]
        calibrated = np.clip(calibrated, 0, 1)
        return calibrated


def calibrate_probabilities(y_prob: np.ndarray, y_true: np.ndarray, 
                           method: str = 'platt', 
                           return_calibrator: bool = False):
    """
    Convenience function to calibrate probabilities.
    
    Args:
        y_prob: Raw predicted probabilities
        y_true: True binary labels
        method: Calibration method ('platt', 'isotonic', or 'temperature')
        return_calibrator: Whether to return the fitted calibrator
        
    Returns:
        Calibrated probabilities (and optionally the calibrator)
    """
    calibrator = ProbabilityCalibrator(method=method)
    calibrator.fit(y_prob, y_true)
    calibrated_probs = calibrator.predict_proba(y_prob)
    
    if return_calibrator:
        return calibrated_probs, calibrator
    return calibrated_probs


class ModelWiseCalibrator:
    """
    Calibrator that fits separate calibration models for each source model.
    Useful when different models have different calibration needs.
    """
    
    def __init__(self, method: str = 'platt'):
        """
        Initialize model-wise calibrator.
        
        Args:
            method: Calibration method to use for each model
        """
        self.method = method
        self.calibrators = {}
        self.is_fitted = False
        
    def fit(self, y_prob: np.ndarray, y_true: np.ndarray, model_names: np.ndarray):
        """
        Fit separate calibrators for each model.
        
        Args:
            y_prob: Predicted probabilities
            y_true: True binary labels
            model_names: Array of model names/identifiers for each prediction
        """
        model_names = np.asarray(model_names).ravel()
        unique_models = np.unique(model_names)
        
        for model in unique_models:
            mask = model_names == model
            if mask.sum() > 0:
                calibrator = ProbabilityCalibrator(method=self.method)
                calibrator.fit(y_prob[mask], y_true[mask])
                self.calibrators[model] = calibrator
        
        self.is_fitted = True
        
    def predict_proba(self, y_prob: np.ndarray, model_names: np.ndarray) -> np.ndarray:
        """
        Predict calibrated probabilities using model-specific calibrators.
        
        Args:
            y_prob: Raw predicted probabilities
            model_names: Array of model names/identifiers
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before prediction")
        
        y_prob = np.asarray(y_prob).ravel()
        model_names = np.asarray(model_names).ravel()
        calibrated = np.zeros_like(y_prob)
        
        for model, calibrator in self.calibrators.items():
            mask = model_names == model
            if mask.sum() > 0:
                calibrated[mask] = calibrator.predict_proba(y_prob[mask])
        
        return calibrated


if __name__ == "__main__":
    # Test calibration
    print("Testing probability calibration...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # True probabilities (but model outputs are too conservative)
    true_probs = np.random.beta(0.5, 10, n_samples)
    y_true = (np.random.random(n_samples) < true_probs).astype(int)
    
    # Model outputs (too conservative)
    y_prob_raw = true_probs * 0.1  # Make them 10x too small
    
    print(f"\nRaw probabilities:")
    print(f"  Mean: {y_prob_raw.mean():.4f}")
    print(f"  Max: {y_prob_raw.max():.4f}")
    
    # Test different calibration methods
    methods = ['platt', 'isotonic', 'temperature']
    for method in methods:
        calibrated = calibrate_probabilities(y_prob_raw, y_true, method=method)
        print(f"\n{method.capitalize()} calibrated probabilities:")
        print(f"  Mean: {calibrated.mean():.4f}")
        print(f"  Max: {calibrated.max():.4f}")

