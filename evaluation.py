"""
Evaluation metrics and visualization for SEP event probability forecasts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    brier_score_loss, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, confusion_matrix,
    classification_report
)
from typing import Optional, Tuple, Dict, List
import warnings


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate Brier score (mean squared error of probabilities).
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        
    Returns:
        Brier score (lower is better)
    """
    return brier_score_loss(y_true, y_prob)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = 'f1') -> Tuple[float, float]:
    """
    Find optimal threshold for a given metric.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        metric: Metric to optimize ('f1', 'f2', 'youden')
        
    Returns:
        Optimal threshold and corresponding metric value
    """
    from sklearn.metrics import f1_score, fbeta_score
    
    thresholds = np.linspace(0, 1, 1000)
    best_threshold = 0
    best_score = 0
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        if y_true.sum() == 0:
            continue
            
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'f2':
            score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        elif metric == 'youden':
            # Youden's J statistic (TPR - FPR)
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            score = tpr - fpr
        else:
            continue
            
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def calculate_metrics(y_true: np.ndarray, 
                     y_prob: np.ndarray,
                     threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        'brier_score': brier_score(y_true, y_prob),
        'roc_auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        'pr_auc': average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        'threshold': threshold
    }
    
    # Classification metrics at threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)
    metrics['false_negatives'] = int(fn)
    
    if tp + fp > 0:
        metrics['precision'] = tp / (tp + fp)
    else:
        metrics['precision'] = 0.0
    
    if tp + fn > 0:
        metrics['recall'] = tp / (tp + fn)
    else:
        metrics['recall'] = 0.0
    
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1_score'] = 0.0
    
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    # Add optimal threshold information
    opt_thresh, opt_f1 = find_optimal_threshold(y_true, y_prob, metric='f1')
    metrics['optimal_threshold'] = opt_thresh
    metrics['optimal_f1'] = opt_f1
    
    return metrics


def reliability_diagram(y_true: np.ndarray, 
                       y_prob: np.ndarray,
                       n_bins: int = 10,
                       ax: Optional[plt.Axes] = None,
                       title: str = 'Reliability Diagram') -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot reliability diagram showing calibration.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration plot
        ax: Matplotlib axes (if None, creates new figure)
        title: Plot title
        
    Returns:
        Figure and axes objects
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Calculate fraction of positives and mean predicted probability in each bin
    fractions = []
    mean_probs = []
    counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            fraction = y_true[in_bin].mean()
            mean_prob = y_prob[in_bin].mean()
            count = in_bin.sum()
            
            fractions.append(fraction)
            mean_probs.append(mean_prob)
            counts.append(count)
        else:
            fractions.append(np.nan)
            mean_probs.append(np.nan)
            counts.append(0)
    
    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    
    # Calibration curve
    valid_mask = ~np.isnan(fractions)
    if valid_mask.sum() > 0:
        ax.plot(np.array(mean_probs)[valid_mask], 
                np.array(fractions)[valid_mask],
                's-', label='Model', linewidth=2, markersize=8)
    
    # Binning
    bin_centers = (bin_lowers + bin_uppers) / 2
    ax.bar(bin_centers[valid_mask], 
           np.array(fractions)[valid_mask],
           width=0.1, alpha=0.3, label='Observed frequency')
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    
    return fig, ax


def plot_roc_curve(y_true: np.ndarray, 
                  y_prob: np.ndarray,
                  ax: Optional[plt.Axes] = None,
                  label: str = 'Model',
                  title: str = 'ROC Curve') -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot ROC curve.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        ax: Matplotlib axes
        label: Label for the curve
        title: Plot title
        
    Returns:
        Figure and axes objects
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    
    if len(np.unique(y_true)) < 2:
        warnings.warn("Cannot plot ROC curve with only one class")
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure
        return fig, ax
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
    
    ax.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_precision_recall_curve(y_true: np.ndarray,
                               y_prob: np.ndarray,
                               ax: Optional[plt.Axes] = None,
                               label: str = 'Model',
                               title: str = 'Precision-Recall Curve') -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        ax: Matplotlib axes
        label: Label for the curve
        title: Plot title
        
    Returns:
        Figure and axes objects
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    
    if len(np.unique(y_true)) < 2:
        warnings.warn("Cannot plot PR curve with only one class")
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure
        return fig, ax
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
    
    ax.plot(recall, precision, linewidth=2, label=f'{label} (AUC = {pr_auc:.3f})')
    baseline = y_true.mean()
    ax.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def evaluate_calibration(y_true: np.ndarray,
                        y_prob: np.ndarray,
                        n_bins: int = 10) -> Dict[str, float]:
    """
    Evaluate calibration using Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        Dictionary with ECE and MCE
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    mce = 0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
            
            ece += prop_in_bin * calibration_error
            mce = max(mce, calibration_error)
    
    return {
        'ece': ece,
        'mce': mce
    }


def compare_predictions(y_true: np.ndarray,
                       predictions_dict: Dict[str, np.ndarray],
                       save_path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple prediction methods.
    
    Args:
        y_true: True binary labels
        predictions_dict: Dictionary mapping method names to predictions
        save_path: Optional path to save comparison plots
        
    Returns:
        Dictionary of metrics for each method
    """
    results = {}
    
    for method_name, y_prob in predictions_dict.items():
        metrics = calculate_metrics(y_true, y_prob)
        calibration = evaluate_calibration(y_true, y_prob)
        metrics.update(calibration)
        results[method_name] = metrics
    
    # Create comparison plots if save_path provided
    if save_path:
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        
        # Reliability diagrams
        ax1 = axes[0, 0]
        for method_name, y_prob in predictions_dict.items():
            reliability_diagram(y_true, y_prob, ax=ax1, title='')
        ax1.set_title('Reliability Diagrams Comparison')
        
        # ROC curves
        ax2 = axes[0, 1]
        for method_name, y_prob in predictions_dict.items():
            plot_roc_curve(y_true, y_prob, ax=ax2, label=method_name, title='')
        ax2.set_title('ROC Curves Comparison')
        
        # PR curves
        ax3 = axes[1, 0]
        for method_name, y_prob in predictions_dict.items():
            plot_precision_recall_curve(y_true, y_prob, ax=ax3, label=method_name, title='')
        ax3.set_title('Precision-Recall Curves Comparison')
        
        # Metrics comparison
        ax4 = axes[1, 1]
        metrics_df = pd.DataFrame(results).T
        metrics_to_plot = ['brier_score', 'roc_auc', 'pr_auc', 'ece']
        metrics_df[metrics_to_plot].plot(kind='bar', ax=ax4, rot=45)
        ax4.set_title('Metrics Comparison')
        ax4.set_ylabel('Score')
        ax4.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return results


def print_metrics_report(metrics: Dict[str, float], method_name: str = 'Model'):
    """
    Print a formatted metrics report.
    
    Args:
        metrics: Dictionary of metrics
        method_name: Name of the method
    """
    print(f"\n{'='*60}")
    print(f"Metrics Report: {method_name}")
    print(f"{'='*60}")
    print(f"Brier Score:        {metrics.get('brier_score', np.nan):.6f} (lower is better)")
    print(f"ROC AUC:            {metrics.get('roc_auc', np.nan):.6f} (higher is better)")
    print(f"PR AUC:             {metrics.get('pr_auc', np.nan):.6f} (higher is better)")
    print(f"ECE:                {metrics.get('ece', np.nan):.6f} (lower is better)")
    print(f"MCE:                {metrics.get('mce', np.nan):.6f} (lower is better)")
    print(f"\nAt threshold {metrics.get('threshold', 0.5):.3f}:")
    print(f"  Precision:        {metrics.get('precision', np.nan):.4f}")
    print(f"  Recall:           {metrics.get('recall', np.nan):.4f}")
    print(f"  F1 Score:         {metrics.get('f1_score', np.nan):.4f}")
    print(f"  Accuracy:         {metrics.get('accuracy', np.nan):.4f}")
    print(f"  TP: {metrics.get('true_positives', 0)}, FP: {metrics.get('false_positives', 0)}, "
          f"TN: {metrics.get('true_negatives', 0)}, FN: {metrics.get('false_negatives', 0)}")
    
    # Show optimal threshold if available
    if 'optimal_threshold' in metrics and 'optimal_f1' in metrics:
        opt_thresh = metrics['optimal_threshold']
        opt_f1 = metrics['optimal_f1']
        print(f"\nOptimal threshold:  {opt_thresh:.4f} (F1 = {opt_f1:.4f})")
        print(f"  Note: For rare events, use threshold ~0.01-0.07 instead of 0.5")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test evaluation functions
    print("Testing evaluation functions...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # True probabilities (calibrated)
    true_probs = np.random.beta(1, 10, n_samples)
    y_true = (np.random.random(n_samples) < true_probs).astype(int)
    
    # Model predictions (too conservative)
    y_prob = true_probs * 0.1
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_prob)
    print_metrics_report(metrics, "Conservative Model")
    
    # Calculate calibration errors
    cal_metrics = evaluate_calibration(y_true, y_prob)
    print(f"ECE: {cal_metrics['ece']:.6f}")
    print(f"MCE: {cal_metrics['mce']:.6f}")

