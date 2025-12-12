"""
Main training pipeline for SEP event probability calibration model.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
from typing import Dict, Tuple, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_data, preprocess_data, identify_sep_events, get_data_summary
from feature_engineering import create_all_features, get_feature_columns
from calibration import ProbabilityCalibrator, ModelWiseCalibrator
from ensemble_model import EnsembleMetaLearner, pivot_to_ensemble_format, create_ensemble_features
from evaluation import (
    calculate_metrics, evaluate_calibration, compare_predictions,
    print_metrics_report, reliability_diagram, plot_roc_curve, plot_precision_recall_curve,
    find_optimal_threshold
)

from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class SEPCalibrationPipeline:
    """
    Main pipeline for SEP event probability calibration.
    """
    
    def __init__(self, 
                 data_path: str = "data/combined_Labelled_modeldata.csv",
                 output_dir: str = "outputs",
                 random_state: int = 42):
        """
        Initialize pipeline.
        
        Args:
            data_path: Path to data file
            output_dir: Directory for outputs
            random_state: Random seed
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        np.random.seed(random_state)
        
        self.df = None
        self.df_features = None
        self.df_pivoted = None
        self.model_cols = None
        
    def load_and_preprocess(self):
        """Load and preprocess data."""
        print("="*60)
        print("STEP 1: Loading and Preprocessing Data")
        print("="*60)
        
        print(f"Loading data from {self.data_path}...")
        self.df = load_data(self.data_path)
        print(f"Loaded {len(self.df)} rows")
        
        print("\nPreprocessing data...")
        self.df = preprocess_data(self.df)
        
        print("\nIdentifying SEP events...")
        self.df = identify_sep_events(self.df)
        
        # Print summary
        summary = get_data_summary(self.df)
        print(f"\nData Summary:")
        print(f"  Total rows: {summary['total_rows']}")
        print(f"  SEP events: {summary['sep_events']} ({summary['event_rate']:.2%})")
        print(f"  Unique models: {summary['unique_models']}")
        print(f"  Probability range: [{summary['probability_stats']['min']:.6f}, "
              f"{summary['probability_stats']['max']:.6f}]")
        print(f"  Probability mean: {summary['probability_stats']['mean']:.6f}")
        
    def engineer_features(self):
        """Create features for modeling."""
        print("\n" + "="*60)
        print("STEP 2: Feature Engineering")
        print("="*60)
        
        print("Creating features...")
        self.df_features = create_all_features(self.df.copy())
        
        feature_cols = get_feature_columns(self.df_features)
        print(f"\nCreated {len(feature_cols)} features")
        
        # Create pivoted format for ensemble
        print("\nCreating ensemble format (pivoted)...")
        self.df_pivoted, self.model_cols = pivot_to_ensemble_format(
            self.df_features,
            prob_col='prob_value',
            model_col='model_short',
            label_col='SEP_event_label',
            window_col='pred_window_id'
        )
        print(f"Pivoted to {len(self.df_pivoted)} prediction windows")
        print(f"Number of models: {len(self.model_cols)}")
        
    def train_calibration_models(self, 
                                calibration_method: str = 'platt',
                                use_model_wise: bool = False) -> Dict[str, ProbabilityCalibrator]:
        """
        Train probability calibration models.
        
        Args:
            calibration_method: Method to use ('platt', 'isotonic', 'temperature')
            use_model_wise: Whether to fit separate calibrators per model
            
        Returns:
            Dictionary of calibrators
        """
        print("\n" + "="*60)
        print("STEP 3: Training Calibration Models")
        print("="*60)
        
        y_prob = self.df_features['prob_value'].values
        y_true = self.df_features['SEP_event_label'].values
        
        if use_model_wise:
            print(f"Training model-wise {calibration_method} calibrators...")
            model_names = self.df_features['model_short'].values
            
            calibrator = ModelWiseCalibrator(method=calibration_method)
            calibrator.fit(y_prob, y_true, model_names)
            
            return {'model_wise': calibrator}
        else:
            print(f"Training global {calibration_method} calibrator...")
            calibrator = ProbabilityCalibrator(method=calibration_method)
            calibrator.fit(y_prob, y_true)
            
            return {'global': calibrator}
    
    def train_ensemble_model(self,
                            calibrators: Optional[Dict] = None,
                            ensemble_method: str = 'logistic') -> EnsembleMetaLearner:
        """
        Train ensemble meta-learner.
        
        Args:
            calibrators: Optional dictionary of calibrators to use
            ensemble_method: Method for ensemble ('logistic', 'gradient_boosting', 'weighted_average')
            
        Returns:
            Trained ensemble model
        """
        print("\n" + "="*60)
        print("STEP 4: Training Ensemble Model")
        print("="*60)
        
        # Create ensemble features
        df_ensemble = create_ensemble_features(self.df_pivoted.copy(), self.model_cols)
        
        # Optionally calibrate model predictions first
        if calibrators:
            print("Applying calibration to model predictions...")
            if 'model_wise' in calibrators:
                calibrator = calibrators['model_wise']
                for model_col in self.model_cols:
                    df_ensemble[model_col] = calibrator.predict_proba(
                        df_ensemble[model_col].values,
                        np.array([model_col] * len(df_ensemble))
                    )
            elif 'global' in calibrators:
                calibrator = calibrators['global']
                for model_col in self.model_cols:
                    df_ensemble[model_col] = calibrator.predict_proba(df_ensemble[model_col].values)
        
        # Prepare features
        feature_cols = self.model_cols + [col for col in df_ensemble.columns 
                                         if col.startswith('ensemble_')]
        X = df_ensemble[feature_cols].values
        y = df_ensemble['SEP_event_label'].values
        
        print(f"Training {ensemble_method} ensemble on {len(X)} samples with {len(feature_cols)} features...")
        print(f"  Positive samples: {y.sum()} ({y.mean():.2%})")
        
        ensemble = EnsembleMetaLearner(method=ensemble_method, class_weight='balanced')
        ensemble.fit(X, y)
        
        return ensemble, feature_cols
    
    def evaluate(self,
                calibrators: Optional[Dict] = None,
                ensemble: Optional[EnsembleMetaLearner] = None,
                feature_cols: Optional[list] = None,
                use_time_split: bool = True) -> Dict:
        """
        Evaluate models using cross-validation.
        
        Args:
            calibrators: Dictionary of calibrators
            ensemble: Trained ensemble model
            feature_cols: Feature columns for ensemble
            use_time_split: Whether to use temporal splitting
            
        Returns:
            Dictionary of evaluation results
        """
        print("\n" + "="*60)
        print("STEP 5: Evaluation")
        print("="*60)
        
        results = {}
        
        # Prepare data for evaluation
        y_prob_raw = self.df_features['prob_value'].values
        y_true = self.df_features['SEP_event_label'].values
        
        # Evaluate raw predictions
        print("\nEvaluating raw predictions...")
        metrics_raw = calculate_metrics(y_true, y_prob_raw, threshold=0.5)
        cal_metrics_raw = evaluate_calibration(y_true, y_prob_raw)
        metrics_raw.update(cal_metrics_raw)
        
        # Add optimal threshold info
        opt_thresh_raw, opt_f1_raw = find_optimal_threshold(y_true, y_prob_raw)
        metrics_raw['optimal_threshold'] = opt_thresh_raw
        metrics_raw['optimal_f1'] = opt_f1_raw
        
        results['raw'] = {
            'probabilities': y_prob_raw,
            'metrics': metrics_raw
        }
        print_metrics_report(metrics_raw, "Raw Predictions")
        
        # Evaluate calibrated predictions
        if calibrators:
            for name, calibrator in calibrators.items():
                print(f"\nEvaluating {name} calibrated predictions...")
                
                if name == 'model_wise':
                    model_names = self.df_features['model_short'].values
                    y_prob_cal = calibrator.predict_proba(y_prob_raw, model_names)
                else:
                    y_prob_cal = calibrator.predict_proba(y_prob_raw)
                
                metrics_cal = calculate_metrics(y_true, y_prob_cal, threshold=0.5)
                cal_metrics_cal = evaluate_calibration(y_true, y_prob_cal)
                metrics_cal.update(cal_metrics_cal)
                
                # Add optimal threshold info
                opt_thresh_cal, opt_f1_cal = find_optimal_threshold(y_true, y_prob_cal)
                metrics_cal['optimal_threshold'] = opt_thresh_cal
                metrics_cal['optimal_f1'] = opt_f1_cal
                
                results[f'{name}_calibrated'] = {
                    'probabilities': y_prob_cal,
                    'metrics': metrics_cal
                }
                print_metrics_report(metrics_cal, f"{name.capitalize()} Calibrated")
        
        # Evaluate ensemble
        if ensemble is not None and feature_cols is not None:
            print("\nEvaluating ensemble model...")
            
            # Prepare ensemble features
            df_ensemble = create_ensemble_features(self.df_pivoted.copy(), self.model_cols)
            X_ensemble = df_ensemble[feature_cols].values
            y_ensemble = df_ensemble['SEP_event_label'].values
            
            y_prob_ensemble = ensemble.predict_proba(X_ensemble)
            
            metrics_ensemble = calculate_metrics(y_ensemble, y_prob_ensemble, threshold=0.5)
            cal_metrics_ensemble = evaluate_calibration(y_ensemble, y_prob_ensemble)
            metrics_ensemble.update(cal_metrics_ensemble)
            
            # Add optimal threshold info
            opt_thresh_ens, opt_f1_ens = find_optimal_threshold(y_ensemble, y_prob_ensemble)
            metrics_ensemble['optimal_threshold'] = opt_thresh_ens
            metrics_ensemble['optimal_f1'] = opt_f1_ens
            
            results['ensemble'] = {
                'probabilities': y_prob_ensemble,
                'metrics': metrics_ensemble
            }
            print_metrics_report(metrics_ensemble, "Ensemble Model")
        
        # Create comparison plots
        print("\nCreating comparison plots...")
        predictions_dict = {name: res['probabilities'] for name, res in results.items()}
        
        if 'raw' in results:
            compare_predictions(
                y_true,
                {k: v for k, v in predictions_dict.items() if k in ['raw'] or 'calibrated' in k},
                save_path=str(self.output_dir / 'calibration_comparison.png')
            )
        
        return results
    
    def run_full_pipeline(self,
                         calibration_method: str = 'platt',
                         use_model_wise: bool = False,
                         ensemble_method: str = 'logistic',
                         use_ensemble: bool = True):
        """
        Run the full pipeline.
        
        Args:
            calibration_method: Calibration method to use
            use_model_wise: Whether to use model-wise calibration
            ensemble_method: Ensemble method to use
            use_ensemble: Whether to train ensemble model
        """
        # Step 1: Load and preprocess
        self.load_and_preprocess()
        
        # Step 2: Engineer features
        self.engineer_features()
        
        # Step 3: Train calibration models
        calibrators = self.train_calibration_models(
            calibration_method=calibration_method,
            use_model_wise=use_model_wise
        )
        
        # Step 4: Train ensemble (optional)
        ensemble = None
        feature_cols = None
        if use_ensemble:
            ensemble, feature_cols = self.train_ensemble_model(
                calibrators=calibrators,
                ensemble_method=ensemble_method
            )
        
        # Step 5: Evaluate
        results = self.evaluate(
            calibrators=calibrators,
            ensemble=ensemble,
            feature_cols=feature_cols
        )
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"Results saved to {self.output_dir}")
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='SEP Event Probability Calibration Pipeline')
    parser.add_argument('--data-path', type=str, default='data/combined_Labelled_modeldata.csv',
                       help='Path to data file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--calibration-method', type=str, default='platt',
                       choices=['platt', 'isotonic', 'temperature'],
                       help='Calibration method')
    parser.add_argument('--model-wise', action='store_true',
                       help='Use model-wise calibration')
    parser.add_argument('--ensemble-method', type=str, default='logistic',
                       choices=['logistic', 'gradient_boosting', 'weighted_average'],
                       help='Ensemble method')
    parser.add_argument('--no-ensemble', action='store_true',
                       help='Skip ensemble training')
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = SEPCalibrationPipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        random_state=42
    )
    
    pipeline.run_full_pipeline(
        calibration_method=args.calibration_method,
        use_model_wise=args.model_wise,
        ensemble_method=args.ensemble_method,
        use_ensemble=not args.no_ensemble
    )


if __name__ == "__main__":
    main()

