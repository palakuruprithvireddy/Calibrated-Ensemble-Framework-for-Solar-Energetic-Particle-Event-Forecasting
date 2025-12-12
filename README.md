This work presents a unified probability calibration and ensemble modeling framework for forecasting Solar Energetic Particle (SEP) events using CCMC SEP Scoreboard data and solar-driver information. We first apply Platt Scaling to correct the conservative and poorly calibrated probability outputs from multiple physics-based models, improving reliability by 96% .

Next, we develop an interpretable ensemble meta-model that fuses calibrated predictions and ensemble statistics, showing that probability-based features account for 80% of total importance. Finally, we integrate CME and GOES physical drivers to enhance discrimination.

The resulting system produces reliable, physically informed, and actionable SEP event forecasts.

This codebase implements a three-stage pipeline:
1. **Data Loading and Preprocessing**: Loads SEP Scoreboard data and performs initial preprocessing
2. **Feature Engineering**: Creates temporal, model-specific, probability-based, and ensemble features
3. **Probability Calibration**: Applies Platt Scaling (or Isotonic/Temperature Scaling) to calibrate probabilities
4. **Ensemble Meta-Learning**: Uses Gradient Boosting or Logistic Regression to combine multiple model predictions
5. **Evaluation**: Comprehensive metrics including calibration (ECE, MCE), discrimination (ROC AUC, PR AUC), and classification metrics

Resource: https://ccmc.gsfc.nasa.gov/scoreboards/sep/#sep-scoreboard-and-data-access
