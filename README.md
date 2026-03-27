# Employee Attrition Prediction

End-to-end employee attrition modeling project using the IBM HR Analytics dataset.

This repository includes:
- A full Python pipeline for data loading, preprocessing, EDA, model training, hyperparameter tuning, and artifact export.
- A visualization notebook for exploratory analysis and model interpretation.

## Project Goals

- Predict whether an employee is likely to leave (binary classification).
- Understand key drivers of attrition through EDA and model outputs.
- Compare multiple supervised learning models with consistent evaluation metrics.
- Prioritize business-relevant performance for imbalanced data.

## Repository Structure

- `Employee Attrition.py`: complete script workflow from raw data to saved outputs.
- `Employee_Attrition_Visualizations.ipynb`: notebook with EDA, model evaluation plots, and interpretation sections.
- `README.md`: project documentation.

## Dataset

- Source: IBM HR Analytics Employee Attrition dataset.
- Typical shape: 1,470 rows and 35 columns.
- Target column: `Attrition` (`Yes`/`No`).

The script attempts to load data from the following locations in order:
1. `ibm_attrition.csv` in the script folder.
2. `WA_Fn-UseC_-HR-Employee-Attrition.csv` in the script folder.
3. `WA_Fn-UseC_-HR-Employee-Attrition.csv` in the Downloads folder.
4. `Employee Atrittion.zip` in the Downloads folder.

If a non-normalized source is loaded, the script writes a normalized local copy named `ibm_attrition.csv`.

## Environment and Dependencies

Required Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- scipy

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn scipy
```

## How to Run

From the repository root:

```bash
python "Employee Attrition.py"
```

The script runs non-interactively and saves charts/files under `employee_attrition_outputs`.

## End-to-End Workflow (Script)

### 1) Load Data

- Resolves dataset path from known candidates.
- Reads CSV (or ZIP-compressed CSV).
- Prints shape and target class distribution.

### 2) Encode Features

- Identifies categorical columns.
- Label-encodes all object-type features except target `Attrition`.
- Creates binary target: `Attrition_bin = 1` for `Yes`, else `0`.

### 3) Train/Test Split and Scaling

- Uses `train_test_split(..., stratify=y, test_size=0.2, random_state=42)`.
- Applies `StandardScaler`.

### 4) Handle Class Imbalance

- Applies `SMOTE(random_state=42)` on the training data only.
- Improves minority class representation for better recall on attrition cases.

### 5) Exploratory Data Analysis (EDA)

The script generates and saves:
- Original class imbalance chart.
- Before/after SMOTE class comparison.
- Correlation heatmap of key features.
- Attrition rate by overtime.
- Attrition rate by job satisfaction.
- Attrition rate by work-life balance.
- Boxplots: age, monthly income, years at company by attrition.
- Top absolute feature correlations with attrition.

### 6) Model Training and Hyperparameter Tuning

Models tuned with `RandomizedSearchCV`:
- Logistic Regression
- Random Forest
- SVM
- KNN
- Gaussian Naive Bayes

Cross-validation setup:
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`

Search objective:
- `scoring='roc_auc'`

Key tuned parameter spaces (summary):
- Logistic Regression: `C`, `solver`
- Random Forest: `n_estimators`, `max_depth`, `min_samples_split`
- SVM: `C`, `kernel`, `gamma`
- KNN: `n_neighbors`, `weights`, `metric`
- Naive Bayes: `var_smoothing`

### 7) Evaluation and Model Selection

For each model, script stores:
- CV best ROC-AUC
- Test ROC-AUC
- Accuracy
- Precision/Recall/F1 for positive class (`Yes`)
- Confusion matrix
- Predicted probabilities

Best model is selected by highest test ROC-AUC.

### 8) Save Artifacts

Output directory: `employee_attrition_outputs`

Saved files include:
- `model_summary.csv`
- `best_model_classification_report.csv`
- `best_model_predictions.csv`
- `model_auc_comparison.png`
- `roc_curves.png`
- `best_model_confusion_matrix.png`
- `run_summary.txt`
- EDA images `01_...png` to `10_...png`

## Notebook Walkthrough

The notebook complements the script with presentation-focused visuals and narrative.

It includes:
- EDA and correlation analysis.
- Model-by-model confusion matrices.
- Hyperparameter tuning with `RandomizedSearchCV`.
- Comparative metrics (Precision, Recall, F1, ROC-AUC).
- ROC curve and model interpretation sections.

## Metrics Used and Why

- Precision: reliability of positive predictions.
- Recall: ability to catch actual attrition cases.
- F1-score: balance between precision and recall.
- ROC-AUC: threshold-independent ranking quality.
- Confusion matrix: direct visibility into TP/TN/FP/FN errors.

Given class imbalance and HR intervention goals, recall and F1 are especially important alongside ROC-AUC.

## Reproducibility

Random seeds are fixed in core steps (`random_state=42`) for:
- train/test split
- SMOTE
- model tuning/search where applicable

## Practical Notes

- Correlation does not imply causation.
- Evaluate threshold strategy before deployment (business cost of FN vs FP).
- Check for leakage-prone identifier features before production training.
- Consider fairness analysis by employee segments in future iterations.

## Suggested Next Steps

- Add threshold optimization based on intervention costs.
- Add calibration analysis for probability quality.
- Add SHAP/permutation importance for stronger explainability.
- Add model monitoring/retraining policy for production.

## Author

Lavender Moraa
