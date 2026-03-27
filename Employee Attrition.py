"""
CS 4630 - Week 11: Supervised Machine Learning
Employee Attrition Binary Classification
IBM HR Analytics Dataset (1,470 employees, 35 features)
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import zipfile
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, accuracy_score, ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / 'employee_attrition_outputs'
NORMALIZED_CSV_PATH = SCRIPT_DIR / 'ibm_attrition.csv'


sns.set_theme(style='whitegrid', palette='deep')


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)


def save_local_csv_copy(df: pd.DataFrame) -> None:
    df.to_csv(NORMALIZED_CSV_PATH, index=False)
    print(f"Saved CSV copy: {NORMALIZED_CSV_PATH}")


def save_eda_plots(df: pd.DataFrame, df_enc: pd.DataFrame, y_train: pd.Series, y_res: np.ndarray) -> None:
    ensure_output_dir()

    class_counts = df['Attrition'].value_counts().rename_axis('Attrition').reset_index(name='Count')
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=class_counts, x='Attrition', y='Count', hue='Attrition', dodge=False,
                     palette={'No': '#198038', 'Yes': '#DA1E28'})
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_title('Original Class Imbalance: Employee Attrition')
    ax.set_xlabel('Attrition')
    ax.set_ylabel('Number of Employees')
    for patch, count in zip(ax.patches, class_counts['Count']):
        ax.annotate(f'{count}',
                    (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                    ha='center', va='bottom', xytext=(0, 4), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_class_imbalance_original.png', dpi=300, bbox_inches='tight')
    plt.close()

    smote_counts = pd.DataFrame({
        'Stage': ['Before SMOTE', 'Before SMOTE', 'After SMOTE', 'After SMOTE'],
        'Class': ['No', 'Yes', 'No', 'Yes'],
        'Count': [int((y_train == 0).sum()), int((y_train == 1).sum()),
                  int((y_res == 0).sum()), int((y_res == 1).sum())],
    })
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=smote_counts, x='Stage', y='Count', hue='Class',
                     palette={'No': '#198038', 'Yes': '#DA1E28'})
    ax.set_title('Class Distribution Before and After SMOTE')
    ax.set_xlabel('Dataset Stage')
    ax.set_ylabel('Number of Employees')
    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_class_balance_after_smote.png', dpi=300, bbox_inches='tight')
    plt.close()

    corr_cols = ['Attrition_bin', 'Age', 'MonthlyIncome', 'JobSatisfaction',
                 'WorkLifeBalance', 'YearsAtCompany', 'TotalWorkingYears',
                 'DistanceFromHome', 'PercentSalaryHike']
    corr_matrix = df_enc[corr_cols].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title('Correlation Heatmap of Key Features')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    overtime_rates = (df.groupby('OverTime')['Attrition']
                        .apply(lambda values: (values == 'Yes').mean())
                        .reset_index(name='AttritionRate'))
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=overtime_rates, x='OverTime', y='AttritionRate', hue='OverTime',
                     dodge=False, palette={'No': '#0F62FE', 'Yes': '#8A3FFC'})
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_title('Attrition Rate by Overtime')
    ax.set_xlabel('Works Overtime')
    ax.set_ylabel('Attrition Rate')
    ax.set_ylim(0, max(0.4, overtime_rates['AttritionRate'].max() + 0.05))
    for patch, rate in zip(ax.patches, overtime_rates['AttritionRate']):
        ax.annotate(f'{rate:.1%}',
                    (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                    ha='center', va='bottom', xytext=(0, 4), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_attrition_by_overtime.png', dpi=300, bbox_inches='tight')
    plt.close()

    satisfaction_rates = (df.groupby('JobSatisfaction')['Attrition']
                            .apply(lambda values: (values == 'Yes').mean())
                            .reset_index(name='AttritionRate'))
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=satisfaction_rates, x='JobSatisfaction', y='AttritionRate',
                     hue='JobSatisfaction', dodge=False, palette='Blues_r')
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_title('Attrition Rate by Job Satisfaction')
    ax.set_xlabel('Job Satisfaction (1 = Low, 4 = High)')
    ax.set_ylabel('Attrition Rate')
    for patch, rate in zip(ax.patches, satisfaction_rates['AttritionRate']):
        ax.annotate(f'{rate:.1%}',
                    (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                    ha='center', va='bottom', xytext=(0, 4), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_attrition_by_job_satisfaction.png', dpi=300, bbox_inches='tight')
    plt.close()

    worklife_rates = (df.groupby('WorkLifeBalance')['Attrition']
                        .apply(lambda values: (values == 'Yes').mean())
                        .reset_index(name='AttritionRate'))
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=worklife_rates, x='WorkLifeBalance', y='AttritionRate',
                     hue='WorkLifeBalance', dodge=False, palette='viridis')
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_title('Attrition Rate by Work-Life Balance')
    ax.set_xlabel('Work-Life Balance (1 = Low, 4 = High)')
    ax.set_ylabel('Attrition Rate')
    for patch, rate in zip(ax.patches, worklife_rates['AttritionRate']):
        ax.annotate(f'{rate:.1%}',
                    (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                    ha='center', va='bottom', xytext=(0, 4), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_attrition_by_work_life_balance.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='Attrition', y='Age', hue='Attrition', dodge=False,
                palette={'No': '#198038', 'Yes': '#DA1E28'})
    plt.title('Age Distribution by Attrition')
    plt.xlabel('Attrition')
    plt.ylabel('Age')
    legend = plt.gca().legend_
    if legend is not None:
        legend.remove()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_age_boxplot_by_attrition.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='Attrition', y='MonthlyIncome', hue='Attrition', dodge=False,
                palette={'No': '#198038', 'Yes': '#DA1E28'})
    plt.title('Monthly Income by Attrition')
    plt.xlabel('Attrition')
    plt.ylabel('Monthly Income')
    legend = plt.gca().legend_
    if legend is not None:
        legend.remove()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08_monthly_income_boxplot_by_attrition.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='Attrition', y='YearsAtCompany', hue='Attrition', dodge=False,
                palette={'No': '#198038', 'Yes': '#DA1E28'})
    plt.title('Years at Company by Attrition')
    plt.xlabel('Attrition')
    plt.ylabel('Years at Company')
    legend = plt.gca().legend_
    if legend is not None:
        legend.remove()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '09_years_at_company_boxplot_by_attrition.png', dpi=300, bbox_inches='tight')
    plt.close()

    feature_target_corr = (df_enc.corr(numeric_only=True)['Attrition_bin']
                           .drop('Attrition_bin')
                           .abs()
                           .sort_values(ascending=False)
                           .head(10)
                           .sort_values())
    plt.figure(figsize=(9, 6))
    ax = feature_target_corr.plot(kind='barh', color='#0F62FE')
    ax.set_title('Top 10 Features Most Correlated with Attrition')
    ax.set_xlabel('Absolute Correlation with Attrition')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '10_top_feature_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved EDA plots to: {OUTPUT_DIR}")


def save_results_files(results: dict, best_name: str, y_test: pd.Series) -> None:
    ensure_output_dir()

    summary_rows = []
    for model_name, metrics in results.items():
        summary_rows.append({
            'model': model_name,
            'cv_auc': metrics['cv_auc'],
            'test_auc': metrics['test_auc'],
            'accuracy': metrics['accuracy'],
            'f1_yes': metrics['f1_yes'],
            'recall_yes': metrics['recall_yes'],
            'precision_yes': metrics['precision_yes'],
        })

    summary_df = pd.DataFrame(summary_rows).sort_values('test_auc', ascending=False)
    summary_path = OUTPUT_DIR / 'model_summary.csv'
    summary_df.to_csv(summary_path, index=False)

    best_report = pd.DataFrame(results[best_name]['report']).transpose()
    report_path = OUTPUT_DIR / 'best_model_classification_report.csv'
    best_report.to_csv(report_path)

    predictions_df = pd.DataFrame({
        'actual': y_test.to_numpy(),
        'predicted': results[best_name]['y_pred'],
        'predicted_probability_yes': results[best_name]['yp_prob'],
    })
    predictions_path = OUTPUT_DIR / 'best_model_predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary_df, x='test_auc', y='model', palette='Blues_r')
    plt.title('Model ROC-AUC Comparison')
    plt.xlabel('Test ROC-AUC')
    plt.ylabel('Model')
    plt.xlim(0, 1)
    plt.tight_layout()
    auc_plot_path = OUTPUT_DIR / 'model_auc_comparison.png'
    plt.savefig(auc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    for model_name, metrics in results.items():
        fpr, tpr, _ = roc_curve(y_test, metrics['yp_prob'])
        plt.plot(fpr, tpr, linewidth=2,
                 label=f"{model_name} (AUC={metrics['test_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    plt.title('ROC Curves by Model')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    roc_plot_path = OUTPUT_DIR / 'roc_curves.png'
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(results[best_name]['cm'], display_labels=['No', 'Yes']).plot(
        cmap='Blues', colorbar=False, ax=ax
    )
    ax.set_title(f'{best_name} Confusion Matrix')
    fig.tight_layout()
    cm_plot_path = OUTPUT_DIR / 'best_model_confusion_matrix.png'
    fig.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    summary_txt_path = OUTPUT_DIR / 'run_summary.txt'
    with summary_txt_path.open('w', encoding='ascii', errors='ignore') as summary_file:
        summary_file.write(f"Best model: {best_name}\n")
        summary_file.write(f"Best ROC-AUC: {results[best_name]['test_auc']:.4f}\n\n")
        summary_file.write(summary_df.to_string(index=False))
        summary_file.write('\n')

    print(f"Saved results to: {OUTPUT_DIR}")


def load_employee_attrition_data() -> pd.DataFrame:
    downloads_dir = Path.home() / 'Downloads'

    candidate_paths = [
        NORMALIZED_CSV_PATH,
        SCRIPT_DIR / 'WA_Fn-UseC_-HR-Employee-Attrition.csv',
        downloads_dir / 'WA_Fn-UseC_-HR-Employee-Attrition.csv',
        downloads_dir / 'Employee Atrittion.zip',
    ]

    for path in candidate_paths:
        if path.exists():
            if path.suffix.lower() == '.zip':
                print(f"Loading data from ZIP: {path}")
                with zipfile.ZipFile(path) as zip_file:
                    csv_members = [name for name in zip_file.namelist() if name.lower().endswith('.csv')]
                    if not csv_members:
                        raise FileNotFoundError(f"No CSV file found inside ZIP: {path}")
                df = pd.read_csv(path, compression='zip')
                save_local_csv_copy(df)
                return df

            print(f"Loading data from CSV: {path}")
            df = pd.read_csv(path)
            if path != NORMALIZED_CSV_PATH:
                save_local_csv_copy(df)
            return df

    searched_paths = '\n'.join(str(path) for path in candidate_paths)
    raise FileNotFoundError(f"Employee attrition dataset not found. Searched:\n{searched_paths}")

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
df = load_employee_attrition_data()
print(f"Shape: {df.shape}\nClass distribution:\n{df['Attrition'].value_counts()}\n")

# ── 2. ENCODE ─────────────────────────────────────────────────────────────────
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols.remove('Attrition')
df_enc = df.copy()
for c in cat_cols:
    df_enc[c] = LabelEncoder().fit_transform(df[c])
df_enc['Attrition_bin'] = (df['Attrition'] == 'Yes').astype(int)

feature_cols = [c for c in df_enc.columns if c not in ['Attrition','Attrition_bin']]
X = df_enc[feature_cols]
y = df_enc['Attrition_bin']

# ── 3. TRAIN/TEST SPLIT + SMOTE ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_te  = scaler.transform(X_test)

X_res, y_res = SMOTE(random_state=42).fit_resample(X_tr, y_train)
print(f"After SMOTE: {np.bincount(y_res)}\n")

# ── 4. EDA PLOTS ─────────────────────────────────────────────────────────────
save_eda_plots(df, df_enc, y_train, y_res)

# ── 5. MODELS + RANDOMIZEDSEARCHCV ───────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

configs = {
    'Logistic Regression': {
        'est': LogisticRegression(max_iter=1000, random_state=42),
        'params': {'C': uniform(0.01, 10), 'solver': ['lbfgs','liblinear']}
    },
    'Random Forest': {
        'est': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {'n_estimators': randint(50,200), 'max_depth': randint(3,20),
                   'min_samples_split': randint(2,10)}
    },
    'SVM': {
        'est': SVC(probability=True, random_state=42),
        'params': {'C': uniform(0.1,10), 'kernel': ['rbf','linear'], 'gamma': ['scale','auto']}
    },
    'KNN': {
        'est': KNeighborsClassifier(),
        'params': {'n_neighbors': randint(3,15), 'weights': ['uniform','distance'],
                   'metric': ['euclidean','manhattan']}
    },
    'Naive Bayes': {
        'est': GaussianNB(),
        'params': {'var_smoothing': uniform(1e-10, 1e-7)}
    },
}

results = {}
for name, cfg in configs.items():
    rs = RandomizedSearchCV(cfg['est'], cfg['params'], n_iter=20, scoring='roc_auc',
                            cv=cv, random_state=42, n_jobs=-1)
    rs.fit(X_res, y_res)
    best = rs.best_estimator_
    yp      = best.predict(X_te)
    yp_prob = best.predict_proba(X_te)[:,1]
    rep     = classification_report(y_test, yp, target_names=['No','Yes'], output_dict=True)
    results[name] = {
        'cv_auc': rs.best_score_, 'test_auc': roc_auc_score(y_test, yp_prob),
        'accuracy': accuracy_score(y_test, yp),
        'f1_yes': rep['Yes']['f1-score'], 'recall_yes': rep['Yes']['recall'],
        'precision_yes': rep['Yes']['precision'],
        'cm': confusion_matrix(y_test, yp), 'yp_prob': yp_prob,
        'y_pred': yp, 'report': rep,
    }
    print(f"{name:22s} AUC={results[name]['test_auc']:.4f}  "
          f"F1(Yes)={results[name]['f1_yes']:.4f}  "
          f"Recall(Yes)={results[name]['recall_yes']:.4f}")

# ── 6. PRINT FULL REPORT ─────────────────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]['test_auc'])
print(f"\n🏆 Best model: {best_name}  (ROC-AUC={results[best_name]['test_auc']:.4f})")

# ── 7. SAVE OUTPUT FILES ─────────────────────────────────────────────────────
save_results_files(results, best_name, y_test)