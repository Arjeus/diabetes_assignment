# %%

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)

from sklearn.utils import resample
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
sns.set(style='whitegrid')

# %% [markdown]
#   ## 1. Load raw data

# %%

DATA_DIR = Path('/home/arjay55/code/datasets/diabetes+130-us+hospitals+for+years+1999-2008')  # change if files are elsewhere
df = pd.read_csv(DATA_DIR / 'diabetic_data.csv')
ids_map = pd.read_csv(DATA_DIR / 'IDS_mapping.csv')
print(f'Data shape: {df.shape}')
df.head()

# %%
# Print columns by data type
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
integer_cols = df.select_dtypes(include=['int64', 'int32']).columns

print("Categorical/Object columns:")
print(f"Count: {len(categorical_cols)}")
print(categorical_cols.tolist())

print("\nInteger columns:")
print(f"Count: {len(integer_cols)}")
print(integer_cols.tolist())

print(f"\nTotal columns analyzed: {len(categorical_cols) + len(integer_cols)}")
print(f"DataFrame shape: {df.shape}")

# %%
# convert ['encounter_id', 'patient_nbr', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id'] to category
categorical_cols = ['encounter_id', 'patient_nbr', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id']
df[categorical_cols] = df[categorical_cols].astype('category')

categorical_cols_rest = df.select_dtypes(include=['object', 'category']).columns
# convert rest of the categorical columns to category
df[categorical_cols_rest] = df[categorical_cols_rest].astype('category')

# %%
def encode_med_change(x):
    """
    Simple ordinal encoder for medication‐change flags. No translates to zero as there is no drug. 
    Down can have the value of 1 as as the probability of relatively lower dosage than is more likely., 2 for steady meaning the drugs are normal,
    3 for up as the probability of relatively higher dosage than is more likely.
    
    Maps:
      "No"     → 0.0
      "Down"   → 1.0
      "Steady" → 2.0
      "Up"     → 3.0
    
    Anything else → np.nan
    """
    mapping = {
        "no":      0.0,
        "down":    1.0,
        "steady":  2.0,
        "up":      3.0,
    }
    # normalize to lower‐case string, then lookup
    return mapping.get(str(x).strip().lower(), np.nan)

# %%
# Apply medication change encoding to all medication columns
medication_cols = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 
    'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 
    'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 
    'metformin-pioglitazone'
]

for col in medication_cols:
    df[col] = df[col].apply(encode_med_change)

print(f"Applied medication change encoding to {len(medication_cols)} columns")
print("Sample encoded values:")
print(df[medication_cols[:5]].head())
   

# %%
# Drop weight as 97% have missing weights and drop impossible genders
df = df[df['gender'] != 'Unknown/Invalid'].copy()
df.drop(columns=['weight'], inplace=True)
freq = df["patient_nbr"].value_counts(normalize=True) # Calculate frequency of each patient. More frequent patients are more likely to have chronic conditions.
df["patient_freq"] = df["patient_nbr"].map(freq)

# Drop patient_nbr as it is not useful for modeling anymore
df.drop(columns=['patient_nbr'], inplace=True)

# Replace '?' with 'Unknown'
categorical_cols = df.select_dtypes(include='object').columns
df[categorical_cols] = df[categorical_cols].replace('?', 'Unknown')

# Remove encounters with discharge disposition indicating death/hospice
hospice_codes = [11, 19, 20, 21]
df = df[~df['discharge_disposition_id'].isin(hospice_codes)]


# %%
# 2. Compute proportions
freq = df['discharge_disposition_id'].value_counts(normalize=True)

# 3. Select “major” IDs (≥1% of all records)
major_ids = set(freq[freq >= 0.01].index)

# 4. Map to reduced categories
def bucket_disp(x):
    return x if x in major_ids else 'Other'

df['disch_reduced'] = df['discharge_disposition_id'].apply(bucket_disp)
# df_pt['disch_reduced'] = df_pt['discharge_disposition_id'].apply(bucket_disp)

# 5. Drop original column
df.drop(columns=['discharge_disposition_id'], inplace=True)
# 6. Dummify the reduced column
df = pd.get_dummies(df, columns=['disch_reduced'], drop_first=True)

# %% [markdown]
#   ### 2.1 Map admission/disposition/source IDs
#   * Translates IDs to descriptions for easier analysis

# %%
# Create mapping for admission_type_id only (since that's what we have)
def build_mapping_from_df(df_map):
    # Remove any rows with NaN values
    df_clean = df_map.dropna()
    return dict(zip(df_clean['admission_type_id'], df_clean['description']))

# Build the admission type mapping
admission_type_mapping = build_mapping_from_df(ids_map)

original_dtype = df['admission_type_id'].dtype

# Create new mapping with converted keys
if original_dtype in ['int64', 'int32', 'float64', 'category']:
    # Convert string keys to numeric
    admission_type_mapping_fixed = {
        int(k): v for k, v in admission_type_mapping.items() 
        if k.isdigit()
    }
else:
    # Keep as strings
    admission_type_mapping_fixed = admission_type_mapping

# Apply the mapping
df['admission_type_id'] = df['admission_type_id'].map(admission_type_mapping_fixed)
# df_pt['admission_type_id'] = df_pt['admission_type_id'].map(admission_type_mapping_fixed).fillna('Other')
print("After applying mapping with converted keys:")
print(df['admission_type_id'].value_counts())

# %% [markdown]
#   ### 2.2 Aggregate ICD‑9 diagnosis codes
# * First if statement are focused on internal, coronary and diabetic diseases, which could have comorbidities with each other, and thus we choose to make this detailed.
# * Other diseases are grouped, as they can have of less influence.

# %%

def diag_category(icd):
    try:
        icd = str(icd)
        code = icd.split('.')[0]  # take 3‑digit root
        if code.startswith('V') or code.startswith('E'):
            return 'Other'
        code = int(code)
    except:
        return 'Other'
    if 390 <= code <= 459 or code == 785 or 460 <= code <= 519 or code == 786 or 520 <= code <= 579 or code == 787 or 250 <= code <= 251:
        return f'icd_{code}'  # will result to very sparse categories
    if 800 <= code <= 999:
        return 'Injury'
    if 710 <= code <= 739:
        return 'Musculoskeletal'
    if 140 <= code <= 239:
        return 'Neoplasms'
    if 580 <= code <= 629 or code == 788:
        return 'Genitourinary'
    return 'Other'

for col in ['diag_1', 'diag_2', 'diag_3']:
    df[f'{col}_cat'] = df[col].apply(diag_category)
    # df_pt[f'{col}_cat'] = df_pt[col].apply(diag_category)

df.drop(columns=['diag_1','diag_2','diag_3'], inplace=True)
# df_pt.drop(columns=['diag_1','diag_2','diag_3'], inplace=True)

# %% [markdown]
# encounter_id has no relevance in the study

# %%

df.drop(columns=['encounter_id'], inplace=True, errors='ignore')

# %%
df_pt = df.copy() # for Pytorch Tabular

# %% [markdown]
#   ## 3. Train‑test split & preprocessing

# %%
def clean_column_name(col_name):
    """Clean column names by removing special characters that XGBoost doesn't allow"""
    return str(col_name).replace('[', '_').replace(']', '_').replace('<', '_lt_').replace('>', '_gt_').replace(',', '_')

# %%

y = (df['readmitted'] == '<30').astype(int)
X = df.drop(columns=['readmitted'])

X_train, X_test, y_train, y_test = train_test_split( #stratified sampling
    X, y, test_size=0.3, stratify=y, random_state=1803)

print('Train size:', X_train.shape, 'Pos rate:', y_train.mean().round(3))
print('Test size:', X_test.shape, 'Pos rate:', y_test.mean().round(3))

# Fix column names to remove special characters that XGBoost doesn't allow
X_train.columns = [clean_column_name(col) for col in X_train.columns]
X_test.columns = [clean_column_name(col) for col in X_test.columns]

# %% [markdown]
#   ### 3.1 Balance training set by random oversampling

# %%

train = pd.concat([X_train, y_train], axis=1)
maj = train[train['readmitted']==0]
minu = train[train['readmitted']==1]
minu_upsampled = resample(minu, replace=True, n_samples=len(maj), random_state=1803)
train_bal = pd.concat([maj, minu_upsampled])
X_train_bal = train_bal.drop(columns=['readmitted'])
y_train_bal = train_bal['readmitted']
print('Balanced class counts:', y_train_bal.value_counts())

# %% [markdown]
#   ### 3.2 One‑hot encode categorical variables

# %%

# categorical features ("object" dtype) are dummified, meaning they are converted to one-hot encoded columns.
cat_feats = X_train_bal.select_dtypes(include=['object','category']).columns
X_train_bal_enc = pd.get_dummies(X_train_bal, columns=cat_feats, drop_first=True) # reduce collinearity
X_test_enc = pd.get_dummies(X_test, columns=cat_feats, drop_first=True) # reduce collinearity
X_train_bal_enc, X_test_enc = X_train_bal_enc.align(X_test_enc, join='left', axis=1, fill_value=0)

## haircut for it to be compatible with XGBoost
X_train_bal_enc.columns = [clean_column_name(col) for col in X_train_bal_enc.columns]
X_test_enc.columns = [clean_column_name(col) for col in X_test_enc.columns]

# Apply standard scaling to numeric features
num_feats = X_train_bal_enc.select_dtypes(include=['int64','float64']).columns
scaler = StandardScaler()
X_train_bal_enc[num_feats] = scaler.fit_transform(X_train_bal_enc[num_feats])
X_test_enc[num_feats] = scaler.transform(X_test_enc[num_feats])

print("Feature engineering for baseline runs completed.")
# Dummify categorical variables for X_train and X_test

print("Creating dummy variables for training and test sets for pipeline use...")

# Get categorical columns
cat_cols = X_train.select_dtypes(include=['object','category']).columns
print(f"Categorical columns to encode: {list(cat_cols)}")



# Dummify X_train and X_test
X_train = pd.get_dummies(X_train, columns=cat_feats, drop_first=True)
X_test = pd.get_dummies(X_test, columns=cat_feats, drop_first=True)

scaler = StandardScaler() # normalize!
X_train[num_feats] = scaler.fit_transform(X_train[num_feats])
X_test[num_feats] = scaler.transform(X_test[num_feats])
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Clean column names for XGBoost compatibility
X_train.columns = [clean_column_name(col) for col in X_train.columns]
X_test.columns = [clean_column_name(col) for col in X_test.columns]

# %% [markdown]
#   ## 4. Model training

# %%

print("Initializing models...")
logreg = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=1803)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1803)
xgb = XGBClassifier(n_estimators=100, max_depth=6, eval_metric='logloss',
                    use_label_encoder=False, verbosity=0, random_state=1803)

print("Training Logistic Regression...")
logreg.fit(X_train_bal_enc, y_train_bal)
print("Training Random Forest...")
rf.fit(X_train_bal_enc, y_train_bal)
print("Training XGBoost...")
xgb.fit(X_train_bal_enc, y_train_bal)

# %%

def eval_model(name, model):
    y_pred = model.predict(X_test_enc)
    y_prob = model.predict_proba(X_test_enc)[:,1]
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name:20} Precision: {prec:.3f} Recall: {rec:.3f} F1: {f1:.3f} ROC-AUC: {auc:.3f} Accuracy: {acc:.3f}")
    return y_pred

preds = {}
preds['Logistic'] = eval_model('Logistic Regression', logreg)
preds['RandomForest'] = eval_model('Random Forest', rf)
preds['XGBoost'] = eval_model('XGBoost', xgb)

# %% [markdown]
# * Results show suboptimal performance. The class imbalance is significant, due to small positivity rate of 0.113.
# * We will proceed with XGBOOST due to its versatility and a go-to algorithm for tabular data.
# * We will use optuna as a hyperparameter tuning tool, a generic hyperparameter tuning framework.

# %%
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline


# Create pipeline with proper order: preprocess -> balance -> model (with Optuna parameters)
def create_model_pipeline(trial=None):
    
    # If trial is provided, optimize hyperparameters
    if trial is not None:
        # Optuna hyperparameter suggestions for XGBoost
        n_estimators = trial.suggest_int('n_estimators', 50, 1000)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
        reg_alpha = trial.suggest_float('reg_alpha', 0, 1.0)
        reg_lambda = trial.suggest_float('reg_lambda', 0, 1.0)
        
        classifier = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0,
            random_state=1803,
            n_jobs=10  # Use all available cores
        )
    else:
        # Use default/best known parameters for XGBoost
        classifier = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', use_label_encoder=False,
            verbosity=0, random_state=1803, scale_pos_weight=1.0
        )
    rng = np.random.default_rng()
    pipeline = ImbPipeline([ 
        ('balancer', SMOTE(random_state=int(rng.integers(2**16)))), # pipeline performs oversampling per each fold, avoiding data leakage
        ('classifier', classifier)
    ])
    return pipeline

# Optuna objective function
def objective(trial): 
    # Create pipeline with trial parameters
    pipeline = create_model_pipeline(trial)
    
    # Cross-validation with proper data handling

    cv_scores = cross_val_score(
        pipeline, X_train, y_train, 
        cv=4,  
        scoring='accuracy',
        n_jobs=1  # Reduced to prevent system overload
    )
    
    return cv_scores.mean()

# Run Optuna optimization
print("Starting Optuna hyperparameter optimization...")
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=1803)
)

# Optimize with progress callback
# def callback(study, trial):
#     if trial.number % 5 == 0:
#         print(f"Trial {trial.number}: Best value = {study.best_value:.4f}")

study.optimize(
    objective, 
    n_trials=2,
    # callbacks=[callback],
    show_progress_bar=True,
)

# Print optimization results
print(f"\nOptimization completed!")
print(f"Best parameters: {study.best_params}")
print(f"Best CV accuracy score: {study.best_value:.4f}")

# Create final pipeline with best parameters
print("\nTraining final model with best parameters...")
best_pipeline = create_model_pipeline()

# Update the classifier with best parameters from Optuna
best_pipeline.named_steps['classifier'].set_params(**study.best_params)

# Cross-validation with best parameters
final_cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"Final CV accuracy score: {final_cv_scores.mean():.3f} ± {final_cv_scores.std():.3f}")

# %%
print("\nTraining final model with best parameters...")
best_pipeline = create_model_pipeline()

# Update the classifier with best parameters from Optuna
best_pipeline.named_steps['classifier'].set_params(**study.best_params,random_state=1803)

# Cross-validation with best parameters
final_cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"Final CV accuracy score: {final_cv_scores.mean():.3f} ± {final_cv_scores.std():.3f}")

# %%
# Train the final model on all training data
best_pipeline.fit(X_train, y_train)

# %%
# Evaluate on test set
y_pred_test = best_pipeline.predict(X_test)
y_prob_test = best_pipeline.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)
test_auc = roc_auc_score(y_test, y_prob_test)

print(f"\nFinal Test Performance:")
print(f"Accuracy: {test_accuracy:.3f}")
print(f"F1: {test_f1:.3f}")
print(f"Precision: {test_precision:.3f}")
print(f"Recall: {test_recall:.3f}")
print(f"ROC-AUC: {test_auc:.3f}")

# Save the best model for later use
best_rf_optimized = best_pipeline.named_steps['classifier']

# %% [markdown]
# * Accuracy was well achieved. However the lack of data for readmissions resulted in highly skewed result plus some other modeling imperfections. The model is not yet safe for deployment. A higher recall is better. Class weighting that biases on readmission rates will be better.

# %%

# Create confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Readmission', '30-day Readmission'], 
            yticklabels=['No Readmission', '30-day Readmission'])
plt.title('Confusion Matrix - Optimized Random Forest')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
# save plot
plt.savefig('confusion_matrix_optimized_rf.png')

# %%
print(f"\nConfusion Matrix:")
print(f"True Negatives: {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives: {cm[1,1]}")

# %%
study.best_params

# %% [markdown]
# # PyTorch Tabular Implementation
# 
# Now we'll implement the same training pipeline using PyTorch Tabular with neural networks instead of XGBoost. Pytorch Tabular aims to implement suitable neural network architectures for tabular data with ease of use in using other popular frameworks, like Pandas.

# %%
# Import PyTorch Tabular
import torch
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

print("PyTorch Tabular imported successfully")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# %%
# Prepare data for PyTorch Tabular
# We'll use the same train/test split as XGBoost but with different preprocessing
print("Preparing data for PyTorch Tabular...")

# Start with the original X_train, X_test, y_train, y_test
# Reset from the original data before one-hot encoding
y = (df_pt['readmitted'] == '<30').astype(int)
X = df_pt.drop(columns=['readmitted'])

X_train_pt, X_test_pt, y_train_pt, y_test_pt = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=1803)

print('PyTorch Tabular - Train size:', X_train_pt.shape, 'Pos rate:', y_train_pt.mean().round(3))
print('PyTorch Tabular - Test size:', X_test_pt.shape, 'Pos rate:', y_test_pt.mean().round(3))

# Clean column names
X_train_pt.columns = [clean_column_name(col) for col in X_train_pt.columns]
X_test_pt.columns = [clean_column_name(col) for col in X_test_pt.columns]

train_df_pt = X_train_pt.copy()
train_df_pt["target"] = y_train_pt.values
print("Data prepared for PyTorch Tabular")

# %%
# Balance training set by random oversampling (same as XGBoost)
print("Balancing training data...")

maj_pt = train_df_pt[train_df_pt['target']==0]
minu_pt = train_df_pt[train_df_pt['target']==1]
minu_upsampled_pt = resample(minu_pt, replace=True, n_samples=len(maj_pt), random_state=1803)
train_bal_df_pt = pd.concat([maj_pt, minu_upsampled_pt])

print('Balanced class counts for PyTorch Tabular:', train_bal_df_pt['target'].value_counts())
print('Balanced training set shape:', train_bal_df_pt.shape)

# %%
# Define categorical and numerical columns for PyTorch Tabular
categorical_cols_pt = [col for col in X_train_pt.columns if X_train_pt[col].dtype == 'object' or X_train_pt[col].dtype.name == 'category']
numerical_cols_pt = [col for col in X_train_pt.columns if X_train_pt[col].dtype != 'object' and X_train_pt[col].dtype.name != 'category']

print(f"Categorical columns ({len(categorical_cols_pt)}): {categorical_cols_pt[:5]}...")
print(f"Numerical columns ({len(numerical_cols_pt)}): {numerical_cols_pt[:5]}...")

# Make sure we have the correct categoricals
categorical_cols_pt = []
numerical_cols_pt = []

for col in X_train_pt.columns:
    if X_train_pt[col].dtype == 'object' or str(X_train_pt[col].dtype) == 'category':
        categorical_cols_pt.append(col)
    else:
        numerical_cols_pt.append(col)

# Dummify categorical columns for PyTorch Tabular
# X_train_pt = pd.get_dummies(X_train_pt, columns=categorical_cols_pt, drop_first=True)
# X_test_pt = pd.get_dummies(X_test_pt, columns=categorical_cols_pt, drop_first=True)

scaler = StandardScaler()
# Normalize numerical columns for PyTorch Tabular
X_train_pt[numerical_cols_pt] = scaler.fit_transform(X_train_pt[numerical_cols_pt])
X_test_pt[numerical_cols_pt] = scaler.transform(X_test_pt[numerical_cols_pt])
X_train_pt, X_test_pt = X_train_pt.align(X_test_pt, join='left', axis=1, fill_value=0)

# Add target column to create complete datasets
train_df_pt = X_train_pt.copy()
train_df_pt['target'] = y_train_pt.values

test_df_pt = X_test_pt.copy()
test_df_pt['target'] = y_test_pt.values

# Configure PyTorch Tabular Data Config
data_config = DataConfig(
    target=['target'],  # Target column
    continuous_cols=numerical_cols_pt,  # Numerical columns
    categorical_cols=categorical_cols_pt,  # Categorical columns
    normalize_continuous_features=True,  # Similar to StandardScaler
)

print("PyTorch Tabular Data Config created successfully")

# %% [markdown]
# * We will use CategoryEmbeddingModelConfig, where categorical data are transformed into high dimensional embeddings.
# * We will go through the process of trying the model incrementally prior to proceeding to hyperparameter tuning.

# %%
# Create baseline PyTorch Tabular model configuration
model_config = CategoryEmbeddingModelConfig(
    task="classification",
    layers="128-64-32",  # Neural network architecture
    activation="ReLU",
    dropout=0.1,
    use_batch_norm=True,  # Correct parameter name
    learning_rate=1e-3,
    seed=1803,
    loss="CrossEntropyLoss",  # Use binary cross-entropy for binary classification
)

trainer_config = TrainerConfig(
    batch_size=1024,
    max_epochs=50,
    early_stopping="valid_loss",
    early_stopping_patience=10,
    checkpoints=None,  # Disable checkpoints to avoid loading issues
    load_best=False,   # Don't try to load best model
    progress_bar="none",  # Disable progress bar for cleaner output
    auto_lr_find=False,  # We'll set learning rate manually
    auto_select_gpus=torch.cuda.is_available(),
    seed=1803,
)

optimizer_config = OptimizerConfig()

print("PyTorch Tabular configurations created successfully")

# %%
# Train baseline PyTorch Tabular model
print("Training baseline PyTorch Tabular model...")

baseline_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)

# Fit the model
baseline_model.fit(train=train_bal_df_pt, validation=test_df_pt)

print("Baseline PyTorch Tabular model training completed")

# %%
# Evaluate baseline PyTorch Tabular model
print("Evaluating baseline PyTorch Tabular model...")

# Make predictions
baseline_pred_proba = baseline_model.predict(test_df_pt)
print("Prediction output shape:", baseline_pred_proba.shape)
print("Prediction output columns:", baseline_pred_proba.columns.tolist())

# Get prediction probabilities - use the correct column name
if '1' in baseline_pred_proba.columns:
    baseline_proba_values = baseline_pred_proba['1'].values
elif '1_probability' in baseline_pred_proba.columns:
    baseline_proba_values = baseline_pred_proba['1_probability'].values
else:
    # Try the first numeric column after the identifier columns
    prob_cols = [col for col in baseline_pred_proba.columns if col not in ['patient_nbr', 'target']]
    baseline_proba_values = baseline_pred_proba[prob_cols[0]].values

baseline_pred = (baseline_proba_values > 0.5).astype(int)

# Calculate metrics
baseline_prec = precision_score(y_test_pt, baseline_pred)
baseline_rec = recall_score(y_test_pt, baseline_pred)
baseline_f1 = f1_score(y_test_pt, baseline_pred)
baseline_auc = roc_auc_score(y_test_pt, baseline_proba_values)

print(f"Baseline PyTorch Tabular Performance:")
print(f"Precision: {baseline_prec:.3f}")
print(f"Recall: {baseline_rec:.3f}")
print(f"F1: {baseline_f1:.3f}")
print(f"ROC-AUC: {baseline_auc:.3f}")
print (f"Accuracy: {accuracy_score(y_test_pt, baseline_pred):.3f}")

# %% [markdown]
# * Results are suboptimal, will proceed to k-fold validation

# %%
# Implement 4-fold cross-validation for PyTorch Tabular
from sklearn.model_selection import StratifiedKFold
import pickle
import tempfile
import os
from imblearn.over_sampling import RandomOverSampler
from pytorch_tabular.models.category_embedding.category_embedding_model import CategoryEmbeddingModel
import pytorch_tabular
import torch.nn as nn

# %%

class WeightedCategoryEmbeddingModel(CategoryEmbeddingModel):
    """CategoryEmbedding model with weighted loss function for class imbalance"""
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # Store class weights for loss function
        self.class_weights = getattr(config, 'class_weights', None)
        
    def loss(self, y_hat, y, tag="train"):
        """Override loss function to use class weights"""
        if self.class_weights is not None:
            # Use weighted CrossEntropyLoss
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(y_hat.device))
        else:
            # Use standard loss
            loss_fn = nn.CrossEntropyLoss()
            
        if self.hparams.task == "classification":
            computed_loss = loss_fn(y_hat, y.long())
        else:
            computed_loss = loss_fn(y_hat, y)
            
        self.log(
            f"{tag}_loss",
            computed_loss,
            on_epoch=(tag == "valid"),
            on_step=(tag == "train"),
            logger=True,
            prog_bar=True,
        )
        return computed_loss

# Register the model directly in the category_embedding_model module
import pytorch_tabular.models.category_embedding.category_embedding_model as ce_module
ce_module.WeightedCategoryEmbeddingModel = WeightedCategoryEmbeddingModel

# %%

# def pytorch_tabular_cv(X_data, y_data, n_folds=4, model_params=None):
#     """
#     Perform cross-validation for PyTorch Tabular model
#     """
#     skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1803)
#     cv_scores = []
    
#     # Combine X and y for easier handling
#     full_data = X_data.copy()
#     full_data['target'] = y_data.values
    
#     for fold, (train_idx, val_idx) in enumerate(skf.split(X_data, y_data)):
#         print(f"Training fold {fold + 1}/{n_folds}...")
        
#         # Split data
#         train_fold = full_data.iloc[train_idx]
#         val_fold = full_data.iloc[val_idx]
        
#         # Balance training fold
#         # Balance training fold using SMOTE from imbalanced-learn
        
#         X_train_fold = train_fold.drop('target', axis=1)
#         y_train_fold = train_fold['target']
        
#         randsamp = RandomOverSampler(random_state=1803+fold)
#         X_train_balanced, y_train_balanced = randsamp.fit_resample(X_train_fold, y_train_fold)
        
#         train_fold_balanced = X_train_balanced.copy()
#         train_fold_balanced['target'] = y_train_balanced
        
#         # Create model configuration
#         if model_params is None:
#             fold_model_config = CategoryEmbeddingModelConfig(
#                 task="classification",
#                 layers="128-64-32",
#                 activation="ReLU", 
#                 dropout=0.1,
#                 use_batch_norm=True,  # Fixed parameter name
#                 learning_rate=1e-4,
#                 seed=1803+fold,
#                 loss="CrossEntropyLoss",  # Use binary cross-entropy for binary classification
#             )
#         else:
#             fold_model_config = CategoryEmbeddingModelConfig(
#                 task="classification",
#                 **model_params,
#                 seed=1803+fold,
#                 loss="CrossEntropyLoss",  # Use binary cross-entropy for binary classification
#             )
        
#         fold_trainer_config = TrainerConfig(
#             batch_size=1024,
#             max_epochs=60,  # Reduced for CV
#             early_stopping="valid_loss",
#             early_stopping_patience=5,
#             checkpoints=None,  # Don't save checkpoints for CV
#             load_best=True,   # Don't try to load best model
#             progress_bar="none",
#             auto_lr_find=False,
#             auto_select_gpus=torch.cuda.is_available(),
#             seed=1803+fold,
#         )
        
#         # Create and train model
#         fold_model = TabularModel(
#             data_config=data_config,
#             model_config=fold_model_config,
#             optimizer_config=optimizer_config,
#             trainer_config=fold_trainer_config,
#         )
        
#         try:
#             # Train model
#             fold_model.fit(train=train_fold_balanced, validation=val_fold)
            
#             # Predict and evaluate
#             val_pred_proba = fold_model.predict(val_fold)
#             # Use correct column name for predictions
#             val_proba_values = val_pred_proba['target_1_probability'].values
#             val_pred = (val_proba_values > 0.5).astype(int)
#             val_accuracy = accuracy_score(val_fold['target'], val_pred)
#             cv_scores.append(val_accuracy)
            
#             print(f"Fold {fold + 1} accuracy score: {val_accuracy:.3f}")
            
#         except Exception as e:
#             print(f"Error in fold {fold + 1}: {e}")
#             cv_scores.append(0.0)  # Add poor score for failed fold
    
#     return cv_scores


# # %%

# # Perform 4-fold cross-validation with baseline model
# print("Performing 4-fold cross-validation with PyTorch Tabular...")
# cv_scores_pt = pytorch_tabular_cv(X_train_pt, y_train_pt, n_folds=4)

# print(f"\nPyTorch Tabular CV Results:")
# print(f"Mean Accuracy: {np.mean(cv_scores_pt):.3f} ± {np.std(cv_scores_pt):.3f}")
# print(f"Individual fold scores: {[f'{score:.3f}' for score in cv_scores_pt]}")

# %% [markdown]
# * Accuracies are suboptimal. Hopefully hyperparameter tuning will enhance this.
# * In this case, layer depths, activation, batch sizes, etc will be modified.

# %%
# Hyperparameter tuning for PyTorch Tabular using Optuna
import optuna

# Modify the pytorch_tabular_objective function to fix the _module_src parameter:
def pytorch_tabular_objective(trial):
    """Optuna objective function for PyTorch Tabular hyperparameter optimization"""
    # Suggest hyperparameters
    layers_depth = trial.suggest_int('layers_depth', 2, 4)
    layer_size = trial.suggest_categorical('layer_size', [512, 1024, 2048])
    
    # Create layer string
    if layers_depth == 2:
        layers = f"{layer_size}-{layer_size//2}"
    elif layers_depth == 3:
        layers = f"{layer_size}-{layer_size//2}-{layer_size//4}"
    else:  # layers_depth == 4
        layers = f"{layer_size}-{layer_size//2}-{layer_size//4}-{layer_size//8}"
    
    dropout = trial.suggest_float('dropout', 0.0, 0.2)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048, 4096])
    activation = trial.suggest_categorical('activation', ['ReLU', 'GELU', 'LeakyReLU'])
    
    # Choose balancing strategy
    balance_strategy = trial.suggest_categorical('balance_strategy', ['class_weights'])
    
    # Model parameters
    model_params = {
        'layers': layers,
        'activation': activation,
        'dropout': dropout,
        'use_batch_norm': True,
        'learning_rate': learning_rate,
    }
    
    rng = np.random.default_rng()
    init_state = int(rng.integers(2**16))
    cv_scores = []
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=init_state)
    
    # Combine X and y for easier handling
    full_data = X_train_pt.copy()
    full_data['target'] = y_train_pt.values

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_pt, y_train_pt)):
        # Split data
        train_fold = full_data.iloc[train_idx]
        val_fold = full_data.iloc[val_idx]
        
        # Use original imbalanced data
        train_fold_balanced = train_fold.copy()
        y_fold = train_fold['target']
        
        # Compute class weights if needed

        
        # Create standard model config - FIX HERE: Use CategoryEmbeddingModelConfig directly
    # Modified model config with correct module reference
        # Create standard model config with correct loss specification
        fold_model_config = CategoryEmbeddingModelConfig(
            task="classification",
            **model_params,
            seed=init_state+fold,
            # Specify loss as a string, not as an instance
            loss="CrossEntropyLoss"  # Keep this as a string
        )

        # Add class weights as a custom attribute
        class_weights = None
        if balance_strategy == 'class_weights':
            class_counts = np.bincount(y_fold)
            total_samples = len(y_fold)
            
            # Calculate inverse frequency weights
            weight_for_0 = total_samples / (2.0 * class_counts[0])
            weight_for_1 = total_samples / (2.0 * class_counts[1])
            class_weights = torch.tensor([weight_for_0, weight_for_1], dtype=torch.float32)
            print(f"Fold {fold + 1} - Class weights: {class_weights.tolist()}")
            
            # Don't set loss as an instance - just add class_weights as an attribute
            setattr(fold_model_config, 'class_weights', class_weights)

        # Add class weights as a custom attribute
        setattr(fold_model_config, 'class_weights', class_weights)
        
        fold_trainer_config = TrainerConfig(
            batch_size=batch_size,
            max_epochs=20,
            early_stopping="valid_loss",
            early_stopping_patience=3,
            checkpoints=None,
            load_best=False,
            progress_bar="none",
            auto_lr_find=False,
            auto_select_gpus=torch.cuda.is_available(),
            seed=1803+fold,
        )
        
        # Create and train model
        fold_model = TabularModel(
            data_config=data_config,
            model_config=fold_model_config,
            optimizer_config=optimizer_config,
            trainer_config=fold_trainer_config,
        )
        
        try:
            # Train model
            fold_model.fit(train=train_fold_balanced, validation=val_fold)
            
            # Predict and evaluate
            val_pred_proba = fold_model.predict(val_fold)
            val_proba_values = val_pred_proba['target_1_probability'].values
            val_pred = (val_proba_values > 0.5).astype(int)
            val_accuracy = accuracy_score(val_fold['target'], val_pred)
            
            # Also log F1 score for imbalanced data
            val_f1 = f1_score(val_fold['target'], val_pred)
            print(f"Fold {fold + 1} - Strategy: {balance_strategy}, Val Accuracy: {val_accuracy:.3f}, Val F1: {val_f1:.3f}")
            cv_scores.append(val_accuracy)
            
        except Exception as e:
            print(f"Error in fold {fold}: {e}")
            return 0.0
    
    return np.mean(cv_scores) if len(cv_scores) > 0 else 0.0

# Run Optuna optimization for PyTorch Tabular (smaller trial for speed)
print("Starting Optuna hyperparameter optimization for PyTorch Tabular...")
study_pt = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=1803)
)

study_pt.optimize(
    pytorch_tabular_objective, 
    n_trials=2,  
    show_progress_bar=True,
)

# Print optimization results
print(f"\nPyTorch Tabular Optimization completed!")
print(f"Best parameters: {study_pt.best_params}")
print(f"Best CV accuracy score: {study_pt.best_value:.4f}")

# %%
# Train final optimized PyTorch Tabular model
print("Training final optimized PyTorch Tabular model...")
# Extract best parameters
best_params_pt = study_pt.best_params
layers_depth = best_params_pt['layers_depth']
layer_size = best_params_pt['layer_size']

# Create layer string
if layers_depth == 2:
    layers = f"{layer_size}-{layer_size//2}"
elif layers_depth == 3:
    layers = f"{layer_size}-{layer_size//2}-{layer_size//4}"
else:  # layers_depth == 4
    layers = f"{layer_size}-{layer_size//2}-{layer_size//4}-{layer_size//8}"

# Create final model configuration with best parameters
final_model_config_pt = CategoryEmbeddingModelConfig(
    task="classification",
    layers=layers,
    activation=best_params_pt['activation'],
    dropout=best_params_pt['dropout'],
    use_batch_norm=True,
    learning_rate=best_params_pt['learning_rate'],
    seed=1803,
    loss="CrossEntropyLoss",  # Use cross entropy loss for classification
)

final_trainer_config_pt = TrainerConfig(
    batch_size=best_params_pt['batch_size'],
    max_epochs=50,  # Full epochs for final model
    early_stopping="valid_loss",
    early_stopping_patience=5,
    checkpoints="valid_loss",
    load_best=False,
    progress_bar="none",
    auto_lr_find=False,
    auto_select_gpus=torch.cuda.is_available(),
    seed=1803,
)

# Create final model
final_model_pt = TabularModel(
    data_config=data_config,
    model_config=final_model_config_pt,
    optimizer_config=optimizer_config,
    trainer_config=final_trainer_config_pt,
)

class_counts = np.bincount(y_train_pt)
y_fold = train_bal_df_pt['target'].values       
total_samples = len(y_fold)

# Calculate inverse frequency weights
weight_for_0 = total_samples / (2.0 * class_counts[0])
weight_for_1 = total_samples / (2.0 * class_counts[1])
class_weights = torch.tensor([weight_for_0, weight_for_1], dtype=torch.float32)
print(f"Complete train set class weights: {class_weights.tolist()}")

# Don't set loss as an instance - just add class_weights as an attribute
setattr(final_model_config_pt, 'class_weights', class_weights)

# Train final model
final_model_pt.fit(train=train_df_pt, validation=test_df_pt)

print("Final PyTorch Tabular model training completed")

# %%
# Final cross-validation with optimized parameters
# print("Performing final 4-fold cross-validation with optimized parameters...")

# # Extract best parameters for CV function
# best_model_params = {
#     'layers': layers,
#     'activation': best_params_pt['activation'],
#     'dropout': best_params_pt['dropout'],
#     'use_batch_norm': True,
#     'learning_rate': best_params_pt['learning_rate'],
# }

# # Perform final CV with optimized parameters
# final_cv_scores_pt = pytorch_tabular_cv(X_train_pt, y_train_pt, n_folds=4, model_params=best_model_params)

# print(f"\nFinal PyTorch Tabular CV Results (with optimization):")
# print(f"Mean Accuracy: {np.mean(final_cv_scores_pt):.3f} ± {np.std(final_cv_scores_pt):.3f}")
# print(f"Individual fold scores: {[f'{score:.3f}' for score in final_cv_scores_pt]}")

# Evaluate on test set
print("\nEvaluating final model on test set...")
final_pred_proba_pt = final_model_pt.predict(test_df_pt)
final_pred_pt = (final_pred_proba_pt['target_1_probability'].values > 0.5).astype(int)
final_accuracy_pt = accuracy_score(y_test_pt, final_pred_pt)
final_prec_pt = precision_score(y_test_pt, final_pred_pt)
final_rec_pt = recall_score(y_test_pt, final_pred_pt)
final_f1_pt = f1_score(y_test_pt, final_pred_pt)
final_auc_pt = roc_auc_score(y_test_pt, final_pred_proba_pt['target_1_probability'].values)

print(f"\nFinal PyTorch Tabular Test Performance:")
print(f"Accuracy: {final_accuracy_pt:.3f}")
print(f"Precision: {final_prec_pt:.3f}")
print(f"Recall: {final_rec_pt:.3f}")
print(f"F1: {final_f1_pt:.3f}")
print(f"ROC-AUC: {final_auc_pt:.3f}")

# %%
# Save the final PyTorch Tabular model
print("Saving PyTorch Tabular model...")

# Save the model
model_save_path = "best_pytorch_tabular_model"
final_model_pt.save_model(model_save_path)

# Also save the best hyperparameters
with open("best_pytorch_tabular_params.txt", "w") as f:
    f.write("Best PyTorch Tabular Hyperparameters:\n")
    f.write("="*50 + "\n")
    for key, value in study_pt.best_params.items():
        f.write(f"{key}: {value}\n")
    f.write(f"\nBest CV Accuracy Score: {study_pt.best_value:.4f}\n")
    f.write(f"Final Test Accuracy Score: {final_accuracy_pt:.4f}\n")
    f.write(f"Final Test F1 Score: {final_f1_pt:.4f}\n")
    f.write(f"Final Test ROC-AUC: {final_auc_pt:.4f}\n")

print("Model and parameters saved successfully")

# Create confusion matrix for PyTorch Tabular
cm_pt = confusion_matrix(y_test_pt, final_pred_pt)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_pt, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Readmission', '30-day Readmission'], 
            yticklabels=['No Readmission', '30-day Readmission'])
plt.title('Confusion Matrix - Optimized PyTorch Tabular')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
# save plot
plt.savefig('confusion_matrix_optimized_pytorch_tabular.png')

print(f"\nConfusion Matrix (PyTorch Tabular):")
print(f"True Negatives: {cm_pt[0,0]}")
print(f"False Positives: {cm_pt[0,1]}")
print(f"False Negatives: {cm_pt[1,0]}")
print(f"True Positives: {cm_pt[1,1]}")

# %% [markdown]
# * The results have slightly worse accuracy than in XGBOOST method. However, the higher recall makes it relatively safer to implement.

# %%
# Compare XGBoost and PyTorch Tabular results
print("="*60)
print("MODEL COMPARISON: XGBoost vs PyTorch Tabular")
print("="*60)

# print("\nCross-Validation Results (4-fold accuracy scores):")
# print(f"XGBoost CV Accuracy:         {final_cv_scores.mean():.3f} ± {final_cv_scores.std():.3f}")
# print(f"PyTorch Tabular CV Accuracy: {np.mean(final_cv_scores_pt):.3f} ± {np.std(final_cv_scores_pt):.3f}")

print(f"\nTest Set Performance:")
print(f"{'Metric':<15} {'XGBoost':<10} {'PyTorch Tabular':<15}")
print("-" * 40)
print(f"{'Accuracy':<15} {test_accuracy:.3f}      {final_accuracy_pt:.3f}")
print(f"{'Precision':<15} {test_precision:.3f}      {final_prec_pt:.3f}")
print(f"{'Recall':<15} {test_recall:.3f}      {final_rec_pt:.3f}")
print(f"{'F1':<15} {test_f1:.3f}      {final_f1_pt:.3f}")
print(f"{'ROC-AUC':<15} {test_auc:.3f}      {final_auc_pt:.3f}")

print(f"\nBest Hyperparameters:")
print(f"\nXGBoost:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
    
print(f"\nPyTorch Tabular:")
for key, value in study_pt.best_params.items():
    print(f"  {key}: {value}")

print("\n" + "="*60)

# %% [markdown]
# * While the accuracy is good, its low recall might make the model unsuitable for deployment yet.

# %% [markdown]
# ## Model interpretation
# * We will use SHAP, a model-agnostic technique to analyze feature contributions the the output.

# %%
# SHAP Analysis for XGBoost Model
print("Performing SHAP analysis on the optimized XGBoost model...")

# Get the XGBoost classifier from the best pipeline
xgb_model = best_pipeline.named_steps['classifier']

# Use a sample of training data for SHAP (to speed up computation)
sample_size = 5000
sample_indices = np.random.choice(X_test.shape[0], size=min(sample_size, X_test.shape[0]), replace=False)
X_sample = X_test.iloc[sample_indices]

# Create SHAP explainer for XGBoost
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_sample)

print(f"SHAP values computed for {len(X_sample)} samples")

# Summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.title('SHAP Feature Importance - XGBoost Model')
plt.tight_layout()
# save plot
plt.savefig('shap_summary_plot_xgboost.png')

# Detailed summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, show=False)
plt.title('SHAP Summary Plot - XGBoost Model')
plt.tight_layout()
# save plot
plt.savefig('shap_detailed_summary_plot_xgboost.png')

# Feature importance ranking
feature_importance = np.abs(shap_values).mean(0)
feature_names = X_sample.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features (SHAP):")
print(importance_df.head(15))

# %%
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss, make_scorer
from sklearn.base import BaseEstimator, ClassifierMixin

class ModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
        # Explicitly set estimator type
        self._estimator_type = "classifier"
        # Set classes_ attribute to indicate this is a binary classifier
        self.classes_ = np.array([0, 1])
    
    def predict(self, X):
        """Return class predictions (0 or 1)"""
        preds = self.model.predict(X)
        # Convert probabilities to binary class predictions
        return (preds["target_1_probability"].values > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Return probability predictions for both classes"""
        preds = self.model.predict(X)
        # Return probabilities for both classes
        prob_0 = preds["target_0_probability"].values
        prob_1 = preds["target_1_probability"].values
        return np.column_stack([prob_0, prob_1])

    def fit(self, X, y):
        # Required by sklearn but we don't need to do anything
        self.classes_ = np.unique(y)
        return self

def custom_log_loss(y_true, y_pred_proba):
    return -log_loss(y_true, y_pred_proba)

custom_scorer = make_scorer(custom_log_loss, response_method='predict', greater_is_better=True)

# Create the wrapper
wrapped_model = ModelWrapper(final_model_pt)

result = permutation_importance(wrapped_model, test_df_pt, y_test_pt, scoring='accuracy', n_repeats=20, random_state=1803)
importance_df = pd.DataFrame({
    'feature': train_df_pt.columns,
    'importance': result.importances_mean
}).sort_values(by='importance', ascending=False)

# %%
importance_df.head(15).plot(kind='barh', x='feature', y='importance', figsize=(12, 8), legend=False)
plt.title('Permutation Feature Importance - PyTorch Tabular Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
print("\nTop 15 Most Important Features (Permutation Importance - PyTorch Tabular):")
print(importance_df.head(15))
# %%
result_xgboost = permutation_importance(xgb_model, X_test, y_test, scoring='accuracy', n_repeats=10, random_state=1803)


# %%
importance_df_xgboost = pd.DataFrame({
    'feature': X_train.columns,
    'importance': result_xgboost.importances_mean
}).sort_values(by='importance', ascending=False)

print("\nTop 15 Most Important Features (Permutation Importance - XGBoost):")
print(importance_df_xgboost.head(15))

# %%
importance_df_xgboost.head(15).plot(kind='barh', x='feature', y='importance', figsize=(12, 8), legend=False)
plt.title('Permutation Feature Importance - XGBoost Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()

# %% [markdown]
# ### As of now we will not be implementing SHAP on the Pytorch Model as I am encountering compatibility issues between SHAP and Pytorch Tabular model.

# %%
# Reload the best PyTorch Tabular model
# print("Reloading the best PyTorch Tabular model...")

# # Load the saved model
# final_model_pt = TabularModel.load_model("best_pytorch_tabular_model")

# print("Model reloaded successfully!")

# %%
# train_df_pt_float = train_df_pt.astype(np.float32)
# # keep the feature order you trained with
# FEATURE_NAMES = train_df_pt_float.columns.tolist()

# def pytorch_tabular_predict_proba(X):
#     """
#     SHAP wrapper for PyTorch-Tabular.
#     Ensures the incoming object is a pandas.DataFrame
#     with the correct column names and dtypes.
#     Returns the positive-class probability reshaped for SHAP.
#     """

#     pred_proba = final_model_pt.predict(X)

#     # keep only P(y=1) and give SHAP a 2-D array
#     return pred_proba["target_1_probability"].to_numpy().reshape(-1, 1)

# # ───  A.  Summarise the background  ────────────────────────────────────────────

# dense_bg   = shap.kmeans(train_df_pt_float, k=100)             # DenseData (not callable)
# masker     = shap.maskers.Independent(dense_bg.data)    # <-- make it callable


# %%
# # ───  B.  Build the explainer  ────────────────────────────────────────────────
# explainer_pt = shap.Explainer(
#     pytorch_tabular_predict_proba,  # returns *logits*, see previous answer
#     masker,
#     link=shap.links.logit,                   # tell SHAP what the wrapper outputs
#     algorithm="permutation"         # same default the auto-chooser would pick
# )

# %%
# # ───  C.  Explain a subset  ───────────────────────────────────────────────────

# sample_size     = 500
# sample_indices  = np.random.choice(len(train_df_pt_float),
#                                    size=min(sample_size, len(train_df_pt_float)),
#                                    replace=False)

# X_sample_pt     = train_df_pt_float.iloc[sample_indices]
# shap_values_pt = explainer_pt(X_sample_pt, max_evals=1300)

# %%
# shap.summary_plot(shap_values_pt.values, features=X_sample_pt,
#                   feature_names=X_sample_pt.columns, show=False)

# %%
# # Feature importance ranking for PyTorch Tabular
# feature_importance_pt = np.abs(shap_values_pt).mean(0)
# feature_names_pt = X_sample_pt.columns
# importance_df_pt = pd.DataFrame({
#     'feature': feature_names_pt,
#     'importance': feature_importance_pt.ravel()
# }).sort_values('importance', ascending=False)

# print("\nTop 60 Most Important Features (SHAP - PyTorch Tabular):")
# importance_df_pt.head(60).to_csv('pytorch_tabular_shap_importance.csv', index=False)


