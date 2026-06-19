import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

EXPERIMENT_NAME = 'bank_marketing_campaign_prediction'
mlflow.set_experiment(EXPERIMENT_NAME)

dataset = fetch_openml(name='bank-marketing', version=1, as_frame=True)
df_bank = dataset.frame.copy()
df_bank.columns = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y']
df_bank['y'] = df_bank['y'].astype(str).replace({'1':'no','2':'yes'})
X_bank = df_bank.drop(columns=['y']).copy()
y_bank = df_bank['y'].map({'no':0,'yes':1}).astype(int)
num_cols = X_bank.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X_bank.select_dtypes(include=['object','category']).columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X_bank, y_bank, test_size=0.2, random_state=42, stratify=y_bank)
num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocess = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
models = {
    'logistic_regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'random_forest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced_subsample', n_jobs=-1),
    'gradient_boosting': GradientBoostingClassifier(random_state=42)
}
best_auc = -1
best_name = None
best_pipe = None
for model_name, model_obj in models.items():
    with mlflow.start_run(run_name=model_name):
        pipe_bank = Pipeline([('preprocess', preprocess), ('model', model_obj)])
        pipe_bank.fit(X_train, y_train)
        pred_vals = pipe_bank.predict(X_test)
        prob_vals = pipe_bank.predict_proba(X_test)[:, 1]
        metric_vals = {
            'accuracy': accuracy_score(y_test, pred_vals),
            'precision': precision_score(y_test, pred_vals),
            'recall': recall_score(y_test, pred_vals),
            'f1': f1_score(y_test, pred_vals),
            'roc_auc': roc_auc_score(y_test, prob_vals)
        }
        mlflow.log_params({'model_name': model_name, 'test_size': 0.2, 'random_state': 42})
        mlflow.log_metrics(metric_vals)
        mlflow.sklearn.log_model(pipe_bank, artifact_path='model')
        if metric_vals['roc_auc'] > best_auc:
            best_auc = metric_vals['roc_auc']
            best_name = model_name
            best_pipe = pipe_bank
models_dir = Path(__file__).parent.parent / "models"
models_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(best_pipe, models_dir / 'bank_marketing_model.joblib')
with open(models_dir / 'best_model.json', 'w') as file_obj:
    json.dump({'best_model': best_name, 'best_auc': best_auc}, file_obj, indent=2)
print(best_name)
print(best_auc)