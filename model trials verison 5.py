import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Load the data
file_path = 'phase_1_2_2023_2024.csv'
habituation_file_path = 'hab_data_sessions.csv'
data = pd.read_csv(file_path, low_memory=False)
habituation_data = pd.read_csv(habituation_file_path)

# Filter out unwanted RatIDs
filtered_data = data[~data['RatID'].isin([9, 16, 19])]

# Define relevant columns
columns_of_interest = [
    'RatID', 'phase', 'session', 'Latency to corr sample', 'Latency to corr match',
    'Time in corr sample', 'Time in inc sample', 'Time in corr match',
    'Time in inc match 1', 'Time in inc match 2', 'False pos inc sample',
    'False pos inc match 1', 'False pos inc match 2'
]
filtered_columns = filtered_data[columns_of_interest]

# Aggregate data at the session level
session_agg = (
    filtered_columns.groupby(['RatID', 'session'], as_index=False)
    .apply(lambda group: pd.Series({
        'total_false_positives_sample': group['False pos inc sample'].sum(),
        'total_false_positives_match': group[['False pos inc match 1', 'False pos inc match 2']].sum(axis=1).sum(),
        'latency_to_corr_sample': group['Latency to corr sample'].mean(),
        'latency_to_corr_match': group['Latency to corr match'].mean(),
        'time_in_correct_sample': group['Time in corr sample'].mean(),
        'time_in_incorrect_target': (
            group['Time in inc sample'] + group['Time in inc match 1'] + group['Time in inc match 2']
        ).mean()
    }))
)

# Aggregate data at the RatID level
rat_level_agg = (
    session_agg.groupby('RatID', as_index=False)
    .mean()
)
rat_level_agg['total_sessions'] = filtered_columns.groupby('RatID')['session'].nunique().values

# Define metrics and calculate 80th percentile thresholds
metrics = [
    'total_false_positives_sample', 'total_false_positives_match',
    'latency_to_corr_sample', 'latency_to_corr_match'
]
thresholds = rat_level_agg[metrics].quantile(0.80)

# Assign binary classes
rat_level_agg['performance_class'] = (rat_level_agg[metrics] > thresholds).any(axis=1).astype(int)

# Aggregate habituation data and merge with RatID data
habituation_agg = habituation_data.groupby('RatID', as_index=False).mean()
merged_data = pd.merge(habituation_agg, rat_level_agg[['RatID', 'performance_class']], on='RatID')
print(habituation_agg)
# Define predictors and target
X = merged_data.drop(columns=['RatID', 'performance_class'])
y = merged_data['performance_class']

# Stratified K-Fold Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# List to store models that meet the performance threshold
ensemble_models = []
performance_threshold = 0.5  # Set a threshold for saving models

# Perform Cross-Validation
for fold, (train_index, test_index) in enumerate(kf.split(X, y), start=1):
    # Split the data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply RFE to select top 4 features
    rfe = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=4)
    rfe.fit(X_train_scaled, y_train)

    # Select features
    X_train_selected = X_train_scaled[:, rfe.support_]
    X_test_selected = X_test_scaled[:, rfe.support_]
    selected_features = X.columns[rfe.support_].tolist()

    # Train the model
    rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")
    rf_model.fit(X_train_selected, y_train)

    # Evaluate the model
    y_pred = rf_model.predict(X_test_selected)
    score = accuracy_score(y_test, y_pred)
    print(f"Fold {fold}: Accuracy = {score:.4f} | Features: {selected_features}")

    # Save the model if it meets the threshold
    if score >= performance_threshold:
        ensemble_models.append({
            "model": rf_model,
            "scaler": scaler,
            "rfe": rfe,
            "feature_names": selected_features,
            "fold": fold,
            "accuracy": score
        })

# Save all models in the ensemble
if ensemble_models:
    joblib.dump(ensemble_models, 'performance_classifier_ensemble.pkl')
    print(f"{len(ensemble_models)} models meeting the threshold saved as 'performance_classifier_ensemble.pkl'.")
else:
    print("No models met the performance threshold.")
from sklearn.metrics import confusion_matrix, classification_report

# Load the saved ensemble models
ensemble_models = joblib.load('performance_classifier_ensemble.pkl')

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Load the saved ensemble models
ensemble_models = joblib.load('performance_classifier_ensemble.pkl')

# Initialize metrics
ensemble_accuracies = []
ensemble_reports = []

# Assuming we want to test on the last fold (as an example)
for model_data in ensemble_models:
    rf_model = model_data['model']
    scaler = model_data['scaler']
    rfe = model_data['rfe']
    selected_features = model_data['feature_names']
    fold = model_data['fold']

    # Select the same test fold used for validation
    _, test_index = list(kf.split(X, y))[fold - 1]
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]

    # Scale and select features
    X_test_scaled = scaler.transform(X_test)
    X_test_selected = X_test_scaled[:, rfe.support_]

    # Predict and evaluate
    y_pred = rf_model.predict(X_test_selected)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Not Proficient', 'Proficient'])
    cm = confusion_matrix(y_test, y_pred)

    # Store results
    ensemble_accuracies.append(accuracy)
    ensemble_reports.append({
        "fold": fold,
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm
    })

    # Print performance for this model
    print(f"Performance for Fold {fold}:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

# Summarize ensemble performance
average_accuracy = np.mean(ensemble_accuracies)
print("\n=== Ensemble Performance Summary ===")
print(f"Average Accuracy Across Models: {average_accuracy:.4f}")

# Aggregate confusion matrices if desired
total_cm = sum([report['confusion_matrix'] for report in ensemble_reports])
print("Aggregate Confusion Matrix:")
print(total_cm)