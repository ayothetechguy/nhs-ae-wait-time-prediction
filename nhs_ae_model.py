"""
NHS A&E WAIT TIME PREDICTION - COMPLETE MACHINE LEARNING MODEL
================================================================
Author: Ayoolumi Melehon
Project: Predictive ML system for Emergency Department wait times
Tech Stack: Python, scikit-learn, pandas, matplotlib, seaborn
================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

print("="*80)
print("🏥 NHS A&E WAIT TIME PREDICTION - ML MODEL TRAINING")
print("="*80)
print(f"Project: Machine Learning for Emergency Department Management")
print(f"Author: Ayoolumi Melehon")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================================================
# PHASE 1: GENERATE SYNTHETIC PATIENT DATA
# ============================================================================

print("\n📊 PHASE 1: Generating Synthetic Patient Data...")
print("-"*80)

np.random.seed(42)
n_samples = 5000

print(f"Creating {n_samples:,} synthetic patient records...")

# Generate realistic A&E patient data
data = {
    'patient_id': [f'AE{i:05d}' for i in range(1, n_samples + 1)],
    'arrival_time': pd.date_range('2024-01-01', periods=n_samples, freq='17min'),
}

df = pd.DataFrame(data)

# Extract temporal features
df['hour'] = df['arrival_time'].dt.hour
df['day_of_week'] = df['arrival_time'].dt.dayofweek
df['month'] = df['arrival_time'].dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if (10 <= x <= 13) or (18 <= x <= 21) else 0)
df['is_night'] = df['hour'].apply(lambda x: 1 if 0 <= x <= 6 else 0)

# Patient demographics
df['age'] = np.random.choice(
    [np.random.randint(18, 35), np.random.randint(35, 50),
     np.random.randint(50, 65), np.random.randint(65, 90)],
    size=n_samples,
    p=[0.25, 0.30, 0.25, 0.20]
)

# Triage category (1=Immediate, 5=Non-urgent)
df['triage_category'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.15, 0.35, 0.30, 0.15])

# Arrival mode
arrival_modes = ['Walk-in', 'Ambulance', 'GP Referral']
df['arrival_mode'] = np.random.choice(arrival_modes, n_samples, p=[0.60, 0.30, 0.10])

# Chief complaint
complaints = ['Chest Pain', 'Breathing Difficulty', 'Abdominal Pain', 
              'Injury/Trauma', 'Fever', 'Mental Health', 'Other']
df['chief_complaint'] = np.random.choice(complaints, n_samples, 
                                         p=[0.15, 0.12, 0.18, 0.20, 0.10, 0.08, 0.17])

# Department conditions
df['department_occupancy'] = np.clip(
    60 + 20 * df['is_peak_hour'] + 10 * df['is_weekend'] + np.random.normal(0, 8, n_samples),
    30, 98
).round(0)

df['staff_available'] = np.clip(
    20 - (df['department_occupancy'] / 10) + np.random.randint(-3, 4, n_samples),
    8, 30
).astype(int)

# Previous visits
df['previous_visits_30d'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.70, 0.20, 0.07, 0.03])

# Generate wait times
base_wait = 45

wait_time = (
    base_wait +
    (df['triage_category'] * 25) +
    (df['department_occupancy'] * 1.5) +
    (-df['staff_available'] * 2) +
    (df['is_peak_hour'] * 30) +
    (df['is_weekend'] * 20) +
    (df['is_night'] * -15) +
    np.where(df['arrival_mode'] == 'Ambulance', -30, 0) +
    np.where(df['arrival_mode'] == 'Walk-in', 15, 0) +
    (df['age'] / 2) +
    np.random.normal(0, 20, n_samples)
)

df['wait_time_minutes'] = np.clip(wait_time, 5, 480).round(0)

# Encode categorical variables
le_mode = LabelEncoder()
le_complaint = LabelEncoder()
df['arrival_mode_encoded'] = le_mode.fit_transform(df['arrival_mode'])
df['complaint_encoded'] = le_complaint.fit_transform(df['chief_complaint'])

print(f"✅ Generated {len(df):,} patient records")
print(f"📅 Date range: {df['arrival_time'].min()} to {df['arrival_time'].max()}")
print(f"⏱️  Wait time range: {df['wait_time_minutes'].min():.0f} - {df['wait_time_minutes'].max():.0f} minutes")

# ============================================================================
# PHASE 2: EXPLORATORY DATA ANALYSIS & VISUALIZATIONS
# ============================================================================

print("\n📈 PHASE 2: Creating Visualizations...")
print("-"*80)

os.makedirs('output', exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Visualization 1: Wait Time Distribution
print("Creating visualization 1/6: Wait Time Distribution...")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df['wait_time_minutes'], bins=50, color='#3b82f6', edgecolor='black', alpha=0.7)
plt.xlabel('Wait Time (minutes)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of A&E Wait Times', fontsize=14, fontweight='bold')
plt.axvline(df['wait_time_minutes'].mean(), color='red', linestyle='--', 
            label=f'Mean: {df["wait_time_minutes"].mean():.1f} min')
plt.legend()

plt.subplot(1, 2, 2)
df.boxplot(column='wait_time_minutes', by='triage_category', ax=plt.gca())
plt.xlabel('Triage Category', fontsize=12)
plt.ylabel('Wait Time (minutes)', fontsize=12)
plt.title('Wait Time by Triage Category', fontsize=14, fontweight='bold')
plt.suptitle('')
plt.tight_layout()
plt.savefig('output/01_wait_time_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved: output/01_wait_time_distribution.png")
plt.close()

# Visualization 2: Wait Time by Hour
print("Creating visualization 2/6: Wait Time by Hour...")
plt.figure(figsize=(12, 5))
hourly_stats = df.groupby('hour')['wait_time_minutes'].agg(['mean', 'median', 'std'])
plt.plot(hourly_stats.index, hourly_stats['mean'], marker='o', linewidth=2, 
         label='Mean Wait Time', color='#3b82f6')
plt.fill_between(hourly_stats.index, 
                 hourly_stats['mean'] - hourly_stats['std'],
                 hourly_stats['mean'] + hourly_stats['std'],
                 alpha=0.2, color='#3b82f6')
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Wait Time (minutes)', fontsize=12)
plt.title('Average Wait Time by Hour of Day', fontsize=14, fontweight='bold')
plt.xticks(range(0, 24))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/02_wait_time_by_hour.png', dpi=300, bbox_inches='tight')
print("✅ Saved: output/02_wait_time_by_hour.png")
plt.close()

# Visualization 3: Occupancy Impact
print("Creating visualization 3/6: Occupancy Impact...")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(df['department_occupancy'], df['wait_time_minutes'], 
           alpha=0.3, c=df['triage_category'], cmap='viridis')
plt.xlabel('Department Occupancy (%)', fontsize=12)
plt.ylabel('Wait Time (minutes)', fontsize=12)
plt.title('Wait Time vs Department Occupancy', fontsize=14, fontweight='bold')
plt.colorbar(label='Triage Category')

plt.subplot(1, 2, 2)
plt.scatter(df['staff_available'], df['wait_time_minutes'], 
           alpha=0.3, color='#10b981')
plt.xlabel('Staff Available', fontsize=12)
plt.ylabel('Wait Time (minutes)', fontsize=12)
plt.title('Wait Time vs Staff Availability', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/03_occupancy_impact.png', dpi=300, bbox_inches='tight')
print("✅ Saved: output/03_occupancy_impact.png")
plt.close()

# Visualization 4: Arrival Mode Analysis
print("Creating visualization 4/6: Arrival Mode Analysis...")
plt.figure(figsize=(12, 5))
arrival_stats = df.groupby('arrival_mode')['wait_time_minutes'].mean().sort_values()
plt.subplot(1, 2, 1)
arrival_stats.plot(kind='barh', color=['#3b82f6', '#10b981', '#f59e0b'])
plt.xlabel('Average Wait Time (minutes)', fontsize=12)
plt.ylabel('Arrival Mode', fontsize=12)
plt.title('Average Wait Time by Arrival Mode', fontsize=14, fontweight='bold')

plt.subplot(1, 2, 2)
df['arrival_mode'].value_counts().plot(kind='pie', autopct='%1.1f%%', 
                                        colors=['#3b82f6', '#10b981', '#f59e0b'])
plt.ylabel('')
plt.title('Distribution of Arrival Modes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/04_arrival_mode_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved: output/04_arrival_mode_analysis.png")
plt.close()

# ============================================================================
# PHASE 3: PREPARE DATA FOR MACHINE LEARNING
# ============================================================================

print("\n🤖 PHASE 3: Preparing Data for Machine Learning...")
print("-"*80)

feature_columns = [
    'hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour', 'is_night',
    'triage_category', 'age', 'arrival_mode_encoded', 'complaint_encoded',
    'department_occupancy', 'staff_available', 'previous_visits_30d'
]

X = df[feature_columns]
y = df['wait_time_minutes']

print(f"Features: {len(feature_columns)}")
print(f"Samples: {len(X):,}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Training set: {len(X_train):,} samples")
print(f"✅ Test set: {len(X_test):,} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# PHASE 4: TRAIN MACHINE LEARNING MODELS
# ============================================================================

print("\n🎯 PHASE 4: Training Machine Learning Models...")
print("-"*80)

# Random Forest
print("Training Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print(f"✅ Random Forest Results:")
print(f"   MAE: {rf_mae:.2f} minutes")
print(f"   RMSE: {rf_rmse:.2f} minutes")
print(f"   R² Score: {rf_r2:.4f}")

# Gradient Boosting
print("\nTraining Gradient Boosting Regressor...")
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=7, random_state=42)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)

gb_mae = mean_absolute_error(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_r2 = r2_score(y_test, gb_pred)

print(f"✅ Gradient Boosting Results:")
print(f"   MAE: {gb_mae:.2f} minutes")
print(f"   RMSE: {gb_rmse:.2f} minutes")
print(f"   R² Score: {gb_r2:.4f}")

# Select best model
if gb_mae < rf_mae:
    best_model = gb_model
    best_name = "Gradient Boosting"
    best_pred = gb_pred
    best_metrics = {'MAE': gb_mae, 'RMSE': gb_rmse, 'R2': gb_r2}
else:
    best_model = rf_model
    best_name = "Random Forest"
    best_pred = rf_pred
    best_metrics = {'MAE': rf_mae, 'RMSE': rf_rmse, 'R2': rf_r2}

print(f"\n🏆 Best Model: {best_name}")

# ============================================================================
# PHASE 5: MODEL EVALUATION & VISUALIZATIONS
# ============================================================================

print("\n📊 PHASE 5: Evaluating Model Performance...")
print("-"*80)

# Feature Importance
print("Creating visualization 5/6: Feature Importance...")
importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': importances
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'], 
         color='#3b82f6')
plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title(f'Feature Importance - {best_name}', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('output/05_feature_importance.png', dpi=300, bbox_inches='tight')
print("✅ Saved: output/05_feature_importance.png")
plt.close()

# Actual vs Predicted
print("Creating visualization 6/6: Actual vs Predicted...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, best_pred, alpha=0.3, color='#3b82f6')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Wait Time (minutes)', fontsize=12)
plt.ylabel('Predicted Wait Time (minutes)', fontsize=12)
plt.title('Actual vs Predicted Wait Times', fontsize=14, fontweight='bold')
plt.legend()

plt.subplot(1, 2, 2)
residuals = y_test - best_pred
plt.hist(residuals, bins=50, color='#10b981', edgecolor='black', alpha=0.7)
plt.xlabel('Prediction Error (minutes)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
plt.axvline(0, color='red', linestyle='--', linewidth=2)
plt.tight_layout()
plt.savefig('output/06_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("✅ Saved: output/06_actual_vs_predicted.png")
plt.close()

# ============================================================================
# PHASE 6: SAVE MODEL & COMPONENTS
# ============================================================================

print("\n💾 PHASE 6: Saving Model and Components...")
print("-"*80)

joblib.dump(best_model, 'output/nhs_ae_model.pkl')
print("✅ Saved: output/nhs_ae_model.pkl")

joblib.dump(scaler, 'output/scaler.pkl')
print("✅ Saved: output/scaler.pkl")

joblib.dump(le_mode, 'output/arrival_mode_encoder.pkl')
joblib.dump(le_complaint, 'output/complaint_encoder.pkl')
print("✅ Saved: output/arrival_mode_encoder.pkl")
print("✅ Saved: output/complaint_encoder.pkl")

with open('output/feature_columns.txt', 'w') as f:
    f.write('\n'.join(feature_columns))
print("✅ Saved: output/feature_columns.txt")

# ============================================================================
# PHASE 7: SAMPLE PREDICTIONS
# ============================================================================

print("\n🔮 PHASE 7: Sample Predictions...")
print("-"*80)

sample_patients = [
    {
        'scenario': 'Peak Hour - High Priority',
        'hour': 19, 'day_of_week': 2, 'month': 10, 'is_weekend': 0,
        'is_peak_hour': 1, 'is_night': 0, 'triage_category': 2,
        'age': 72, 'arrival_mode': 'Ambulance', 'complaint': 'Chest Pain',
        'department_occupancy': 85, 'staff_available': 12, 'previous_visits_30d': 0
    },
    {
        'scenario': 'Night Shift - Low Priority',
        'hour': 3, 'day_of_week': 4, 'month': 10, 'is_weekend': 0,
        'is_peak_hour': 0, 'is_night': 1, 'triage_category': 5,
        'age': 28, 'arrival_mode': 'Walk-in', 'complaint': 'Other',
        'department_occupancy': 45, 'staff_available': 18, 'previous_visits_30d': 1
    },
    {
        'scenario': 'Weekend - Moderate Priority',
        'hour': 14, 'day_of_week': 6, 'month': 10, 'is_weekend': 1,
        'is_peak_hour': 0, 'is_night': 0, 'triage_category': 3,
        'age': 45, 'arrival_mode': 'GP Referral', 'complaint': 'Abdominal Pain',
        'department_occupancy': 75, 'staff_available': 15, 'previous_visits_30d': 0
    }
]

for i, patient in enumerate(sample_patients, 1):
    arrival_encoded = le_mode.transform([patient['arrival_mode']])[0]
    complaint_encoded = le_complaint.transform([patient['complaint']])[0]
    
    features = [[
        patient['hour'], patient['day_of_week'], patient['month'],
        patient['is_weekend'], patient['is_peak_hour'], patient['is_night'],
        patient['triage_category'], patient['age'], arrival_encoded, complaint_encoded,
        patient['department_occupancy'], patient['staff_available'],
        patient['previous_visits_30d']
    ]]
    
    features_scaled = scaler.transform(features)
    prediction = best_model.predict(features_scaled)[0]
    
    print(f"\n📋 Example {i}: {patient['scenario']}")
    print(f"   Patient: {patient['age']}yo, {patient['arrival_mode']}, {patient['complaint']}")
    print(f"   Triage: Category {patient['triage_category']}")
    print(f"   Department: {patient['department_occupancy']}% occupancy, {patient['staff_available']} staff")
    print(f"   ⏱️  Predicted Wait Time: {prediction:.0f} minutes ({prediction/60:.1f} hours)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ PROJECT COMPLETE!")
print("="*80)
print(f"\n📊 Dataset: {n_samples:,} synthetic patient records")
print(f"🤖 Best Model: {best_name}")
print(f"📈 Performance:")
print(f"   • Mean Absolute Error: {best_metrics['MAE']:.2f} minutes")
print(f"   • Root Mean Squared Error: {best_metrics['RMSE']:.2f} minutes")
print(f"   • R² Score: {best_metrics['R2']:.4f}")
print(f"\n💾 Saved Files:")
print(f"   • 6 visualization images in output/")
print(f"   • Trained model (nhs_ae_model.pkl)")
print(f"   • Feature scaler and encoders")
print(f"\n🎯 Next Steps:")
print(f"   1. Review visualizations in the 'output' folder")
print(f"   2. Test model with custom predictions")
print(f"   3. Build Streamlit dashboard")
print(f"   4. Add to portfolio website")
print("\n" + "="*80)
print("🎉 Ready for Portfolio Showcase!")
print("="*80)