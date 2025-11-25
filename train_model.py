import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

print("="*60)
print("Training CYBER-DEF25 Malware Detection Model")
print("="*60)

# Generate synthetic data
np.random.seed(42)
n_samples = 10000

data = {
    'packet_size': np.random.randint(50, 1500, n_samples),
    'duration': np.random.uniform(0.1, 300, n_samples),
    'src_bytes': np.random.randint(0, 10000, n_samples),
    'dst_bytes': np.random.randint(0, 10000, n_samples),
    'wrong_fragment': np.random.randint(0, 5, n_samples),
    'urgent': np.random.randint(0, 3, n_samples),
    'hot': np.random.randint(0, 10, n_samples),
    'num_failed_logins': np.random.randint(0, 5, n_samples),
    'num_file_creations': np.random.randint(0, 10, n_samples),
    'num_access_files': np.random.randint(0, 15, n_samples),
    'count': np.random.randint(0, 500, n_samples),
    'srv_count': np.random.randint(0, 500, n_samples),
}

df = pd.DataFrame(data)
df['is_malware'] = ((df['wrong_fragment'] > 2) | (df['num_failed_logins'] > 2) | (df['urgent'] > 1)).astype(int)

print(f"\nDataset created: {len(df)} samples")
print(f"Malware: {df['is_malware'].sum()}")
print(f"Normal: {(df['is_malware'] == 0).sum()}")

X = df.drop('is_malware', axis=1)
y = df['is_malware']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("\nSaving model...")
joblib.dump(model, 'model.pkl')
print("✅ model.pkl created!")

with open('feature_names.txt', 'w') as f:
    f.write(','.join(X.columns.tolist()))
print("✅ feature_names.txt created!")

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
