import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Generate synthetic network log data for demonstration
# In real scenario, you would load actual network traffic data
def generate_synthetic_data(n_samples=10000):
    """
    Generate synthetic network log features for malware detection
    Features: packet_size, duration, protocol_type, flag_count, error_rate, etc.
    """
    np.random.seed(42)
    
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
    
    # Create labels (0 = Normal, 1 = Malware)
    # Simple rule: if multiple suspicious features are high, label as malware
    df['is_malware'] = (
        (df['wrong_fragment'] > 2) | 
        (df['num_failed_logins'] > 2) |
        (df['urgent'] > 1)
    ).astype(int)
    
    return df

print("="*60)
print("CYBER-DEF25 Malware Detection Model Training")
print("="*60)

# Generate training data
print("\n[1/6] Generating synthetic network log data...")
data = generate_synthetic_data(10000)
print(f"Dataset shape: {data.shape}")
print(f"Malware samples: {data['is_malware'].sum()}")
print(f"Normal samples: {(data['is_malware'] == 0).sum()}")

# Separate features and target
X = data.drop('is_malware', axis=1)
y = data['is_malware']

# Split data
print("\n[2/6] Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Train Random Forest model
print("\n[3/6] Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Model training completed!")

# Evaluate model
print("\n[4/6] Evaluating model performance...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Malware']))

# Feature importance
print("\n[5/6] Feature Importance:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)

# Save the model
print("\n[6/6] Saving model to disk...")
joblib.dump(model, 'model.pkl')
print("Model saved as 'model.pkl'")

# Save feature names for inference
with open('feature_names.txt', 'w') as f:
    f.write(','.join(X.columns.tolist()))
print("Feature names saved as 'feature_names.txt'")

print("\n" + "="*60)
print("Model training completed successfully!")
print("="*60)
