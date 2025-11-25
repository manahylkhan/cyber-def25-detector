#!/bin/bash

echo "Setting up CYBER-DEF25 project..."

# Create train_model.py
cat > train_model.py << 'EOF'
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

print("Training malware detection model...")

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

X = df.drop('is_malware', axis=1)
y = df['is_malware']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl')
with open('feature_names.txt', 'w') as f:
    f.write(','.join(X.columns.tolist()))

print("✅ Model saved as model.pkl")
print("✅ Features saved as feature_names.txt")
EOF

# Create generate_test_logs.py
cat > generate_test_logs.py << 'EOF'
import pandas as pd
import numpy as np
import os

os.makedirs('network_logs', exist_ok=True)
np.random.seed(123)

test_data = {
    'packet_size': np.random.randint(50, 1500, 100),
    'duration': np.random.uniform(0.1, 300, 100),
    'src_bytes': np.random.randint(0, 10000, 100),
    'dst_bytes': np.random.randint(0, 10000, 100),
    'wrong_fragment': np.random.randint(0, 5, 100),
    'urgent': np.random.randint(0, 3, 100),
    'hot': np.random.randint(0, 10, 100),
    'num_failed_logins': np.random.randint(0, 5, 100),
    'num_file_creations': np.random.randint(0, 10, 100),
    'num_access_files': np.random.randint(0, 15, 100),
    'count': np.random.randint(0, 500, 100),
    'srv_count': np.random.randint(0, 500, 100),
}

test_df = pd.DataFrame(test_data)
test_df.loc[10:15, 'wrong_fragment'] = 4
test_df.loc[10:15, 'num_failed_logins'] = 4

test_df.to_csv('network_logs/test_logs_001.csv', index=False)
print("✅ Test logs created in network_logs/")
EOF

# Run the scripts
echo "Installing dependencies..."
pip3 install pandas numpy scikit-learn joblib

echo "Training model..."
python3 train_model.py

echo "Generating test data..."
python3 generate_test_logs.py

echo "✅ Setup complete!"
echo "Files created:"
ls -lh model.pkl feature_names.txt network_logs/
