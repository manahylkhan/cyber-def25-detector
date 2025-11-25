import pandas as pd
import joblib
import os
from datetime import datetime
import glob

print("="*60)
print("CYBER-DEF25 Malware Detection - Inference")
print("="*60)

# Load the trained model
print("\n[1/4] Loading trained model...")
try:
    model = joblib.load('model.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Load feature names
with open('feature_names.txt', 'r') as f:
    feature_names = f.read().strip().split(',')
print(f"Expected features: {feature_names}")

# Read log files from /input/logs
print("\n[2/4] Reading network log files...")
input_dir = '/input/logs'
output_dir = '/output'

# Find all CSV files in input directory
log_files = glob.glob(os.path.join(input_dir, '*.csv'))

if not log_files:
    print("Warning: No log files found in /input/logs!")
    print("Creating sample output file...")
    # Create empty alerts file
    alerts_df = pd.DataFrame(columns=['timestamp', 'log_file', 'threat_detected', 'confidence'])
    alerts_df.to_csv(os.path.join(output_dir, 'alerts.csv'), index=False)
    exit(0)

print(f"Found {len(log_files)} log file(s)")

# Process each log file
all_alerts = []

print("\n[3/4] Analyzing logs for threats...")
for log_file in log_files:
    try:
        print(f"\nProcessing: {os.path.basename(log_file)}")
        
        # Read log file
        logs_df = pd.read_csv(log_file)
        print(f"  - Loaded {len(logs_df)} records")
        
        # Ensure all required features are present
        missing_features = set(feature_names) - set(logs_df.columns)
        if missing_features:
            print(f"  - Warning: Missing features: {missing_features}")
            # Add missing features with default values
            for feature in missing_features:
                logs_df[feature] = 0
        
        # Select only the features used in training
        X = logs_df[feature_names]
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Create alerts for detected threats
        threat_indices = predictions == 1
        num_threats = threat_indices.sum()
        
        print(f"  - Threats detected: {num_threats}/{len(logs_df)}")
        
        if num_threats > 0:
            for idx in logs_df[threat_indices].index:
                alert = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'log_file': os.path.basename(log_file),
                    'record_id': idx,
                    'threat_detected': 'MALWARE',
                    'confidence': f"{probabilities[idx][1]:.4f}",
                    'packet_size': logs_df.loc[idx, 'packet_size'],
                    'duration': logs_df.loc[idx, 'duration'],
                }
                all_alerts.append(alert)
    
    except Exception as e:
        print(f"  - Error processing {log_file}: {e}")
        continue

# Save results
print("\n[4/4] Saving detection results...")
if all_alerts:
    alerts_df = pd.DataFrame(all_alerts)
    output_file = os.path.join(output_dir, 'alerts.csv')
    alerts_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    print(f"Total alerts generated: {len(all_alerts)}")
    print("\nSample alerts:")
    print(alerts_df.head())
else:
    print("No threats detected in any log files.")
    # Still create an empty alerts file
    alerts_df = pd.DataFrame(columns=['timestamp', 'log_file', 'threat_detected', 'confidence'])
    alerts_df.to_csv(os.path.join(output_dir, 'alerts.csv'), index=False)

print("\n" + "="*60)
print("Inference completed successfully!")
print("="*60)
