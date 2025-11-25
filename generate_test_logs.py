import pandas as pd
import numpy as np
import os

# Create network_logs directory
os.makedirs('network_logs', exist_ok=True)

print("Generating test network log files...")

# Generate test log file with some malicious patterns
np.random.seed(123)
n_samples = 100

test_data = {
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

# Create DataFrame
test_df = pd.DataFrame(test_data)

# Inject some obvious malicious patterns
test_df.loc[10:15, 'wrong_fragment'] = 4
test_df.loc[10:15, 'num_failed_logins'] = 4
test_df.loc[30:35, 'urgent'] = 2

# Save test logs
test_df.to_csv('network_logs/test_logs_001.csv', index=False)
print(f"Created: network_logs/test_logs_001.csv ({len(test_df)} records)")

print("Test log generation completed!")
