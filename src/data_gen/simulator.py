
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

class BenchmarkSimulator:
    def __init__(self, start_time=None, duration_minutes=60, interval_seconds=5):
        self.start_time = start_time if start_time else datetime.now()
        self.duration_minutes = duration_minutes
        self.interval_seconds = interval_seconds
        self.data = []

    def generate_base_metrics(self):
        """Generates normal benchmark metrics."""
        current_time = self.start_time
        end_time = self.start_time + timedelta(minutes=self.duration_minutes)

        while current_time <= end_time:
            record = {
                "timestamp": current_time,
                "cpu_usage": round(random.uniform(20, 60), 2),  # Normal CPU %
                "memory_usage": round(random.uniform(2048, 4096), 2),  # Normal Memory MB
                "disk_io": round(random.uniform(50, 200), 2), # MB/s
                "network_latency": round(random.uniform(10, 50), 2), # ms
                "throughput": round(random.uniform(500, 1500), 2) # req/sec
            }
            self.data.append(record)
            current_time += timedelta(seconds=self.interval_seconds)
        
        return pd.DataFrame(self.data)

    def inject_anomalies(self, df, anomaly_ratio=0.05):
        """Injects performance spikes and drops to simulate anomalies."""
        n_anomalies = int(len(df) * anomaly_ratio)
        indices = np.random.choice(df.index, n_anomalies, replace=False)

        for idx in indices:
            anomaly_type = random.choice(["cpu_spike", "memory_leak", "network_lag", "disk_bottleneck"])
            
            if anomaly_type == "cpu_spike":
                df.at[idx, "cpu_usage"] = round(random.uniform(90, 100), 2)
            elif anomaly_type == "memory_leak":
                df.at[idx, "memory_usage"] = round(random.uniform(7000, 8000), 2)
            elif anomaly_type == "network_lag":
                df.at[idx, "network_latency"] = round(random.uniform(200, 1000), 2)
            elif anomaly_type == "disk_bottleneck":
                df.at[idx, "disk_io"] = round(random.uniform(0.1, 5), 2) # Very slow IO

        df["is_anomaly"] = 0
        df.loc[indices, "is_anomaly"] = 1
        return df

if __name__ == "__main__":
    simulator = BenchmarkSimulator(duration_minutes=10)
    df = simulator.generate_base_metrics()
    df = simulator.inject_anomalies(df)
    
    import os
    # Get directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to reach project root, then into data
    output_path = os.path.join(script_dir, "..", "..", "data", "benchmark_data.csv")
    output_path = os.path.normpath(output_path)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records. Saved to {output_path}")
    print(df.head())
