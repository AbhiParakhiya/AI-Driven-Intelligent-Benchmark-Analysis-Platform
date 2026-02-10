
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
        """Generates normal transaction metrics."""
        current_time = self.start_time
        # We'll generate one transaction per interval
        n_records = (self.duration_minutes * 60) // self.interval_seconds

        for i in range(n_records):
            record = {
                "transaction_id": f"TXN{10000 + i}",
                "transaction_date": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "amount": round(random.uniform(50, 500), 2),  # Normal amount
                "tx_hour": current_time.hour,
                "tx_day_of_week": current_time.strftime("%A"),
                "account_age_days": random.randint(30, 1000) # established accounts
            }
            self.data.append(record)
            current_time += timedelta(seconds=self.interval_seconds)
        
        return pd.DataFrame(self.data)

    def inject_anomalies(self, df, anomaly_ratio=0.05):
        """Injects suspicious transaction patterns."""
        n_anomalies = int(len(df) * anomaly_ratio)
        indices = np.random.choice(df.index, n_anomalies, replace=False)

        for idx in indices:
            anomaly_type = random.choice(["high_value", "new_account_spike", "late_night_unusual"])
            
            if anomaly_type == "high_value":
                df.at[idx, "amount"] = round(random.uniform(5000, 20000), 2)
            elif anomaly_type == "new_account_spike":
                df.at[idx, "account_age_days"] = random.randint(1, 10)
                df.at[idx, "amount"] = round(random.uniform(1000, 3000), 2)
            elif anomaly_type == "late_night_unusual":
                # We can't easily change the timestamp in a fixed interval easily here, 
                # but we can force the tx_hour feature which the AI uses.
                df.at[idx, "tx_hour"] = random.randint(2, 4)
                df.at[idx, "amount"] = round(random.uniform(200, 1000), 2)

        df["is_anomaly"] = 0
        df.loc[indices, "is_anomaly"] = 1
        return df

if __name__ == "__main__":
    simulator = BenchmarkSimulator(duration_minutes=10)
    df = simulator.generate_base_metrics()
    df = simulator.inject_anomalies(df)
    
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "..", "..", "data", "transaction_data.csv")
    output_path = os.path.normpath(output_path)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} transactions. Saved to {output_path}")
