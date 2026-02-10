
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

class BenchmarkAI:
    def __init__(self):
        self.anomaly_model = IsolationForest(contamination=0.05, random_state=42)
        self.scaler = StandardScaler()
        # Get directory of this script to anchor paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.normpath(os.path.join(script_dir, "..", "..", "models", "anomaly_model.pkl"))
        self.scaler_path = os.path.normpath(os.path.join(script_dir, "..", "..", "models", "scaler.pkl"))

    def train_anomaly_detector(self, data_path=None):
        """Trains Isolation Forest on benchmark data."""
        if data_path is None:
             script_dir = os.path.dirname(os.path.abspath(__file__))
             data_path = os.path.normpath(os.path.join(script_dir, "..", "..", "data", "transaction_data.csv"))

        if not os.path.exists(data_path):
            print(f"Data file not found at {data_path}")
            return

        df = pd.read_csv(data_path)
        # Numerical features for transaction anomaly detection
        features = ["amount", "tx_hour", "account_age_days"]
        
        # Fit scaler
        X = self.scaler.fit_transform(df[features])
        
        # Train model
        self.anomaly_model.fit(X)
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save artifacts
        joblib.dump(self.anomaly_model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print("Anomaly model and scaler trained and saved.")

    def detect_anomalies(self, metrics):
        """
        Predicts if a new data point is an anomaly.
        metrics: dict or list of dicts with keys [cpu_usage, memory_usage, disk_io, network_latency, throughput]
        """
        if not os.path.exists(self.model_path):
            print("Model not found. Please train first.")
            return None

        model = joblib.load(self.model_path)
        scaler = joblib.load(self.scaler_path)

        df = pd.DataFrame(metrics)
        # Transaction-specific features
        features = ["amount", "tx_hour", "account_age_days"]
        
        # Validate that all required features are present
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required transaction columns: {', '.join(missing_features)}. "
                             "Please ensure your CSV contains these fields.")
        
        X = scaler.transform(df[features])
        predictions = model.predict(X) # -1 for anomaly, 1 for normal
        
        results = []
        for i, pred in enumerate(predictions):
            is_anomaly = True if pred == -1 else False
            score = self.calculate_performance_score(df.iloc[i])
            recommendation = self.generate_recommendation(df.iloc[i], is_anomaly)
            
            results.append({
                "is_anomaly": is_anomaly,
                "performance_score": score,
                "recommendation": recommendation
            })
            
        return results

    def calculate_performance_score(self, row):
        """Calculates a 0-100 risk/health score based on transaction metrics."""
        # For transactions, "health" might mean "normalcy" or "low risk"
        score = 100
        
        try:
            # Simple heuristics for risky transactions
            amount = row.get("amount", 0)
            account_age = row.get("account_age_days", 365)
            tx_hour = row.get("tx_hour", 12)
            
            # Deduction for very large amounts
            if amount > 100000: score -= 30
            elif amount > 50000: score -= 15
            
            # Deduction for very new accounts making large transactions
            if account_age < 30 and amount > 1000: score -= 40
            
            # Deduction for odd-hour transactions (e.g., 2 AM - 5 AM)
            if tx_hour >= 2 and tx_hour <= 5: score -= 20
            
        except Exception:
            pass
            
        return max(0, min(100, score))

    def generate_recommendation(self, row, is_anomaly):
        """Generates natural language insights for transactions."""
        recommendations = []
        
        if is_anomaly:
            recommendations.append("🚩 High-Risk Pattern: This transaction deviates significantly from typical behavior.")
        
        try:
            amount = row.get("amount", 0)
            account_age = row.get("account_age_days", 365)
            tx_hour = row.get("tx_hour", 12)
            
            if amount > 100000:
                recommendations.append("Extreme Value: Unusually large transaction amount.")
            
            if account_age < 30:
                recommendations.append("New Account: Transaction originated from an account less than 30 days old.")
                
            if tx_hour >= 2 and tx_hour <= 5:
                recommendations.append("Off-Peak Timing: Transaction occurred during unusual late-night hours.")
                
        except Exception:
            pass

        if not recommendations:
            recommendations.append("Transaction appears consistent with normal usage patterns.")
            
        return " ".join(recommendations)

if __name__ == "__main__":
    ai = BenchmarkAI()
    ai.train_anomaly_detector()
