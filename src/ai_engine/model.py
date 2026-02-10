
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
             data_path = os.path.normpath(os.path.join(script_dir, "..", "..", "data", "benchmark_data.csv"))

        if not os.path.exists(data_path):
            print(f"Data file not found at {data_path}")
            return

        df = pd.read_csv(data_path)
        features = ["cpu_usage", "memory_usage", "disk_io", "network_latency", "throughput"]
        
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
        features = ["cpu_usage", "memory_usage", "disk_io", "network_latency", "throughput"]
        
        # Validate that all required features are present
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required benchmark columns: {', '.join(missing_features)}. "
                             "Please ensure your CSV contains these metrics.")
        
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
        """Calculates a 0-100 performance score based on weighted metrics."""
        # Weights (higher usage/latency = lower score)
        # We invert the metrics to get a score where higher is better.
        
        # Max-Min Normalization bounds (approximate from data gen)
        max_cpu = 100
        max_mem = 8192
        max_latency = 1000
        
        # Let's define score as: 100 - (deductions)
        score = 100
        
        try:
            if row.get("cpu_usage", 0) > 80: score -= 20
            if row.get("memory_usage", 0) > 6000: score -= 20
            if row.get("network_latency", 0) > 200: score -= 30
            
            # Throughput bonus
            if row.get("throughput", 0) > 1000: score += 10
        except Exception:
            # Fallback if row is not dict-like or missing keys (though upstream validates)
            pass
            
        return max(0, min(100, score))

    def generate_recommendation(self, row, is_anomaly):
        """Generates natural language recommendations."""
        recommendations = []
        
        if is_anomaly:
            recommendations.append("Anomaly Detected! Immediate investigation required.")
        
        try:
            if row.get("cpu_usage", 0) > 85:
                recommendations.append("High CPU Usage: Consider scaling up CPU resources or optimizing process threads.")
            
            if row.get("memory_usage", 0) > 6000:
                recommendations.append("High Memory Usage: Potential memory leak or insufficient RAM. check allocators.")
                
            if row.get("network_latency", 0) > 100:
                recommendations.append("Network Lag: Check regional endpoints or CDN configuration.")
                
            if row.get("disk_io", 0) < 10:
                recommendations.append("Low Disk I/O: Possible bottleneck in storage subsystem.")
        except Exception:
            pass

        if not recommendations:
            recommendations.append("System performing within normal parameters.")
            
        return " ".join(recommendations)

if __name__ == "__main__":
    ai = BenchmarkAI()
    ai.train_anomaly_detector()
