
import pandas as pd
import numpy as np
import sys
import os

# Mock the AI Engine for a moment, or use the real one
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ai_engine.model import BenchmarkAI

def test_reproduction():
    ai = BenchmarkAI()
    
    # This is what the user uploaded (from the screenshot)
    user_data = [
        {"transaction_id": "TXN100039", "customer_id": "CUST1039", "amount": 3366, "transaction_type": "debit"},
        {"transaction_id": "TXN100024", "customer_id": "CUST1007", "amount": 59687, "transaction_type": "withdrawal"}
    ]
    df = pd.DataFrame(user_data)
    metrics = df.to_dict(orient="records")
    
    print("Testing with user-like data...")
    try:
        results = ai.detect_anomalies(metrics)
        print("Results generated (Success path - unexpected)")
    except ValueError as ve:
        print(f"Caught expected ValueError: {ve}")
    except Exception as e:
        print(f"Caught unexpected Exception ({type(e).__name__}): {e}")

if __name__ == "__main__":
    # Ensure models exist (might need training or stubbing)
    # For this test, we just want to see if it catches the exception
    test_reproduction()
