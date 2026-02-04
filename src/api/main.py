
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

# Add src to python path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_engine.model import BenchmarkAI

app = FastAPI(title="AI Benchmark Analyzer", description="API for detecting anomalies and scoring system performance.")

# Initialize AI Engine
ai_engine = BenchmarkAI()

class BenchmarkMetric(BaseModel):
    timestamp: str
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_latency: float
    throughput: float

class AnalysisResult(BaseModel):
    is_anomaly: bool
    performance_score: float
    recommendation: str

@app.get("/")
def home():
    return {"message": "AI Benchmark Analysis API is running."}

@app.post("/analyze", response_model=List[AnalysisResult])
def analyze_benchmark(metrics: List[BenchmarkMetric]):
    try:
        # Convert Pydantic models to dicts
        data = [m.dict() for m in metrics]
        results = ai_engine.detect_anomalies(data)
        
        if results is None:
            raise HTTPException(status_code=500, detail="Model not initialized. Please train the model first.")
            
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
