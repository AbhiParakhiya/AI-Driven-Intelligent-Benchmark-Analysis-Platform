
# AI-Driven Intelligent Benchmark Analysis Platform

This platform uses Artificial Intelligence to analyze system benchmark metrics. It automatically detects anomalies, scores system performance, and provides natural language recommendations.

## Features
- **Synthetic Data Generation**: Simulates CPU, Memory, Disk I/O, and Network Latency.
- **Anomaly Detection**: Uses Isolation Forest to identify unusual performance patterns.
- **Performance Scoring**: Calculates a health score (0-100) based on weighted metrics.
- **Recommendations**: actionable advice based on metric analysis.
- **Dashboard**: Interactive Streamlit UI for visualization.
- **API**: FastAPI service for integrating analysis into other tools.

## Installation

1.  Navigate to the project directory:
    ```bash
    cd AI_Benchmark_Platform
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Run the Dashboard (Recommended)
The dashboard allows you to generate data, analyze it, and view results interactively.

```bash
streamlit run src/dashboard/app.py
```

### 2. Run the Benchmark Simulator manually
To generate a new dataset in `data/benchmark_data.csv`:

```bash
python src/data_gen/simulator.py
```

### 3. Run the AI Analysis API
To start the REST API server:

```bash
python src/api/main.py
```
Access the API docs at `http://localhost:8000/docs`.

## Project Structure
- `src/data_gen`: Data simulation logic.
- `src/ai_engine`: Machine Learning models (Isolation Forest).
- `src/api`: FastAPI backend.
- `src/dashboard`: Streamlit frontend.
- `data/`: Stores generated CSVs.
- `models/`: Stores trained ML models (`.pkl`).
