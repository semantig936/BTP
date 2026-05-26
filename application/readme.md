# Algorithmic RL Trader

## Project Structure

- `src/config`  
  JSON configs: `ticker.json`, `risk_profiles.json`, `sentiment.json`, `run_config.json`
- `src/data`  
  Pipeline outputs: `train.parquet`, `test.parquet`, `prices.parquet`, `returns.parquet`
- `src/data/stat`  
  Data analysis charts and correlation matrix
- `src/model`  
  Trained PPO files and model manifest JSON
- `src/output`  
  Common state graphs and profile-wise outputs (`aggressive`, `moderate`, `conservative`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
