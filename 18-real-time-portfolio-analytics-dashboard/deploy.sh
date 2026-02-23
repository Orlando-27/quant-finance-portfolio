#!/usr/bin/env bash
# Project 18 — Deploy script | Jose Orlando Bobadilla Fuentes | CQF
set -euo pipefail
echo "Installing dependencies..."
pip install -q -r requirements.txt
echo "Running tests..."
python -m pytest tests/ -v --tb=short 2>&1 | tail -15
echo "Starting Streamlit (port 8501)..."
echo "Cloud Shell: Web Preview -> Port 8501"
MPLBACKEND=Agg streamlit run app.py
