#!/usr/bin/env bash
set -e
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
uvicorn backend.src.main:app --host 0.0.0.0 --port 8000
