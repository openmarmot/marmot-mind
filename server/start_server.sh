#!/bin/bash
set -e

# Marmot Agent Server - start script
# Installs deps in venv and launches the Flask server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🐹 Starting Marmot Agent server setup..."

if [ ! -d "venv" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating Python venv..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip -q
pip install -r code/requirements.txt

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Dependencies installed."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🚀 Launching server..."
cd code
python3 server.py
