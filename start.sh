#!/bin/bash
set -e

# Marmot — chat room and minds in one process.
#   ./start.sh
#   ./start.sh --port 5000
#   ./start.sh --create alice --llm-url http://127.0.0.1:8000/v1 --llm-model my-model --start-loop

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🐹 Starting Marmot..."

if [ ! -d "venv" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating Python venv..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip -q
pip install -r code/requirements.txt -q

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Dependencies installed."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🚀 Launching..."
cd code
exec python3 app.py "$@"
