#!/bin/bash
set -e

# Marmot Mind - independent AI chat participant
# Usage:
#   ./start_mind.sh
#   ./start_mind.sh --create alice
#   ./start_mind.sh --resume alice --start-loop
#   ./start_mind.sh --resume alice --chat-server http://127.0.0.1:5000 --llm-url http://10.12.0.50:8000/v1 --llm-model my-model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🧠 Starting Marmot Mind..."

if [ ! -d "venv" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating Python venv..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip -q
pip install -r code/requirements.txt

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Dependencies installed."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🚀 Launching mind..."
cd code
python3 mind.py "$@"
