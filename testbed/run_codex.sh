#!/bin/bash

cd "$(dirname "$0")/da_agent/images/codex"
docker build -t codex .

cd "$(dirname "$0")"
python3 run.py --agent codex --suffix qwen --model qwen3.5-397b-a17b