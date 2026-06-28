#!/bin/bash

cd "$(dirname "$0")/da_agent/images/claude-code"
docker build -t claude-code .

cd "$(dirname "$0")"
python3 run.py --agent claude-code --suffix qwen --model qwen3.5-397b-a17b