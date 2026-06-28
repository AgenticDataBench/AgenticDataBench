#!/bin/bash

cd "$(dirname "$0")/da_agent/images/smolagents"
docker build -t smolagents .

cd "$(dirname "$0")"
python3 run.py --agent smolagents --suffix qwen --model qwen3.5-397b-a17b --port 30008 --max_steps 40