#!/bin/bash

IFS=',' read -r -a targets <<< "$1"

for BACKEND in "${targets[@]}"; do
  case "$BACKEND" in
    openai)
      python3 run_experiment.py --backend openai --model gpt-3.5-turbo-0125
      python3 run_experiment.py --backend openai --model gpt-4o-2024-08-06
      ;;
    anthropic)
      python3 run_experiment.py --backend anthropic --model claude-3-5-haiku-20241022
      ;;
    deepseek)
      python3 run_experiment.py --backend deepseek --model deepseek-chat
      ;;
    local)
      python3 run_experiment.py --backend local --model meta-llama-3-8b-instruct
      python3 run_experiment.py --backend local --model qwen3-14b
      python3 run_experiment.py --backend local --model qwen2.5-coder-14b
      ;;
  esac
done