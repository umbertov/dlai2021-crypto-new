#!/bin/bash

RUN_ID="$1"
shift

## USAGE: ./evaluate_classification.sh WANDB_RUN_ID
PYTHONPATH=. python -- src/evaluation/backtests.py --run-id $RUN_ID $@
