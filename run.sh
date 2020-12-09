#!/bin/bash

CURRENT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"
PYTHON_BIN=$CURRENT_PATH/venv/bin/python3
PYTHON_MAIN=$CURRENT_PATH/main.py

cd "$CURRENT_PATH" # Change directory to where bash script resides. 
"$PYTHON_BIN" "$PYTHON_MAIN"
