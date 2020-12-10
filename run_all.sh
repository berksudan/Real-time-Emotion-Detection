#!/bin/bash

CURRENT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"
PYTHON_BIN=$CURRENT_PATH/venv/bin/python3

PYTHON_MAIN_1=$CURRENT_PATH/emotion_detection_evaluator.py
PYTHON_MAIN_2=$CURRENT_PATH/real_time_emotion_detector.py

cd "$CURRENT_PATH" # Change directory to where bash script resides.
"$PYTHON_BIN" "$PYTHON_MAIN_1"
"$PYTHON_BIN" "$PYTHON_MAIN_2"
