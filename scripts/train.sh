#!/usr/bin/env bash

mkdir -p ../logs

nohup python3 -u train.py > ../logs/Model.log 2>&1 &
echo "Training started with PID $!"