#!/bin/sh

python train3d.py --exp_id 3 &
PID=$!

sleep 3600 #1h0m
kill $PID
./train.sh
