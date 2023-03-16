#!/bin/sh

python train3d.py --exp_id 4 &
PID=$!

sleep 7200 #2h0m
kill $PID
./train.sh
