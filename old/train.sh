#!/bin/sh

python train3d.py --exp_id 4 &
PID=$!

sleep 7200 #2h0m
kill $PID
./train.sh

#Script that runs the training script for 2 hours, then kills it and restarts it. 
#This is probably to avoid the training script from running for too long and using up all the resources??