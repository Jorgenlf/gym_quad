#!/bin/bash

module purge
module load Anaconda3/2020.07
module load intel/2022a

conda create -n quad-env python=3.7.10

