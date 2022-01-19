#!/bin/bash

bash ./prepare.sh xp_$1
python3 ../src/prepareVectors.py ../experiments/xp_$1/frameid "google/electra-base-discriminator"

