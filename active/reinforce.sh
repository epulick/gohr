#!/bin/csh

#-- The directory where this script is
set sc=`dirname $0`
echo $sc
set h=`(cd $sc; cd captive/game; pwd)`
echo $h
source "$h/scripts/set-var-captive.sh"

# Use to run the driver script
python3 experiment_driver.py "captive/game/game-data/rules" "params/test_param_reinforce.yaml"