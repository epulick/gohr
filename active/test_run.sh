#!/bin/csh

#-- The directory where this script is
set sc=`dirname $0`
echo $sc
set h=`(cd $sc; cd captive/game; pwd)`
echo $h
source "$h/scripts/set-var-captive.sh"

# Use to test the engine class
#python3 rule_game_engine.py "captive/game/game-data/rules"

# Use to test the environment class
#python3 rule_game_env.py "captive/game/game-data/rules"

# Use to test the featurization class
#python3 featurization.py "captive/game/game-data/rules"

# Use to test the driver script (note the shift of the args to a parameter file)
python3 driver.py "captive/game/game-data/rules" "params/test_param.yaml"

# Use to run the driver script
#python3 experiment_driver.py "captive/game/game-data/rules" "params/test_param.yaml"

# Use for hyperparameter tuning
#python3 experiment_driver.py "captive/game/game-data/rules" "params/test_param.yaml" "params/hyperparameter/test_hyperparameter.yaml"