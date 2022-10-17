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
python3 rule_game_env.py "captive/game/game-data/rules"
