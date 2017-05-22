#!/bin/bash

cat experiment.log | sed -e $'/ @root\]/!  s,\[[^A-Z]*\, INFO.*,\e[93m&\e[m,' -e $'s,\[[^A-Z]*\, INFO @root].*,\e[36m&\e[m,' -e $'s,\[[^A-Z]*\, WARNING.*,\e[91m&\e[m,' -e $'s,\[[^A-Z]*\, ERROR.*,\e[41m&\e[m,'