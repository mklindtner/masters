#!/bin/bash
#
# Title: set_path.sh
# Description: Sets the PYTHONPATH environment variable to the current working directory.
#
# --- IMPORTANT USAGE NOTE ---
# You must 'source' this script for the changes to affect your current terminal session.
# Sourcing executes the script's commands in the current shell environment.
#
# Correct usage:
# source ./set_path.sh
#
# Incorrect usage (will not work as expected):
# ./set_path.sh

# Get the absolute path of the current directory and export it as PYTHONPATH
export PYTHONPATH=$(pwd)

# Print a confirmation message to the user
echo "PYTHONPATH has been successfully set to:"
echo "$PYTHONPATH"