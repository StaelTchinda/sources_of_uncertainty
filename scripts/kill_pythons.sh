#!/bin/bash

# Get the list of process IDs for Python processes started by your user
pids=$(pgrep -u $USER -f "python")

# Loop through the process IDs and kill each one
for pid in $pids; do
  echo "Killing $pid"
  kill $pid
done
