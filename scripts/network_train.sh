#!/bin/sh

# while true; do echo "Do you want to exit? No = 0, Yes = 1"; read input; if [[ $input -eq 1 ]]; then break; fi; done;

wait_for_exit="while true; do echo 'Do you want to exit? No = 0, Yes = 1'; read input; if [[ \$input -eq 1 ]]; then break; fi; done"
# eval "$wait_for_exit";
data_mode='cifar10'
network_mode='vgg11'
session_name="stael_demo_${data_mode}_${network_mode}" 
tmux new -As $task_name -d "source ./venv/bin/activate; python3 -O ./demo/network/train.py --data $data_mode --model $network_mode; $wait_for_exit;"
tmux attach -t $task_name