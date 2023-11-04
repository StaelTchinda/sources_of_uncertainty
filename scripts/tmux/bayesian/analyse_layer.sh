#!/bin/sh
# Parse command line arguments
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --data)
    data="$2"
    shift
    shift
    ;;
    --model)
    model="$2"
    shift
    shift
    ;;
    --port)
    port="$2"
    shift
    shift
    ;;
    *)
    echo "Unknown option: $1"
    exit 1
    ;;
esac
done

# Check if data and model are set
if [ -z "$data" ] || [ -z "$model" ]
then
    echo "Usage: ./scripts/tmux/bayesian/analyse_layer.sh --data [data] --model [model]"
    echo "Example: ./scripts/tmux/bayesian/analyse_layer.sh --data mnist --model lenet5"
    exit 1
fi

# Get the path of the current file
SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"
# Get the path of the project root
PROJECT_PATH="$(dirname "$(dirname "$(dirname "$SCRIPT_PATH")")")"

wait_for_exit="while true; do echo 'Do you want to exit? No = 0, Yes = 1'; read input; if [[ \$input -eq 1 ]]; then break; fi; done"

# Launch training
echo "Analysing layers of network $model on dataset $data"
session_name="bayesian_analyse_layer_${data}_${model}"
tmux new -As $session_name -d "$PROJECT_PATH/venv/bin/python -O $PROJECT_PATH/demo/bayesian/analyse_layer.py --data $data --model $model $@; $wait_for_exit;"
tmux attach -t $session_name