#!/bin/sh
# Parse command line arguments
while [[ $# -gt 0 ]]
do
key="$1"
device=0
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
    --device)
    device="$2"
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
    echo "Usage: ./scripts/prune_laplace.sh --data [data] --model [model]"
    echo "Example: ./scripts/prune_laplace.sh --data mnist --model lenet5"
    exit 1
fi

# Get the path of the current file
SCRIPT_PATH="$( cd "$(dirname "$0")" ; cd ../.. ; pwd -P )"
# Get the path of the project root
PROJECT_PATH="$(dirname "$SCRIPT_PATH")"

# Define output verbose file based on model, dataset and timestamp
output_file="$PROJECT_PATH/verbose/${data}/${model}/bayesian/prune/run_$(date +%Y%m%d_%H%M%S).txt"

# Create output directory if it does not exist
mkdir -p "$(dirname "$output_file")"

# Launch training
echo "Pruning model $model on dataset $data. Output file: $output_file"
session_name="laplace_train_${data}_${model}"
tmux new -As $session_name -d "$PROJECT_PATH/venv/bin/python -O $PROJECT_PATH/demo/bayesian/prune.py --data $data --model $model --device $device > >(tee -a $output_file) 2> >(tee -a $output_file >&2);"
