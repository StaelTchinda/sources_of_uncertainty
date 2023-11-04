
# Launch tensorboard with the right logdir depending on the parameters data and model
# Usage: ./scripts/start_tensorboard.sh --data [data] --model [model]
# Example: ./scripts/start_tensorboard.sh --data mnist --model cnn
#!/bin/bash

# Set default port
port=6006

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
    echo "Usage: ./scripts/start_tensorboard.sh --data [data] --model [model]"
    echo "Example: ./scripts/start_tensorboard.sh --data mnist --model cnn"
    exit 1
fi

# Get the path of the current file
SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"
# Get the path of the project root
PROJECT_PATH="$(dirname "$SCRIPT_PATH")"
# Set logdir based on data and model
logdir="$PROJECT_PATH/checkpoints/$data/$model"

# Launch tensorboard
session_name="tensorboard_${data}_${model}_${port}"
tmux new -As $session_name -d "echo 'Starting Tensorboard at port $port and directory $logdir under the session $session_name'; tensorboard --logdir='$logdir' --port=$port;read;"
tmux attach -t $session_name