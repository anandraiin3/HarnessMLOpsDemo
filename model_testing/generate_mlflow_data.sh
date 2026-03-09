#!/bin/zsh

# Command to be executed
COMMAND="jupyter nbconvert --to notebook --execute credit_card_approval.ipynb"

# Number of times to execute the command
COUNT=20

# Time to wait between executions (in seconds)
WAIT_TIME=30

# Loop to execute the command
for ((i = 1; i <= COUNT; i++)); do
    echo "Execution $i: Running command..."
    eval $COMMAND
    
    # If it's not the last execution, wait for the specified time
    if ((i < COUNT)); then
        echo "Waiting for $WAIT_TIME seconds before next execution..."
        sleep $WAIT_TIME
    fi
done

echo "All executions completed."
