#!/bin/sh

# Function to handle PID files and processes
handle_pid_file() {
    local pid_file=$1
    local start_command=$2
    local log_file=$3
    echo "Checking for existing PID in $pid_file"

    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        echo "Found PID: $pid"
        
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "Stopping process $pid"
            kill -9 "$pid"
        else
            echo "No running process found with PID $pid"
        fi
        rm "$pid_file"
    fi

    echo "Starting new process"
    nohup sh -c "$start_command" > "$log_file" 2>&1 &
    echo $! > "$pid_file"
    echo "New process started with PID $(cat $pid_file)"
}

# Run Inference API
handle_pid_file "api_pid.txt" "poetry run python data_ml_assignment/api/main.py" "api.log"

# Run Streamlit Dashboard
handle_pid_file "streamlit_pid.txt" "poetry run streamlit run --server.port 8000 dashboard.py" "streamlit.log"
