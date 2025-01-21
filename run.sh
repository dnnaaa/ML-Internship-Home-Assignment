#!/bin/sh
export PYTHONPATH=$(pwd)

# Run API
API_PID_FILE=api_pid.txt
if test -f "$API_PID_FILE"; then
    pid=$(cat "$API_PID_FILE")
    echo "Checking API PID: $pid"
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "Killing old API process: $pid"
        kill -9 "$pid"
    fi
    rm "$API_PID_FILE"
fi
echo "RUNNING INFERENCE API"
nohup sh -c 'poetry run python data_ml_assignment/api/main.py' > api.log 2>&1 &
echo $! > "$API_PID_FILE"

# Run Streamlit
STREAMLIT_PID_FILE=streamlit_pid.txt
if test -f "$STREAMLIT_PID_FILE"; then
    pid=$(cat "$STREAMLIT_PID_FILE")
    echo "Checking Streamlit PID: $pid"
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "Killing old Streamlit process: $pid"
        kill -9 "$pid"
    fi
    rm "$STREAMLIT_PID_FILE"
fi
echo "RUNNING STREAMLIT DASHBOARD"
nohup sh -c 'poetry run streamlit run --server.port 8000 dashboard.py' > streamlit.log 2>&1 &
echo $! > "$STREAMLIT_PID_FILE"
