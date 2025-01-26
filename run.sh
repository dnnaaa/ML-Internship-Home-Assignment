#!/bin/sh
#Run API
API_PID_FILE=api_pid.txt
if test -f "$API_PID_FILE";
then
    pid=`cat "$API_PID_FILE"`
    echo $pid
    if [ "$pid" != '' ]
    then
        kill -9 $pid
        rm "$API_PID_FILE"
    fi
    echo "RUNNING INFERENCE API"
    nohup sh -c 'poetry run python data_ml_assignment/api/main.py' 2>&1 > api.log &
    echo $! > "$API_PID_FILE"
fi

STREAMLIT_PID_FILE=streamlit_pid.txt
if test -f "$STREAMLIT_PID_FILE";
then
    pid=`cat "$STREAMLIT_PID_FILE"`
    echo $pid
    if [ "$pid" != '' ]
    then
        kill -9 $pid
        rm "$STREAMLIT_PID_FILE"
    fi
    echo "RUNNING STREAMLIT DASHBOARD"
    nohup sh -c 'poetry run streamlit run --server.port 8000 dashboard.py' 2>&1 > streamlit.log &
    echo $! > "$STREAMLIT_PID_FILE"
fi