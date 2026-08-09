#!/bin/bash
# Start ml_monitor in a detached screen session.
REPO=/Users/marek/mf-trade-bot
PYTHON=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3

if /usr/bin/screen -list | grep -q "ml_monitor"; then
    echo "ml_monitor already running — attach with: screen -r ml_monitor"
    exit 0
fi

cd "$REPO"
/usr/bin/screen -dmS ml_monitor "$PYTHON" src/ml_monitor.py
echo "ml_monitor started. Attach with: screen -r ml_monitor"
