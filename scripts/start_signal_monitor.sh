#!/bin/bash
# Start signal_monitor in a detached screen session.
# Attach any time with:  screen -r signal_monitor
REPO=/Users/marek/mf-trade-bot
SESSION=signal_monitor
PYTHON=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3

if /usr/bin/screen -list | grep -q "$SESSION"; then
    echo "$(date): $SESSION already running — skipping" >> "$REPO/logs/cron.log"
    exit 0
fi

cd "$REPO"
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
/usr/bin/screen -U -dmS "$SESSION" "$PYTHON" src/signal_monitor.py
echo "$(date): started $SESSION" >> "$REPO/logs/cron.log"
