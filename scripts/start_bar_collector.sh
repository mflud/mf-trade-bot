#!/bin/bash
# Start bar_collector in a detached screen session.
# Attach any time with:  screen -r bar_collector
REPO=/Users/marek/mf-trade-bot
SESSION=bar_collector
PYTHON=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3

if pgrep -f "src/bar_collector.py" > /dev/null 2>&1; then
    echo "$(date): $SESSION already running — skipping" >> "$REPO/logs/cron.log"
    exit 0
fi

cd "$REPO"
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
/usr/bin/screen -U -dmS "$SESSION" "$PYTHON" src/bar_collector.py
echo "$(date): started $SESSION" >> "$REPO/logs/cron.log"
