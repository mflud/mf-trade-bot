#!/bin/bash
# Start slr_monitor in a detached screen session.
# Attach any time with:  screen -r slr_monitor
REPO=/Users/marek/mf-trade-bot
SESSION=slr_monitor
PYTHON=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3

if /usr/bin/screen -list | grep -q "$SESSION"; then
    echo "$(date): $SESSION already running — skipping" >> "$REPO/logs/cron.log"
    exit 0
fi

cd "$REPO"
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
/usr/bin/screen -U -dmS "$SESSION" bash -c "export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES && $PYTHON src/slr_monitor.py 2>> $REPO/logs/slr_monitor.log"
echo "$(date): started $SESSION" >> "$REPO/logs/cron.log"
