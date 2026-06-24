#!/bin/bash
# Start mes_monitor in a detached screen session.
# Attach with:  screen -r mes_monitor
REPO=/Users/marek/mf-trade-bot
SESSION=mes_monitor
PYTHON=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3

if /usr/bin/screen -list | grep -q "$SESSION"; then
    echo "$(date): $SESSION already running — skipping" >> "$REPO/logs/cron.log"
    exit 0
fi

cd "$REPO"
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
/usr/bin/screen -U -dmS "$SESSION" bash -c "export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES && $PYTHON src/mes_monitor.py 2>> $REPO/logs/mes_monitor.log"
echo "$(date): started $SESSION" >> "$REPO/logs/cron.log"
