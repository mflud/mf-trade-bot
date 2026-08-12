#!/bin/bash
REPO=/Users/marek/mf-trade-bot
SESSION=focused_monitor

/usr/bin/screen -S "$SESSION" -X quit 2>/dev/null
pkill -f "src/focused_monitor.py" 2>/dev/null
echo "$(date): stopped $SESSION" >> "$REPO/logs/cron.log"
