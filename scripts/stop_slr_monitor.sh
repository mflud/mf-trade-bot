#!/bin/bash
REPO=/Users/marek/mf-trade-bot
SESSION=slr_monitor

/usr/bin/screen -S "$SESSION" -X quit 2>/dev/null
pkill -f "src/slr_monitor.py" 2>/dev/null
echo "$(date): stopped $SESSION" >> "$REPO/logs/cron.log"
