#!/bin/bash
REPO=/Users/marek/mf-trade-bot
SESSION=globex_monitor

/usr/bin/screen -S "$SESSION" -X quit 2>/dev/null
echo "$(date): stopped $SESSION" >> "$REPO/logs/cron.log"
