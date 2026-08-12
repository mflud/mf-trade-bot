#!/bin/bash
REPO=/Users/marek/mf-trade-bot
PIDFILE="$REPO/logs/trading_bot.pid"

if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "$(date): stopped focused_bot (pid $PID)" >> "$REPO/logs/cron.log"
    fi
    rm -f "$PIDFILE"
fi

pkill -f "src/trading_bot.py" 2>/dev/null
echo "$(date): focused_bot stop complete" >> "$REPO/logs/cron.log"
