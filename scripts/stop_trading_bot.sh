#!/bin/bash
REPO=/Users/marek/mf-trade-bot
PIDFILE="$REPO/logs/trading_bot.pid"

if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "$(date): stopped trading_bot (pid $PID)" >> "$REPO/logs/cron.log"
    else
        echo "$(date): trading_bot pid $PID not found — already stopped" >> "$REPO/logs/cron.log"
    fi
    rm -f "$PIDFILE"
else
    echo "$(date): no trading_bot pidfile found" >> "$REPO/logs/cron.log"
fi
