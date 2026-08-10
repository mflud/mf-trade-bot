#!/bin/bash
REPO=/Users/marek/mf-trade-bot

for SYM in MES MNQ; do
    PIDFILE="$REPO/logs/ml_trading_bot_${SYM}.pid"
    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            echo "$(date): stopped ml_trading_bot $SYM (pid $PID)" >> "$REPO/logs/cron.log"
            echo "  Stopped ml_trading_bot $SYM (pid $PID)"
        else
            echo "$(date): ml_trading_bot $SYM pid $PID not found — already stopped" >> "$REPO/logs/cron.log"
            echo "  ml_trading_bot $SYM (pid $PID) already stopped"
        fi
        rm -f "$PIDFILE"
    else
        # Fallback: kill by pattern if no pidfile
        if pgrep -f "ml_trading_bot.py --symbol $SYM" > /dev/null 2>&1; then
            pkill -f "ml_trading_bot.py --symbol $SYM"
            echo "$(date): stopped ml_trading_bot $SYM (by pattern)" >> "$REPO/logs/cron.log"
            echo "  Stopped ml_trading_bot $SYM (no pidfile, killed by pattern)"
        else
            echo "  ml_trading_bot $SYM not running"
        fi
    fi
done
