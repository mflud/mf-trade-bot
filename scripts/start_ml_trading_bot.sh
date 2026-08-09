#!/bin/bash
# Start ml_trading_bot for both MES and MNQ as background processes.
REPO=/Users/marek/mf-trade-bot
PYTHON=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3

mkdir -p "$REPO/logs"

# Wait for bar_collector to write a fresh token (max 60s)
TOKEN="$REPO/data/auth_token.txt"
for i in $(seq 1 60); do
    if [ -f "$TOKEN" ]; then
        AGE=$(( $(date +%s) - $(stat -f %m "$TOKEN") ))
        if [ "$AGE" -lt 300 ]; then
            break
        fi
    fi
    sleep 1
done

cd "$REPO"

for SYM in MES MNQ; do
    PIDFILE="$REPO/logs/ml_trading_bot_${SYM}.pid"
    LOGFILE="$REPO/logs/ml_trading_bot_${SYM}.log"

    if pgrep -f "ml_trading_bot.py --symbol $SYM" > /dev/null 2>&1; then
        echo "$(date): ml_trading_bot $SYM already running — skipping" >> "$REPO/logs/cron.log"
        echo "  ml_trading_bot $SYM already running — skipping"
        continue
    fi

    nohup "$PYTHON" src/ml_trading_bot.py --symbol "$SYM" >> "$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"
    echo "$(date): started ml_trading_bot $SYM (pid $!)" >> "$REPO/logs/cron.log"
    echo "  Started ml_trading_bot $SYM (pid $!)"
done
