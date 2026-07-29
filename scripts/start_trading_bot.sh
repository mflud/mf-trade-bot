#!/bin/bash
# Start trading_bot as a background process (no display).
# Logs to logs/trading_bot.log; PID tracked in logs/trading_bot.pid.
# Waits up to 60s for a fresh bar_collector auth token before launching.
REPO=/Users/marek/mf-trade-bot
PIDFILE="$REPO/logs/trading_bot.pid"
LOGFILE="$REPO/logs/trading_bot.log"
PYTHON=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3

mkdir -p "$REPO/logs"

if pgrep -f "src/trading_bot.py" > /dev/null 2>&1; then
    echo "$(date): trading_bot already running (pid $(pgrep -f 'src/trading_bot.py' | tr '\n' ' ')) — skipping" >> "$REPO/logs/cron.log"
    exit 0
fi

# Wait for bar_collector to write a fresh token (max 60s).
# Without this, the bot exits immediately when the prior session's token is stale.
TOKEN="$REPO/data/auth_token.txt"
for i in $(seq 1 60); do
    if [ -f "$TOKEN" ]; then
        AGE=$(( $(date +%s) - $(stat -f %m "$TOKEN") ))
        if [ "$AGE" -lt 300 ]; then   # token < 5 min old = fresh enough
            break
        fi
    fi
    sleep 1
done

cd "$REPO"
nohup "$PYTHON" src/trading_bot.py >> "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
echo "$(date): started trading_bot (pid $!)" >> "$REPO/logs/cron.log"
