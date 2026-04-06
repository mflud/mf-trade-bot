#!/bin/bash
# Start trading_bot as a background process (no display).
# Logs to logs/trading_bot.log; PID tracked in logs/trading_bot.pid.
REPO=/Users/marek/mf-trade-bot
PIDFILE="$REPO/logs/trading_bot.pid"
LOGFILE="$REPO/logs/trading_bot.log"
PYTHON=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3

mkdir -p "$REPO/logs"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "$(date): trading_bot already running (pid $(cat "$PIDFILE")) — skipping" >> "$REPO/logs/cron.log"
    exit 0
fi

cd "$REPO"
nohup "$PYTHON" src/trading_bot.py >> "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
echo "$(date): started trading_bot (pid $!)" >> "$REPO/logs/cron.log"
