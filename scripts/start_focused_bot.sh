#!/bin/bash
# Start trading_bot with VWASLR, PL_REV, Wall Break, ORB strategies only.
# Logs to logs/focused_bot.log; PID tracked in logs/trading_bot.pid (shared).
# Waits up to 60s for a fresh bar_collector auth token before launching.
REPO=/Users/marek/mf-trade-bot
PIDFILE="$REPO/logs/trading_bot.pid"
LOGFILE="$REPO/logs/focused_bot.log"
PYTHON=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
STRATEGIES="vwaslr,pl_rev,wall,orb"

mkdir -p "$REPO/logs"

# Use a lock file so concurrent invocations (watchdog + launchd, or manual + watchdog)
# don't race past the pgrep check simultaneously.
LOCKFILE="$REPO/logs/start_focused_bot.lock"
if [ -e "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
    echo "$(date): start_focused_bot already in progress — skipping" >> "$REPO/logs/cron.log"
    exit 0
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

if pgrep -f "src/trading_bot.py" > /dev/null 2>&1; then
    echo "$(date): trading_bot already running (pid $(pgrep -f 'src/trading_bot.py' | tr '\n' ' ')) — skipping" >> "$REPO/logs/cron.log"
    exit 0
fi

# Wait for bar_collector to write a fresh token (max 60s).
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
nohup "$PYTHON" src/trading_bot.py --strategies "$STRATEGIES" "$@" >> "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
echo "$(date): started focused_bot (pid $!) strategies=$STRATEGIES args: $*" >> "$REPO/logs/cron.log"
