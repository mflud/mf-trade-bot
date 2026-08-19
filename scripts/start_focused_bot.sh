#!/bin/bash
# Start trading_bot with PL_REV, Wall Break, ORB strategies (VWASLR disabled 2026-08-19).
# Logs to logs/focused_bot.log; PID tracked in logs/trading_bot.pid (shared).
# Guards: skips before 08:25 ET; waits up to 5 min for a fresh bar_collector token.
REPO=/Users/marek/mf-trade-bot
PIDFILE="$REPO/logs/trading_bot.pid"
LOGFILE="$REPO/logs/focused_bot.log"
PYTHON=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
STRATEGIES="pl_rev,wall,orb"  # vwaslr disabled 2026-08-19: -700pts over 75 days, no edge in current conditions

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

# Don't start before 08:25 ET — bar_collector isn't logged in yet and any start
# attempt would use a stale token and silently fail.
hhmm=$(TZ=America/New_York date +%H%M)
hhmm=$((10#$hhmm))
if [[ $hhmm -lt 825 ]]; then
    echo "$(date): too early to start focused_bot (ET=$(TZ=America/New_York date +%H:%M), need ≥08:25) — skipping" >> "$REPO/logs/cron.log"
    exit 0
fi

# Check PID file first (most reliable).
if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "$(date): trading_bot already running (pid $PID) — skipping" >> "$REPO/logs/cron.log"
        exit 0
    fi
fi
# Fallback: scan for any trading_bot process (e.g. manually started without the script).
# Use --strategies in the pattern to avoid matching Claude's shell wrapper.
RUNNING_PID=$(pgrep -f "trading_bot.py --strategies" 2>/dev/null | head -1)
if [ -n "$RUNNING_PID" ]; then
    echo "$RUNNING_PID" > "$PIDFILE"   # adopt it so future checks use the PID file
    echo "$(date): trading_bot already running (pid $RUNNING_PID, adopted into pidfile) — skipping" >> "$REPO/logs/cron.log"
    exit 0
fi

# Wait for bar_collector to write a fresh token (max 5 min).
# bar_collector doesn't log in until 08:30 ET — if the bot starts before that,
# the token on disk is stale and the first API call will 401.
TOKEN="$REPO/data/auth_token.txt"
for i in $(seq 1 300); do
    if [ -f "$TOKEN" ]; then
        AGE=$(( $(date +%s) - $(stat -f %m "$TOKEN") ))
        if [ "$AGE" -lt 300 ]; then
            break
        fi
    fi
    sleep 1
done

cd "$REPO"
nohup "$PYTHON" -u src/trading_bot.py --strategies "$STRATEGIES" "$@" >> "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
echo "$(date): started focused_bot (pid $!) strategies=$STRATEGIES args: $*" >> "$REPO/logs/cron.log"
