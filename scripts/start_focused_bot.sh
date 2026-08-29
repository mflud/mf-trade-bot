#!/bin/bash
# Start trading_bot with ORB + BA-BRK strategies only (switched 2026-08-29 to
# closely monitor the new BA-BRK wall-cascade signal; PL_REV/Wall Break/VWASLR
# disabled here — re-add to STRATEGIES if going back to the prior mix).
# Logs to logs/focused_bot.log; PID tracked in logs/trading_bot.pid (shared).
# Guards: skips before 08:45 ET; waits up to 60s for bar_collector to be actively writing bars.
REPO=/Users/marek/mf-trade-bot
PIDFILE="$REPO/logs/trading_bot.pid"
LOGFILE="$REPO/logs/focused_bot.log"
PYTHON=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
STRATEGIES="orb,ba_brk"  # 2026-08-29: focused on ORB + BA-BRK only for close monitoring

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

# Don't start before 08:45 ET. Use Python for a DST-safe check (TZ= is unreliable in launchd).
et_hhmm=$("$PYTHON" -c "
from datetime import datetime
from zoneinfo import ZoneInfo
now = datetime.now(ZoneInfo('America/New_York'))
print(f'{now.hour:02d}{now.minute:02d}')
" 2>/dev/null)
et_hhmm=$((10#${et_hhmm:-0000}))
if [[ $et_hhmm -lt 845 ]]; then
    echo "$(date): too early to start focused_bot (ET=$(TZ=America/New_York date +%H:%M 2>/dev/null || date -u +%H:%M)UTC, need ≥08:45 ET) — skipping" >> "$REPO/logs/cron.log"
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

# Require bar_collector to be actively running (dom.db updated within last 30s).
# dom.db is written every few seconds by bar_collector — this is a reliable liveness
# signal, unlike auth_token.txt which can be stale from a prior session.
# Wait up to 60s; if not ready, skip — watchdog will retry in 5 minutes.
DOM_DB="$REPO/data/dom.db"
DOM_OK=0
for i in $(seq 1 60); do
    if [ -f "$DOM_DB" ]; then
        DOM_AGE=$(( $(date +%s) - $(stat -f %m "$DOM_DB") ))
        if [ "$DOM_AGE" -lt 30 ]; then
            DOM_OK=1
            break
        fi
    fi
    sleep 1
done
if [[ $DOM_OK -eq 0 ]]; then
    echo "$(date): bar_collector not yet live (dom.db absent or stale) — skipping (will retry)" >> "$REPO/logs/cron.log"
    exit 0
fi

cd "$REPO"
nohup "$PYTHON" -u src/trading_bot.py --strategies "$STRATEGIES" "$@" >> "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
echo "$(date): started focused_bot (pid $!) strategies=$STRATEGIES args: $*" >> "$REPO/logs/cron.log"
